"""
PairFEQ-Net alchemical simulation with dynamic CM5 charge prediction.

The ligand intramolecular potential is interpolated from the classical force
field to PairFEQ-Net over a series of lambda windows. PairFEQ-Net predicts the
ligand forces, energy, and partial charges at every MD step.

Author: Zihan Kong <zihan.kong-2@postgrad.manchester.ac.uk>
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import openmm.unit as mmunit
import tensorflow as tf
from openff.interchange import Interchange
from openff.interchange.components._packmol import (
    RHOMBIC_DODECAHEDRON,
    pack_box,
)
from openff.toolkit import ForceField, Molecule
from openff.toolkit import unit as ffunit

from utils import alchemical
from utils.network import Molecule as PairNetMolecule, Network


# Paths are relative to the example directory.
INPUT_FILE = Path("input.sdf")
MODEL_DIR = Path("trained_model")
OUTPUT_DIR = Path("md_output")
TRAJECTORY_FILE = Path("solv_coords_12.npz")

# System construction.
N_WATER_MOLECULES = 1_000
BOX_LENGTH_ANGSTROM = 35

# Simulation protocol.
LAMBDA_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TEMPERATURE = 300 * mmunit.kelvin
FRICTION = 5.0 / mmunit.picoseconds
TIMESTEP = 1.0 * mmunit.femtoseconds
EQUILIBRATION_TIME = 50 * mmunit.picoseconds
PRODUCTION_TIME = 3.0 * mmunit.nanoseconds
SAVE_INTERVAL = 1_000
CHARGE_SCALING = 1.2

# Runtime safeguards and OpenMM settings.
FORCE_STABILITY_LIMIT = 1_000.0
TF_SESSION_CLEAR_INTERVAL = 1_000
NONBONDED_CUTOFF = 1.0 * mmunit.nanometer
INITIAL_VELOCITY_TEMPERATURE = 50 * mmunit.kelvin
BAROSTAT_PRESSURE = 1 * mmunit.bar
BAROSTAT_INTERVAL = 50


@dataclass
class LigandNonbondedParameters:
    """Mutable ligand parameters used for on-the-fly charge updates."""

    force: object
    current_charges: np.ndarray
    original_charges: np.ndarray
    sigmas: np.ndarray
    epsilons: np.ndarray


def predict_charges(prediction, n_atoms, charge_scaling):
    """Correct the predicted net charge and apply the requested scaling."""
    ligand_charges = prediction[2].T
    net_charge = round(sum(ligand_charges[:, 0]))
    correction = (sum(ligand_charges) - net_charge) / n_atoms

    for atom_index in range(n_atoms):
        ligand_charges[atom_index] -= correction
        ligand_charges[atom_index] *= charge_scaling

    return ligand_charges


def _prepare_alchemical_system(
    system,
    ligand_indices,
    lambda_value,
    temperature,
):
    """Configure nonbonded terms and add the ML/MM interpolation forces."""
    nonbonded = next(
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    )
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(NONBONDED_CUTOFF)
    nonbonded.setUseDispersionCorrection(True)

    n_ligand_atoms = len(ligand_indices)
    parameters = LigandNonbondedParameters(
        force=nonbonded,
        current_charges=np.zeros(n_ligand_atoms),
        original_charges=np.zeros(n_ligand_atoms),
        sigmas=np.zeros(n_ligand_atoms),
        epsilons=np.zeros(n_ligand_atoms),
    )
    for local_index, global_index in enumerate(ligand_indices):
        charge, sigma, epsilon = nonbonded.getParticleParameters(global_index)
        charge_value = charge.value_in_unit(mmunit.elementary_charge)
        parameters.current_charges[local_index] = charge_value
        parameters.original_charges[local_index] = charge_value
        parameters.sigmas[local_index] = sigma.value_in_unit(mmunit.nanometer)
        parameters.epsilons[local_index] = epsilon.value_in_unit(
            mmunit.kilojoule_per_mole
        )

    system = alchemical.new_nb_force(system, False, ligand_indices)
    system, ml_force = alchemical.pairnet_force(ligand_indices, system)
    system, _ = alchemical.ml_interpolation(
        system,
        ligand_indices,
        lambda_value,
    )
    system = alchemical.remove_nb(ligand_indices, system)
    system = alchemical.remove_bond(system, ligand_indices)
    system.addForce(
        openmm.MonteCarloBarostat(
            BAROSTAT_PRESSURE,
            temperature,
            BAROSTAT_INTERVAL,
        )
    )
    return system, ml_force, parameters


def _load_pairnet_model():
    """Load ligand atom types and reconstruct the trained inference model."""
    ligand_atom_types = np.loadtxt(
        MODEL_DIR / "nuclear_charges.txt",
        dtype=np.float32,
    ).reshape(-1)

    tf.config.set_visible_devices([], "GPU") # use CPU to save I/O memory
    tf.get_logger().setLevel("ERROR")

    molecule = PairNetMolecule()
    molecule.n_atom = len(ligand_atom_types)
    keras_model = Network(molecule).load(molecule, MODEL_DIR)

    compiled_inference = tf.function(
        lambda coordinates, atom_types: keras_model(
            [coordinates, atom_types],
            training=False,
        ),
        input_signature=[
            tf.TensorSpec(
                shape=(1, len(ligand_atom_types), 3),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(1, len(ligand_atom_types)),
                dtype=tf.float32,
            ),
        ],
    )
    return ligand_atom_types, compiled_inference


def _predict_and_update(
    model,
    simulation,
    ligand_indices,
    ligand_atom_types,
    ml_force,
    nonbonded_parameters,
    lambda_value,
    charge_scaling,
    phase,
    step,
):
    """Predict and apply PairFEQ-Net forces and charges for one MD step."""
    if step % TF_SESSION_CLEAR_INTERVAL == 0:
        tf.keras.backend.clear_session()

    coordinates = (
        simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .in_units_of(mmunit.angstrom)
    )
    coordinate_tensor = tf.convert_to_tensor(
        np.reshape(
            coordinates[ligand_indices] / mmunit.angstrom,
            (1, -1, 3),
        ),
        dtype=tf.float32,
    )
    atom_type_tensor = tf.convert_to_tensor(
        np.reshape(ligand_atom_types, (1, -1)),
        dtype=tf.float32,
    )
    prediction = [
        output.numpy()
        for output in model(coordinate_tensor, atom_type_tensor)
    ]

    if np.any(prediction[0] >= FORCE_STABILITY_LIMIT):
        print(
            f"Error: predicted force exceeded the stability threshold during "
            f"{phase} at step {step}.",
            flush=True,
        )
        print("Final coordinates (angstrom):", flush=True)
        print(
            np.reshape(
                coordinates[ligand_indices] / mmunit.angstrom,
                (1, -1, 3),
            ),
            flush=True,
        )
        print(f"Final energy: {prediction[1][0][0]}", flush=True)
        return None

    ml_forces = np.reshape(
        prediction[0]
        * mmunit.kilocalories_per_mole
        / mmunit.angstrom,
        (-1, 3),
    )
    ml_forces = lambda_value * ml_forces
    for local_index, global_index in enumerate(ligand_indices):
        ml_force.setParticleParameters(
            local_index,
            global_index,
            tuple(ml_forces[local_index]),
        )
    ml_force.updateParametersInContext(simulation.context)

    predicted_charges = predict_charges(
        prediction,
        len(ligand_indices),
        charge_scaling,
    )
    for local_index, global_index in enumerate(ligand_indices):
        predicted_charge = np.squeeze(predicted_charges[local_index]).item()
        updated_charge = (
            predicted_charge * lambda_value
            + nonbonded_parameters.original_charges[local_index]
            * (1 - lambda_value)
        )
        nonbonded_parameters.force.setParticleParameters(
            global_index,
            updated_charge,
            nonbonded_parameters.sigmas[local_index] * mmunit.nanometer,
            nonbonded_parameters.epsilons[local_index]
            * mmunit.kilojoule_per_mole,
        )
        nonbonded_parameters.current_charges[local_index] = updated_charge
    nonbonded_parameters.force.updateParametersInContext(simulation.context)

    return prediction, ml_forces


def _save_sample(
    simulation,
    lambda_value,
    prediction,
    ml_forces,
    ligand_indices,
    charges,
):
    """Write one production sample and return its positions and box vectors."""
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions().value_in_unit(mmunit.nanometer)
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        mmunit.nanometer
    )
    total_forces = (
        simulation.context.getState(getForces=True)
        .getForces(asNumpy=True)
        .in_units_of(mmunit.kilocalories_per_mole / mmunit.angstrom)
    )
    ligand_total_forces = total_forces[ligand_indices]
    mm_forces = ligand_total_forces - ml_forces

    with (OUTPUT_DIR / f"energies_{lambda_value}.txt").open("a") as handle:
        handle.write(f"{prediction[1][0][0]:.8f}\n")
    with (OUTPUT_DIR / f"MMforces_{lambda_value}.txt").open("a") as handle:
        np.savetxt(handle, mm_forces)
    with (OUTPUT_DIR / f"totalforces_{lambda_value}.txt").open("a") as handle:
        np.savetxt(handle, ligand_total_forces)
    with (OUTPUT_DIR / f"MLforces_{lambda_value}.txt").open("a") as handle:
        np.savetxt(handle, ml_forces)
    with (OUTPUT_DIR / f"charges_{lambda_value}.txt").open("a") as handle:
        np.savetxt(handle, charges)

    return positions, box_vectors


def run_mix(
    system,
    ligand_indices,
    lambda_value,
    temperature,
    friction,
    timestep,
    charge_scaling,
    topology,
    starting_positions,
    n_equilibration_steps,
    n_production_steps,
    n_saved_frames,
    n_atoms,
    save_interval,
    window_index,
):
    """Run one alchemical lambda window and return its sampled trajectory."""
    system, ml_force, nonbonded_parameters = _prepare_alchemical_system(
        system,
        ligand_indices,
        lambda_value,
        temperature,
    )

    integrator = openmm.LangevinIntegrator(temperature, friction, timestep)
    simulation = openmm.app.Simulation(topology, system, integrator)
    simulation.context.setPositions(starting_positions)
    simulation.context.setVelocitiesToTemperature(INITIAL_VELOCITY_TEMPERATURE)
    alchemical.print_information(system, ligand_indices, integrator)

    ligand_atom_types, model = _load_pairnet_model()

    for step in range(n_equilibration_steps):
        result = _predict_and_update(
            model,
            simulation,
            ligand_indices,
            ligand_atom_types,
            ml_force,
            nonbonded_parameters,
            lambda_value,
            charge_scaling,
            "equilibration",
            step,
        )
        if result is None:
            break
        simulation.step(1)

    print("Equilibration finished; starting production.", flush=True)
    sample_index = 0
    sampled_coordinates = np.zeros((n_saved_frames, n_atoms, 3))
    sampled_box_vectors = np.zeros((n_saved_frames, 3, 3))

    for step in range(n_production_steps):
        result = _predict_and_update(
            model,
            simulation,
            ligand_indices,
            ligand_atom_types,
            ml_force,
            nonbonded_parameters,
            lambda_value,
            charge_scaling,
            "production",
            step,
        )
        if result is None:
            break

        prediction, ml_forces = result
        if (step + 1) % save_interval == 0:
            (
                sampled_coordinates[sample_index],
                sampled_box_vectors[sample_index],
            ) = _save_sample(
                simulation,
                lambda_value,
                prediction,
                ml_forces,
                ligand_indices,
                nonbonded_parameters.current_charges,
            )
            sample_index += 1

        simulation.step(1)

    with (OUTPUT_DIR / f"old_charges_{lambda_value}.txt").open("a") as handle:
        np.savetxt(handle, nonbonded_parameters.original_charges)

    print(
        f"Collected {sample_index} samples in window {window_index + 1} "
        f"(lambda = {lambda_value}).",
        flush=True,
    )
    return sampled_coordinates, sampled_box_vectors


def _build_solvated_system():
    """Build the ligand/water topology and its OpenFF interchange object."""
    ligand = Molecule(str(INPUT_FILE))
    water = Molecule.from_smiles("[H][O][H]")
    for atom in water.atoms:
        atom.metadata["residue_name"] = "HOH"

    topology = pack_box(
        molecules=[ligand, water],
        number_of_copies=[1, N_WATER_MOLECULES],
        box_vectors=(
            BOX_LENGTH_ANGSTROM
            * RHOMBIC_DODECAHEDRON
            * ffunit.angstrom
        ),
    )
    ligand_molecule = topology.molecule(0)
    ligand_start = topology.molecule_atom_start_index(ligand_molecule)
    ligand_indices = list(
        range(ligand_start, ligand_start + ligand_molecule.n_atoms)
    )

    force_field = ForceField(
        "openff_unconstrained-2.1.0.offxml",
        "tip3p.offxml",
    )
    interchange = Interchange.from_smirnoff(
        force_field=force_field,
        topology=topology,
    )
    return topology, ligand_indices, interchange


def main():
    """Run every lambda window and save the combined trajectory."""
    topology, ligand_indices, interchange = _build_solvated_system()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    n_equilibration_steps = int(EQUILIBRATION_TIME / TIMESTEP)
    n_production_steps = int(PRODUCTION_TIME / TIMESTEP)
    n_saved_frames = n_production_steps // SAVE_INTERVAL
    n_windows = len(LAMBDA_SCHEDULE)
    n_atoms = topology.n_atoms
    starting_positions = interchange.positions.to_openmm()

    all_coordinates = np.zeros((n_windows, n_saved_frames, n_atoms, 3))
    all_box_vectors = np.zeros((n_windows, n_saved_frames, 3, 3))

    print(f"Running {n_windows} lambda windows.", flush=True)
    for window_index, lambda_value in enumerate(LAMBDA_SCHEDULE):
        coordinates, box_vectors = run_mix(
            interchange.to_openmm_system(),
            ligand_indices,
            lambda_value,
            TEMPERATURE,
            FRICTION,
            TIMESTEP,
            CHARGE_SCALING,
            interchange.to_openmm_topology(),
            starting_positions,
            n_equilibration_steps,
            n_production_steps,
            n_saved_frames,
            n_atoms,
            SAVE_INTERVAL,
            window_index,
        )
        all_coordinates[window_index] = coordinates
        all_box_vectors[window_index] = box_vectors

    np.savez_compressed(
        TRAJECTORY_FILE,
        coordinates=all_coordinates,
        box_vectors=all_box_vectors,
    )


if __name__ == "__main__":
    main()
