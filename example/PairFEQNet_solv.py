"""Caffeine in water: NPT sampling with the full dynamic-charge response force."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
import openmm as mm
from openmm import app, unit
import tensorflow as tf

from utils.network import Molecule, Network
from utils.charge_response import ChargeResponse
from utils.barostat import MolecularBarostat, KB
from utils.system import build_system, add_ml_forces


# User settings. Inputs are resolved relative to this script.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "trained_model"
INPUT_FILE = ROOT / "input.sdf"
OUTPUT_DIR = ROOT / "md_output"
N_WATERS = 1000
BOX_LENGTH_ANGSTROM = 35.0
LAMBDA_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CHARGE_SCALING = 1.2
TEMPERATURE_K = 300.0
PRESSURE_BAR = 1.0
FRICTION_PS = 5.0
TIMESTEP_FS = 1.0
EQUILIBRATION_PS = 50.0
PRODUCTION_PS = 3000.0
SAVE_INTERVAL_PS = 1.0
BAROSTAT_INTERVAL = 50
RANDOM_SEED = 20261104
PLATFORM = "CPU"  # Set to "CUDA" for an NVIDIA GPU.
INJECTED_GROUP = 31
PHYSICAL_GROUPS = ((1 << 32) - 1) ^ (1 << INJECTED_GROUP)
KCAL = 4.184


def platform_properties(platform):
    return ({"Precision": "mixed", "DeterministicForces": "true", "UseCpuPme": "false"}
            if platform == "CUDA" else {"Threads": "4", "DeterministicForces": "true"})


def run_window(model, z, system_xml, initial, lambda_value, timestep_fs, platform,
               equilibration_steps, production_steps, save_every, output, seed):
    output.mkdir()
    system = mm.XmlSerializer.deserialize(system_xml)
    if not any(isinstance(force, mm.CMMotionRemover) for force in system.getForces()):
        system.addForce(mm.CMMotionRemover())
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    injected = next(f for f in system.getForces() if isinstance(f, mm.CustomExternalForce))
    integrator = mm.LangevinIntegrator(TEMPERATURE_K * unit.kelvin,
                                       FRICTION_PS / unit.picosecond, timestep_fs * unit.femtosecond)
    integrator.setConstraintTolerance(1e-6)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), platform_properties(platform))
    context.setParameter("lambda_interpolate", lambda_value)
    context.setPeriodicBoxVectors(*initial["box_nm"])
    context.setPositions(initial["positions_nm"])
    context.applyConstraints(1e-6)
    response = ChargeResponse(model, z, nonbonded, context, CHARGE_SCALING)
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3

    def coordinates():
        state = context.getState(getPositions=True)
        return (state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))

    def physical_energy(ml_energy=None):
        if ml_energy is None:
            positions, _ = coordinates()
            _, ml_energy, _, _ = response.prediction(positions, lambda_value)
        state = context.getState(getEnergy=True, groups=PHYSICAL_GROUPS)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) + lambda_value * ml_energy

    def update():
        positions, box = coordinates()
        ml_energy, forces, charges = response.evaluate(positions, box, lambda_value)
        for i, force in enumerate(forces):
            injected.setParticleParameters(i, i, force)
        injected.updateParametersInContext(context)
        return positions, box, ml_energy, charges

    update()  # Validate the PME response before any integration.
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, seed + 10000)
    context.applyVelocityConstraints(1e-6)
    barostat = MolecularBarostat(context, physical_energy, TEMPERATURE_K, PRESSURE_BAR,
                                 nonbonded.getCutoffDistance().value_in_unit(unit.nanometer), seed + 50000)
    n_frames = production_steps // save_every + 1
    trajectory = np.lib.format.open_memmap(output / "coordinates.npy", mode="w+",
        dtype="float64", shape=(n_frames, system.getNumParticles(), 3))
    boxes = np.lib.format.open_memmap(output / "box_vectors.npy", mode="w+",
        dtype="float64", shape=(n_frames, 3, 3))
    charge_history = np.lib.format.open_memmap(output / "charges.npy", mode="w+",
        dtype="float64", shape=(n_frames, len(z)))
    total_steps = equilibration_steps + production_steps
    with (output / "energy.csv").open("w", newline="", buffering=1) as handle:
        writer = csv.writer(handle)
        writer.writerow(["production_time_ps", "potential_kcal_mol", "kinetic_kcal_mol",
                         "total_kcal_mol", "temperature_K", "volume_nm3", "mc_accepted", "mc_attempts"])
        for step in range(total_steps + 1):
            if step > 0 and step % BAROSTAT_INTERVAL == 0:
                barostat.attempt()
            # Every accepted/rejected MC move is followed by a full force refresh.
            positions, box, ml_energy, charges = update()
            production_step = step - equilibration_steps
            if production_step >= 0 and production_step % save_every == 0:
                frame = production_step // save_every
                trajectory[frame], boxes[frame], charge_history[frame] = positions, box, charges
                potential = physical_energy(ml_energy)
                kinetic = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                writer.writerow([production_step * timestep_fs / 1000, potential / KCAL, kinetic / KCAL,
                                 (potential + kinetic) / KCAL, 2 * kinetic / (KB * dof),
                                 np.linalg.det(box), barostat.accepted, barostat.attempts])
            if step < total_steps:
                integrator.step(1)
        trajectory.flush()
        boxes.flush()
        charge_history.flush()
    final = context.getState(getPositions=True, getVelocities=True, getParameters=True)
    (output / "final_state.xml").write_text(mm.XmlSerializer.serialize(final))
    print(f"lambda={lambda_value:g}: finished NPT; MC accepted {barostat.accepted}/{barostat.attempts}; {output}", flush=True)


def run(duration_ps=PRODUCTION_PS, equilibration_ps=EQUILIBRATION_PS, timestep_fs=TIMESTEP_FS,
        lambda_values=LAMBDA_SCHEDULE, platform=PLATFORM, output=OUTPUT_DIR, save_interval_ps=SAVE_INTERVAL_PS):
    dt = timestep_fs / 1000
    intervals = np.asarray([equilibration_ps, duration_ps, save_interval_ps]) / dt if dt > 0 else np.array([np.nan])
    if (not np.isfinite(intervals).all() or np.any(intervals < 0) or intervals[1] < 1 or intervals[2] < 1
            or not np.allclose(intervals, np.rint(intervals), rtol=0, atol=1e-7)):
        raise ValueError("Times must be integer multiples of a positive timestep; production/save times must be positive")
    if not lambda_values or len(set(lambda_values)) != len(lambda_values) or any(not 0 <= x <= 1 for x in lambda_values):
        raise ValueError("Provide unique lambda values between 0 and 1")
    equilibration_steps, production_steps, save_every = np.rint(intervals).astype(int)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    z = np.loadtxt(MODEL_DIR / "nuclear_charges.txt", dtype=np.float32)
    molecule = Molecule()
    molecule.n_atom = len(z)
    model = Network(molecule).load(molecule, MODEL_DIR)
    print(f"Building caffeine + {N_WATERS} waters from {INPUT_FILE.name}", flush=True)
    system, initial, topology = build_system(INPUT_FILE, z, N_WATERS, BOX_LENGTH_ANGSTROM,
                                            platform, platform_properties(platform))
    with (output / "initial.pdb").open("w") as handle:
        app.PDBFile.writeFile(topology, initial["positions_nm"] * unit.nanometer, handle)
    system_xml = mm.XmlSerializer.serialize(add_ml_forces(system, len(z)))
    for index, lambda_value in enumerate(lambda_values):
        run_window(model, z, system_xml, initial, lambda_value, timestep_fs, platform,
                   equilibration_steps, production_steps, save_every,
                   output / f"window_{index:02d}_lambda_{lambda_value:g}", RANDOM_SEED + index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-ps", type=float, default=PRODUCTION_PS)
    parser.add_argument("--equilibration-ps", type=float, default=EQUILIBRATION_PS)
    parser.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS)
    parser.add_argument("--save-interval-ps", type=float, default=SAVE_INTERVAL_PS)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=LAMBDA_SCHEDULE)
    parser.add_argument("--platform", choices=("CPU", "CUDA"), default=PLATFORM)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    run(**vars(parser.parse_args()))
