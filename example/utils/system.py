"""Build from SDF using OpenFF, AmberTools and Packmol; add ML/MM interpolation."""

import numpy as np
import openmm as mm
from openmm import unit
from openff.interchange import Interchange
from openff.interchange.components._packmol import RHOMBIC_DODECAHEDRON, pack_box
from openff.toolkit import ForceField, Molecule
from openff.toolkit import unit as ffunit


def build_system(sdf_file, atomic_numbers, n_waters, box_length_angstrom, platform, properties):
    """Pack water, assign AM1-BCC/OpenFF parameters, and minimize the MM system."""
    ligand = Molecule(str(sdf_file))
    if not np.array_equal([atom.atomic_number for atom in ligand.atoms], atomic_numbers):
        raise ValueError("SDF atom order must match the trained model's nuclear_charges.txt")
    if ligand.total_charge.m_as(ffunit.elementary_charge) != 0:
        raise ValueError("This example requires neutral caffeine")
    ligand.assign_partial_charges("am1bcc", use_conformers=ligand.conformers)
    water = Molecule.from_smiles("[H][O][H]")
    for atom in water.atoms:
        atom.metadata["residue_name"] = "HOH"
    topology = pack_box(molecules=[ligand, water], number_of_copies=[1, n_waters],
                        box_vectors=box_length_angstrom * RHOMBIC_DODECAHEDRON * ffunit.angstrom)
    interchange = Interchange.from_smirnoff(
        force_field=ForceField("openff_unconstrained-2.1.0.offxml", "tip3p.offxml"),
        topology=topology, charge_from_molecules=[ligand])
    system = interchange.to_openmm_system()
    nonbonded = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1 * unit.nanometer)
    nonbonded.setUseDispersionCorrection(True)
    # Relax packing contacts before starting ML/MM dynamics.
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    context.setPositions(interchange.positions.to_openmm())
    context.applyConstraints(1e-6)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=500)
    state = context.getState(getPositions=True)
    initial = {"positions_nm": state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
               "box_nm": state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)}
    return system, initial, interchange.to_openmm_topology()


def add_ml_forces(system, n_solute):
    """Scale ligand MM terms by (1-lambda); inject the ML and response forces."""
    nonbonded = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    parameters = [nonbonded.getParticleParameters(i) for i in range(n_solute)]
    exceptions = {}
    for index in range(nonbonded.getNumExceptions()):
        i, j, *values = nonbonded.getExceptionParameters(index)
        exceptions[tuple(sorted((i, j)))] = values
    internal = mm.CustomBondForce("138.9354558466661*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)")
    for name in ("chargeProd", "sigma", "epsilon"):
        internal.addPerBondParameter(name)
    for i in range(n_solute):
        for j in range(i):
            q1, s1, e1 = parameters[i]
            q2, s2, e2 = parameters[j]
            values = exceptions.get((j, i), (q1*q2, (s1+s2)/2, (e1*e2)**0.5))
            internal.addBond(i, j, values)
            nonbonded.addException(i, j, 0, 1, 0, True)
    ligand_terms = [internal]
    # Keep zeroed ligand bonds in the outer System to preserve molecule grouping.
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            selected = mm.HarmonicBondForce()
            for index in range(force.getNumBonds()):
                i, j, length, k = force.getBondParameters(index)
                if max(i, j) < n_solute:
                    selected.addBond(i, j, length, k)
                    force.setBondParameters(index, i, j, length, 0)
        elif isinstance(force, mm.HarmonicAngleForce):
            selected = mm.HarmonicAngleForce()
            for index in range(force.getNumAngles()):
                i, j, k, angle, stiffness = force.getAngleParameters(index)
                if max(i, j, k) < n_solute:
                    selected.addAngle(i, j, k, angle, stiffness)
                    force.setAngleParameters(index, i, j, k, angle, 0)
        elif isinstance(force, mm.PeriodicTorsionForce):
            selected = mm.PeriodicTorsionForce()
            for index in range(force.getNumTorsions()):
                i, j, k, l, periodicity, phase, stiffness = force.getTorsionParameters(index)
                if max(i, j, k, l) < n_solute:
                    selected.addTorsion(i, j, k, l, periodicity, phase, stiffness)
                    force.setTorsionParameters(index, i, j, k, l, periodicity, phase, 0)
        else:
            continue
        ligand_terms.append(selected)
    names = [f"mm_{i}" for i in range(len(ligand_terms))]
    interpolation = mm.CustomCVForce(f"(1-lambda_interpolate)*({'+'.join(names)})")
    interpolation.addGlobalParameter("lambda_interpolate", 0)
    for name, force in zip(names, ligand_terms):
        interpolation.addCollectiveVariable(name, force)
    system.addForce(interpolation)
    injected = mm.CustomExternalForce("-fx*x-fy*y-fz*z")
    injected.setForceGroup(31)
    for name in ("fx", "fy", "fz"):
        injected.addPerParticleParameter(name)
    for i in range(n_solute):
        injected.addParticle(i, (0, 0, 0))
    system.addForce(injected)
    return system
