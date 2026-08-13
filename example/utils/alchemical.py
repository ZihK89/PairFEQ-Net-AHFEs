"""Minimal OpenMM alchemical-force helpers for PairFEQNet_solv.py."""

from __future__ import annotations

from copy import deepcopy

import openmm
import openmm.unit as mmunit


def _should_remove(term_atoms, atom_set, remove_in_set):
    """Return whether a bonded term belongs to the selected side of the system."""
    return all(atom in atom_set for atom in term_atoms) == remove_in_set


def _remove_bonded_interactions(force, ligand_indices, remove_in_set):
    """Zero bonded interactions either inside or outside the ligand atom set."""
    atom_set = set(ligand_indices)

    if hasattr(force, "addBond"):
        for index in range(force.getNumBonds()):
            try:
                atom1, atom2, length, force_constant = force.getBondParameters(index)
                if _should_remove((atom1, atom2), atom_set, remove_in_set):
                    force.setBondParameters(index, atom1, atom2, length, 0)
            except ValueError:
                atom1, atom2, parameters = force.getBondParameters(index)
                if _should_remove((atom1, atom2), atom_set, remove_in_set):
                    force.setBondParameters(
                        index,
                        atom1,
                        atom2,
                        [0 for _ in parameters],
                    )

    if hasattr(force, "addAngle"):
        for index in range(force.getNumAngles()):
            atom1, atom2, atom3, angle, force_constant = force.getAngleParameters(
                index
            )
            if _should_remove((atom1, atom2, atom3), atom_set, remove_in_set):
                force.setAngleParameters(index, atom1, atom2, atom3, angle, 0)

    if hasattr(force, "addTorsion"):
        for index in range(force.getNumTorsions()):
            (
                atom1,
                atom2,
                atom3,
                atom4,
                periodicity,
                phase,
                force_constant,
            ) = force.getTorsionParameters(index)
            if _should_remove(
                (atom1, atom2, atom3, atom4), atom_set, remove_in_set
            ):
                force.setTorsionParameters(
                    index,
                    atom1,
                    atom2,
                    atom3,
                    atom4,
                    periodicity,
                    phase,
                    0,
                )

    return force


def print_information(system, ligand_indices, integrator):
    """Print the force-field settings used by the simulation."""
    water_constraints = []
    ligand_constraints = []
    for index in range(system.getNumConstraints()):
        atom1, atom2, distance = system.getConstraintParameters(index)
        target = (
            ligand_constraints
            if atom1 in ligand_indices or atom2 in ligand_indices
            else water_constraints
        )
        target.append((atom1, atom2, distance))

    print("===== System Information =====", flush=True)
    print(f"Total constraints: {system.getNumConstraints()}", flush=True)
    print(
        f" - Constraints involving ligand: {len(ligand_constraints)}", flush=True
    )
    print(
        " - Constraints involving solvent (e.g. TIP3P water OH bonds): "
        f"{len(water_constraints)}",
        flush=True,
    )
    if water_constraints:
        print(
            "   => Water is modeled as rigid (constraints enforce OH / HOH geometry).",
            flush=True,
        )
    else:
        print("   => Water is flexible (no constraints found).", flush=True)
    if ligand_constraints:
        print("Ligand has constraints (bonds constrained).", flush=True)
    else:
        print("Ligand has no constraints (all bonds flexible).", flush=True)

    nonbonded = next(
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    )
    method_names = {
        openmm.NonbondedForce.NoCutoff: "NoCutoff",
        openmm.NonbondedForce.CutoffNonPeriodic: "CutoffNonPeriodic",
        openmm.NonbondedForce.CutoffPeriodic: "CutoffPeriodic",
        openmm.NonbondedForce.Ewald: "Ewald",
        openmm.NonbondedForce.PME: "PME",
        openmm.NonbondedForce.LJPME: "LJPME",
    }
    print(
        f"Nonbonded method = {method_names.get(nonbonded.getNonbondedMethod(), 'Unknown')}",
        flush=True,
    )
    print(
        "Cutoff distance  = "
        f"{nonbonded.getCutoffDistance().value_in_unit(mmunit.nanometer):.3f} nm",
        flush=True,
    )
    print(
        f"Reaction field dielectric = {nonbonded.getReactionFieldDielectric():.2f}",
        flush=True,
    )
    print(
        f"Dispersion correction = {nonbonded.getUseDispersionCorrection()}",
        flush=True,
    )
    print(
        "Target temperature = "
        f"{integrator.getTemperature().value_in_unit(mmunit.kelvin):.2f} K",
        flush=True,
    )
    print("===== End of System Information =====\n", flush=True)


def new_nb_force(system, reaction_field, ligand_indices):
    """Copy ligand-ligand nonbonded terms into a CustomBondForce."""
    forces = {force.__class__.__name__: force for force in system.getForces()}
    nonbonded = forces["NonbondedForce"]

    if reaction_field:
        cutoff = nonbonded.getCutoffDistance()
        solvent_dielectric = nonbonded.getReactionFieldDielectric()
        krf = (
            (1 / cutoff**3)
            * (solvent_dielectric - 1)
            / (2 * solvent_dielectric + 1)
        )
        crf = (1 / cutoff) * (3 * solvent_dielectric) / (
            2 * solvent_dielectric + 1
        )
        energy_expression = (
            "138.9354558466661*chargeProd*(1/r + krf*r*r - crf) + "
            "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            f"krf = {krf.value_in_unit(mmunit.nanometer**-3)};"
            f"crf = {crf.value_in_unit(mmunit.nanometer**-1)}"
        )
    else:
        energy_expression = (
            "138.9354558466661*chargeProd/r + "
            "4*epsilon*((sigma/r)^12-(sigma/r)^6)"
        )

    internal_nonbonded = openmm.CustomBondForce(energy_expression)
    internal_nonbonded.addPerBondParameter("chargeProd")
    internal_nonbonded.addPerBondParameter("sigma")
    internal_nonbonded.addPerBondParameter("epsilon")

    particle_parameters = [
        nonbonded.getParticleParameters(index)
        for index in range(system.getNumParticles())
    ]
    exceptions = {}
    for index in range(nonbonded.getNumExceptions()):
        atom1, atom2, charge_product, sigma, epsilon = (
            nonbonded.getExceptionParameters(index)
        )
        exceptions[(atom1, atom2)] = (charge_product, sigma, epsilon)

    for atom1 in ligand_indices:
        for atom2 in ligand_indices:
            if atom1 == atom2:
                break
            if (atom1, atom2) in exceptions:
                charge_product, sigma, epsilon = exceptions[(atom1, atom2)]
            elif (atom2, atom1) in exceptions:
                charge_product, sigma, epsilon = exceptions[(atom2, atom1)]
            else:
                charge1, sigma1, epsilon1 = particle_parameters[atom1]
                charge2, sigma2, epsilon2 = particle_parameters[atom2]
                charge_product = charge1 * charge2
                sigma = 0.5 * (sigma1 + sigma2)
                epsilon = mmunit.sqrt(epsilon1 * epsilon2)
            if charge_product._value != 0 or epsilon._value != 0:
                internal_nonbonded.addBond(
                    atom1,
                    atom2,
                    [charge_product, sigma, epsilon],
                )

    system.addForce(internal_nonbonded)
    return system


def pairnet_force(ligand_indices, system):
    """Add the external force whose per-particle values are updated by PairFEQ-Net."""
    ml_force = openmm.CustomExternalForce("-fx*x-fy*y-fz*z")
    system.addForce(ml_force)
    ml_force.addPerParticleParameter("fx")
    ml_force.addPerParticleParameter("fy")
    ml_force.addPerParticleParameter("fz")
    ml_force.setName("PairNetML")
    for atom_index in ligand_indices:
        ml_force.addParticle(atom_index, (0, 0, 0))
    return system, ml_force


def ml_interpolation(system, ligand_indices, lambda_value):
    """Add the lambda-scaled MM contribution used with the external ML force."""
    cv_force = openmm.CustomCVForce("")
    cv_force.addGlobalParameter("lambda_interpolate", lambda_value)

    bonded_forces = []
    for force in system.getForces():
        if (
            hasattr(force, "addBond")
            or hasattr(force, "addAngle")
            or hasattr(force, "addTorsion")
        ):
            ligand_force = deepcopy(force)
            _remove_bonded_interactions(ligand_force, ligand_indices, False)
            bonded_forces.append(ligand_force)

    variable_names = []
    for index, force in enumerate(bonded_forces):
        name = f"{force.getName()}{index + 1}"
        cv_force.addCollectiveVariable(name, deepcopy(force))
        variable_names.append(name)

    mm_sum = "+".join(variable_names) if variable_names else "0"
    cv_force.setEnergyFunction(f"(1-lambda_interpolate)*({mm_sum})")
    cv_force.setName("MLInterpolationCV")
    system.addForce(cv_force)
    return system, cv_force


def remove_nb(ligand_indices, system):
    """Disable ligand-ligand terms in the original nonbonded force."""
    atom_list = list(ligand_indices)
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for index, atom1 in enumerate(atom_list):
                for atom2 in atom_list[:index]:
                    force.addException(atom1, atom2, 0, 1, 0, True)
        elif isinstance(force, openmm.CustomNonbondedForce):
            existing = {
                tuple(force.getExclusionParticles(index))
                for index in range(force.getNumExclusions())
            }
            for index, atom1 in enumerate(atom_list):
                for atom2 in atom_list[:index]:
                    if (atom1, atom2) not in existing and (atom2, atom1) not in existing:
                        force.addExclusion(atom1, atom2)
    return system


def remove_bond(system, ligand_indices):
    """Zero ligand-only bonded terms in the original force objects."""
    for force in system.getForces():
        _remove_bonded_interactions(force, ligand_indices, True)
    return system
