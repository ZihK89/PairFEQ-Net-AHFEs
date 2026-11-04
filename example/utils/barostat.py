"""Isotropic molecular Monte Carlo volume moves using the physical ML/MM energy."""

import numpy as np
from openmm import unit

KB = 0.00831446261815324  # kJ/mol/K
BAR_NM3 = 0.0602214076  # bar * nm^3 -> kJ/mol


class MolecularBarostat:
    def __init__(self, context, energy, temperature, pressure, cutoff, seed):
        self.context, self.energy = context, energy
        self.temperature, self.pressure, self.cutoff = temperature, pressure, cutoff
        self.molecules = [np.asarray(atoms) for atoms in context.getMolecules()]
        self.rng = np.random.default_rng(seed)
        box = context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        # A fixed, symmetric proposal in volume; no production adaptation.
        self.volume_step = 0.005 * np.linalg.det(box)
        self.attempts = self.accepted = 0

    def attempt(self):
        """Keep intramolecular coordinates and velocities unchanged by the move."""
        self.attempts += 1
        state = self.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        volume = np.linalg.det(box)
        new_volume = volume + self.rng.uniform(-self.volume_step, self.volume_step)
        if new_volume <= 0:
            return False
        scale = (new_volume / volume)**(1/3)
        if np.any(np.diag(box * scale) <= 2 * self.cutoff):
            return False
        old_energy = self.energy()
        trial = positions.copy()
        for atoms in self.molecules:
            trial[atoms] += (scale - 1) * positions[atoms].mean(axis=0)
        self.context.setPeriodicBoxVectors(*(box * scale))
        self.context.setPositions(trial)
        accepted = False
        try:
            work = self.energy() - old_energy + BAR_NM3 * self.pressure * (new_volume - volume)
            log_accept = -work / (KB * self.temperature) + len(self.molecules) * np.log(new_volume / volume)
            if not np.isfinite(log_accept):
                raise FloatingPointError("Nonfinite volume-move acceptance probability")
            accepted = np.log(self.rng.random()) < min(0.0, log_accept)
        finally:
            if not accepted:
                self.context.setPeriodicBoxVectors(*box)
                self.context.setPositions(positions)
                self.energy()  # Restore the charges at the original coordinates.
        self.accepted += int(accepted)
        return bool(accepted)
