"""PairFEQ-Net inference and the chain-rule force -sum_i (dU/dq_i)(dq_i/dR)."""

import numpy as np
import tensorflow as tf
from openmm import unit

from .pme import PMEPotential


class ChargeResponse:
    def __init__(self, model, atomic_numbers, nonbonded, context, charge_scaling):
        self.model = model
        self.charge_model = tf.keras.Model(model.inputs, model.outputs[2])
        self.z = tf.constant(atomic_numbers[None], dtype=tf.float32)
        self.n = len(atomic_numbers)
        self.nb, self.context = nonbonded, context
        self.scale = charge_scaling
        self.parameters = [nonbonded.getParticleParameters(i) for i in range(self.n)]
        self.charges = np.array([nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                                 for i in range(nonbonded.getNumParticles())])
        self.q0 = self.charges[:self.n].copy()
        self.solver = None
        self.predict = tf.function(self._predict, input_signature=[
            tf.TensorSpec([1, self.n, 3], tf.float32)], autograph=False)
        self.response = tf.function(self._response, input_signature=[
            tf.TensorSpec([1, self.n, 3], tf.float32),
            tf.TensorSpec([1, self.n], tf.float32)], autograph=False)

    def _predict(self, coordinates):
        return self.model([coordinates, self.z], training=False)

    def _response(self, coordinates, potentials):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(coordinates)
            raw = self.charge_model([coordinates, self.z], training=False)
            charges = self.scale * (raw - tf.reduce_mean(raw, axis=1, keepdims=True))
        # Network coordinates are angstrom; OpenMM forces are kJ/mol/nm.
        return -10.0 * tape.gradient(charges, coordinates, output_gradients=potentials)

    def response_force(self, coordinates, phi):
        return self.response(coordinates, np.asarray(phi[None], dtype=np.float32)).numpy()[0].astype(float)

    def set_charges(self, charges):
        for i, charge in enumerate(charges):
            _, sigma, epsilon = self.parameters[i]
            self.nb.setParticleParameters(i, float(charge), sigma, epsilon)
        self.nb.updateParametersInContext(self.context)
        self.charges[:self.n] = charges

    def validate(self, coordinates, phi):
        """Compare the PME derivative with neutral charge probes before dynamics."""
        saved = [self.nb.getParticleParameters(i) for i in range(self.n)]
        q = self.charges[:self.n].copy()
        groups = 1 << self.nb.getForceGroup()
        reciprocal = self.nb.getReciprocalSpaceForceGroup()
        if reciprocal >= 0:
            groups |= 1 << reciprocal
        try:
            for h in (0.5, 1.0):
                reference = np.zeros(self.n)
                for i in range(self.n - 1):
                    energies = []
                    for sign in (1, -1):
                        self.nb.setParticleParameters(i, q[i] + sign*h, *saved[i][1:])
                        self.nb.setParticleParameters(self.n - 1, q[-1] - sign*h, *saved[-1][1:])
                        self.nb.updateParametersInContext(self.context)
                        state = self.context.getState(getEnergy=True, groups=groups)
                        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    reference[i] = (energies[0] - energies[1]) / (2*h)
                    self.nb.setParticleParameters(i, *saved[i])
                reference -= reference.mean()
                phi_error = np.max(np.abs(phi - reference))
                force_error = np.max(np.abs(self.response_force(coordinates, phi)
                                            - self.response_force(coordinates, reference))) / 41.84
                if not (phi_error <= 0.02 and force_error <= 0.005):
                    raise RuntimeError(f"PME validation failed: dU/dq error={phi_error:g} kJ/mol/e; "
                                       f"response-force error={force_error:g} kcal/mol/angstrom")
        finally:
            for i, parameters in enumerate(saved):
                self.nb.setParticleParameters(i, *parameters)
            self.nb.updateParametersInContext(self.context)
        print("PME charge-response check passed.", flush=True)

    def prediction(self, positions, lambda_value):
        """Update charges; also used for physical energies during volume moves."""
        coordinates = np.asarray(positions[None, :self.n] * 10, dtype=np.float32)
        forces, energy, raw = [v.numpy() for v in self.predict(coordinates)]
        if round(float(raw.sum())) != 0:
            raise ValueError("This example requires neutral caffeine charges")
        predicted = self.scale * (raw[0] - raw[0].mean())
        charges = (1 - lambda_value)*self.q0 + lambda_value*predicted
        self.set_charges(charges)
        ml_energy = float(energy[0, 0])*4.184
        if not np.isfinite(ml_energy) or not np.isfinite(charges).all():
            raise FloatingPointError("Invalid model energy or charge")
        return coordinates, ml_energy, forces[0].astype(float)*41.84, charges

    def evaluate(self, positions, box, lambda_value):
        coordinates, ml_energy, forces, charges = self.prediction(positions, lambda_value)
        if self.solver is None:
            self.solver = PMEPotential(self.nb, self.context, self.n)
            phi = self.solver.potentials(positions, self.charges, box)
            self.validate(coordinates, phi)
        else:
            phi = self.solver.potentials(positions, self.charges, box)
        injected = lambda_value * (forces + self.response_force(coordinates, phi))
        if not np.isfinite(injected).all() or np.max(np.abs(injected)) > 41840:
            raise FloatingPointError("Invalid response-corrected force")
        return ml_energy, injected, charges
