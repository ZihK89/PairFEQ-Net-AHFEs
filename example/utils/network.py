from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


tf.get_logger().setLevel("ERROR")


class Molecule:
    """Minimal model metadata container.

    PairFEQNet_solv.py only sets ``n_atom`` before loading the network, so
    the full dataset-oriented Molecule implementation is not required.
    """

    n_atom: int


def _read_model_parameters(path: str | Path) -> dict:
    """Read only the ANN parameters required to reconstruct the model."""
    raw: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            raw[key.strip()] = value.strip()

    required = {
        "activations",
        "n_layers",
        "n_nodes",
        "charge_scheme",
        "norm_scheme",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing ANN parameters: {', '.join(missing)}")

    activations = raw["activations"]
    if activations not in {"silu", "linear"}:
        raise ValueError(f"Unsupported activation function: {activations}")

    n_layers = int(raw["n_layers"])
    if raw["n_nodes"] == "auto":
        n_nodes: str | list[int] = "auto"
    else:
        n_nodes = [int(value.strip()) for value in raw["n_nodes"].split(",")]
        if len(n_nodes) != n_layers:
            raise ValueError("n_nodes must contain one value per hidden layer")

    charge_scheme = int(raw["charge_scheme"])
    if charge_scheme not in {1, 2}:
        raise ValueError(f"Unsupported charge scheme: {charge_scheme}")

    norm_scheme = raw["norm_scheme"]
    if norm_scheme not in {"z-score", "force", "none"}:
        raise ValueError(f"Unsupported normalisation scheme: {norm_scheme}")

    return {
        "activations": activations,
        "n_layers": n_layers,
        "n_nodes": n_nodes,
        "charge_scheme": charge_scheme,
        "norm_scheme": norm_scheme,
    }


class NuclearChargePairs(Layer):
    def __init__(self, n_pairs, n_atoms, **kwargs):
        super().__init__()
        self.n_pairs = n_pairs
        self.n_atoms = n_atoms

    def call(self, atom_nc):
        a = tf.expand_dims(atom_nc, 2)
        b = tf.expand_dims(atom_nc, 1)
        products = a * b
        lower_triangle = tf.linalg.band_part(products, -1, 0)
        diagonal = tf.linalg.band_part(lower_triangle, 0, 0)
        strict_lower_triangle = lower_triangle - diagonal
        indices = tf.where(
            tf.not_equal(strict_lower_triangle, tf.zeros_like(strict_lower_triangle))
        )
        values = tf.gather_nd(strict_lower_triangle, indices)
        return tf.reshape(values, shape=(tf.shape(atom_nc)[0], self.n_pairs))


class CoordsToNRF(Layer):
    def __init__(self, max_nrf, n_pairs, n_atoms, **kwargs):
        super().__init__()
        self.max_nrf = max_nrf
        self.n_pairs = n_pairs
        self.n_atoms = n_atoms
        self.au2kcalmola = 627.5095 * 0.529177

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pairs)

    def call(self, coords_nc):
        coords, atom_nc = coords_nc
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        squared_distances = tf.reduce_sum((a - b) ** 2, axis=-1)
        lower_triangle = tf.linalg.band_part(squared_distances, -1, 0)
        indices = tf.where(tf.not_equal(lower_triangle, tf.zeros_like(lower_triangle)))
        values = tf.gather_nd(lower_triangle, indices)
        distances = tf.reshape(values, shape=(tf.shape(lower_triangle)[0], -1)) ** 0.5
        nrf = (atom_nc * self.au2kcalmola) * (1 / distances**2) / self.max_nrf
        return tf.reshape(nrf, shape=(tf.shape(coords)[0], self.n_pairs))


class E(Layer):
    def __init__(self, prescale, norm_scheme, **kwargs):
        super().__init__()
        self.prescale = prescale
        self.norm_scheme = norm_scheme

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def call(self, energy_scaled):
        if self.norm_scheme == "z-score":
            return energy_scaled * self.prescale[1] + self.prescale[0]
        if self.norm_scheme == "force":
            return (
                (energy_scaled - self.prescale[2])
                / (self.prescale[3] - self.prescale[2])
                * (self.prescale[1] - self.prescale[0])
                + self.prescale[0]
            )
        return energy_scaled


class Eij(Layer):
    def __init__(self, n_pairs, max_eij, **kwargs):
        super().__init__()
        self.n_pairs = n_pairs
        self.max_eij = max_eij

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pairs)

    def call(self, decomposition_scaled):
        decomposition_scaled = tf.reshape(
            decomposition_scaled, shape=(tf.shape(decomposition_scaled)[0], -1)
        )
        decomposition = (decomposition_scaled - 0.5) * (2 * self.max_eij)
        return tf.reshape(
            decomposition,
            shape=(tf.shape(decomposition_scaled)[0], self.n_pairs),
        )


class ERecomposition(Layer):
    def __init__(self, n_atoms, n_pairs, **kwargs):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def call(self, coords_decomposition):
        coords, decomposition = coords_decomposition
        decomposition = tf.reshape(decomposition, shape=(tf.shape(decomposition)[0], -1))
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        squared_distances = tf.reduce_sum((a - b) ** 2, axis=-1)
        lower_triangle = tf.linalg.band_part(squared_distances, -1, 0)
        indices = tf.where(tf.not_equal(lower_triangle, tf.zeros_like(lower_triangle)))
        values = tf.gather_nd(lower_triangle, indices)
        inverse_distances = 1 / (
            tf.reshape(values, shape=(tf.shape(lower_triangle)[0], -1)) ** 0.5
        )
        norm = tf.reduce_sum(inverse_distances**2, axis=1, keepdims=True) ** 0.5
        pair_weights = inverse_distances / norm
        energy = tf.einsum("bi,bi->b", pair_weights, decomposition)
        return tf.reshape(energy, shape=(tf.shape(coords)[0], 1))


class F(Layer):
    def __init__(self, n_atoms, n_pairs, **kwargs):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.n_atoms, 3)

    def call(self, energy_coords):
        energy, coords = energy_coords
        gradients = tf.gradients(energy, coords, unconnected_gradients="zero")
        return -gradients[0]


class Q(Layer):
    def __init__(self, n_atoms, n_pairs, charge_scheme, **kwargs):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs
        self.charge_scheme = charge_scheme

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_atoms)

    def call(self, charges):
        if self.charge_scheme == 1:
            return charges
        net_charge_per_atom = tf.reduce_sum(charges) / self.n_atoms
        return charges - net_charge_per_atom


class Network:
    """Build and load the inference-only PairFEQ-Net model."""

    def __init__(self, molecule):
        self.model = None

    def load(self, mol, input_dir):
        input_path = Path(input_dir)
        ann_params = _read_model_parameters(input_path / "ann_params.txt")
        prescale = np.loadtxt(input_path / "prescale.txt", dtype=np.float64).reshape(-1)
        model = self.build(mol, ann_params, prescale)
        model.load_weights(str(input_path / "best_model")).expect_partial()
        self.model = model
        return model

    def build(self, mol, ann_params, prescale):
        n_atoms = mol.n_atom
        n_pairs = n_atoms * (n_atoms - 1) // 2
        n_layers = ann_params["n_layers"]
        n_nodes = ann_params["n_nodes"]
        if n_nodes == "auto":
            n_nodes = [n_atoms * 30] * n_layers

        max_nrf = tf.constant(prescale[4], dtype=tf.float32)
        max_eij = tf.constant(prescale[5], dtype=tf.float32)
        prescale = tf.constant(prescale, dtype=tf.float32)

        coords = Input(shape=(n_atoms, 3), name="coords_layer")
        nuclear_charges = Input(shape=(n_atoms,), name="nuclear_charge_layer")
        charge_pairs = NuclearChargePairs(n_pairs, n_atoms)(nuclear_charges)
        connected = CoordsToNRF(max_nrf, n_pairs, n_atoms, name="NRF_layer")(
            [coords, charge_pairs]
        )

        for layer_index in range(n_layers):
            connected = Dense(
                units=n_nodes[layer_index],
                activation=ann_params["activations"],
                name=f"net_layerA{layer_index}",
            )(connected)

        pair_energies_scaled = Dense(
            units=n_pairs, activation="linear", name="net_layer_n_pair"
        )(connected)
        charges_uncorrected = Dense(
            units=n_atoms, activation="linear", name="net_layer_n_atm"
        )(connected)

        pair_energies = Eij(n_pairs, max_eij, name="unscale_E_layer")(
            pair_energies_scaled
        )
        energy_scaled = ERecomposition(n_atoms, n_pairs, name="E_layer")(
            [coords, pair_energies]
        )
        energy = E(prescale, ann_params["norm_scheme"], name="energy")(
            energy_scaled
        )
        forces = F(n_atoms, n_pairs, name="force")([energy, coords])
        charges = Q(
            n_atoms,
            n_pairs,
            ann_params["charge_scheme"],
            name="charge",
        )(charges_uncorrected)

        return Model(
            inputs=[coords, nuclear_charges],
            outputs=[forces, energy, charges],
        )
