# Caffeine ML/MM NPT sampling

A simulation-only example of caffeine (24 atoms) in 1,000 TIP3P waters.
PairFEQ-Net predicts the solute energy, forces and neutralized CM5 charges at
every step. Charges are scaled by 1.2, and the force includes the full charge
response `-sum_i (dU/dq_i)(dq_i/dR)` with PME electrostatics.

## Run

Use Python 3.9–3.11 and the packages in `requirements.txt` (TensorFlow 2 / Keras 2).
From the repository root in a suitable Python environment:

```bash
python -m pip install -r example/requirements.txt
python example/PairFEQNet_solv.py
```

The defaults follow the original sampling example:

- NPT at 300 K and 1 bar; 1 fs timestep and Langevin friction of 5 ps^-1.
- Lambda windows: 0, 0.2, 0.4, 0.6, 0.8, 1.
- Each window starts from the supplied equilibrated configuration, initializes
  velocities at 300 K, and runs 50 ps equilibration plus 3 ns production.
- Save production data every 1 ps; attempt a volume move every 50 steps.

Paths and settings are grouped at the top of `PairFEQNet_solv.py`. Use
`--platform CUDA` for NVIDIA GPUs. ML inference and the auxiliary PME solve
use the CPU on both platforms. For a short, single-window CPU check:

```bash
python example/PairFEQNet_solv.py --lambda-values 1 --equilibration-ps 0.2 \
    --duration-ps 1 --save-interval-ps 0.1 --output example/md_output_check
```

Choose a new output directory for each run; existing results are not overwritten.

## Pressure control and energy

`utils/barostat.py` implements isotropic molecular Monte Carlo volume moves
with a fixed symmetric proposal width of 0.5% of the initial volume. Molecular
centers move with the box; internal coordinates and velocities stay unchanged.
The Metropolis criterion uses the physical ML/MM energy difference, pressure
work and the molecular volume Jacobian. This follows the molecular-volume
measure described in the [OpenMM barostat implementation](https://github.com/openmm/openmm/blob/8.2.0/openmmapi/src/MonteCarloBarostatImpl.cpp).

Trial-state charges and ML energy are recomputed explicitly. The linear
force-injection surrogate is excluded from the acceptance energy. Every volume
move is followed by a full charge/response-force refresh before integration;
there is no additional OpenMM `MonteCarloBarostat` attached to the system.

The physical potential is the OpenMM potential without the injection surrogate,
plus lambda times the actual ML energy. Kinetic energy is reported by OpenMM's
Langevin integrator. NPT exchanges heat and pressure work, so total energy is
not expected to remain constant. The conservative dynamic-charge force
formulation is retained; this example runs NPT sampling, not an NVE test.

## Files and outputs

- `PairFEQNet_solv.py`: NPT sampling and output.
- `utils/network.py`: inference-only model loader.
- `utils/charge_response.py`, `utils/pme.py`: charge-response forces and their
  startup check against OpenMM charge derivatives.
- `utils/barostat.py`: physical-energy molecular volume moves.
- `trained_model/`: the existing caffeine model.
- `inputs/system.xml`: prepared OpenFF 2.1.0 / TIP3P system with the ML/MM
  interpolation terms; caffeine occupies particles 0–23.
- `inputs/initial_state.npz`: initial positions and box vectors in nm.
- `input.sdf`: caffeine identity and atom order; rebuilding is outside this example.

Each `md_output/window_XX_lambda_Y/` contains `coordinates.npy` (nm),
`box_vectors.npy` (nm) and `charges.npy` (elementary charge) at the production
save interval, including production time zero. `energy.csv` records production
time, physical potential/kinetic/total energies (kcal/mol), temperature (K),
volume (nm^3) and cumulative accepted/attempted volume moves (including
equilibration). `final_state.xml` holds the final OpenMM state. Only complete
save intervals are written; the final state is always saved.

No cross-state energy evaluation, `u_kln`/`u_kn` construction, MBAR, training or
cluster job management is included.
