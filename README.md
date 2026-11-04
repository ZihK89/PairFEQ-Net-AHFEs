Supporting information for the PairFEQ-Net work on **absolute hydration free energy calculations**.

`./data` contains the molecular input files and the **QM reference datasets** calculated at the **B3LYP-D3/6-31+G\*** level for model training.
`./figs` contains the scripts used to reproduce the figures.

`./example` contains a compact [caffeine NPT example](example/PairFEQNet_solv.py) with the full dynamic-charge response force. It runs NPT ML/MM sampling and records physical energies and trajectories; it does not calculate free-energy matrices.
