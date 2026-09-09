# Experiments

This directory contains one-off diagnostics, asset-building scripts, publication
rendering tools, and their focused checks. None of these modules belongs to the
normal LabUtopia runtime or the default test suite.

- `main.py`: extended experimental runner used by rendering and pipette probes.
- `config/`: rendering presets and temporary physics candidates.
- `paper/`: publication rendering and frame-selection tools.
- `pipette/`: pipette investigation and paired-evaluation utilities.
- `assets/`: one-off asset generation, calibration, and validation scripts.
- `robots/`: opt-in multi-arm compatibility sweep, configs, and generated assets.
- `sim2real/`: one-off dataset/camera alignment utilities.
- `urdf_to_usd/`: one-off repair passes for generated robot assets.
- `tests/`: checks for the experimental tools and generated assets.
- `support/`: helpers used only by experiments.

Run Python entry points from the repository root, preferably as modules, for
example `python -m experiments.pipette.evaluate_pipette_pairs --help`.
