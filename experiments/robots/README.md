# Experimental robots

This directory contains the multi-arm compatibility sweep. These robot variants,
their generated USD closures, and the matching pick configurations are not part
of the default LabUtopia benchmark runtime.

- `definitions/`: experimental robot classes, URDFs, and RMPFlow descriptions.
- `assets/robots/`: generated USD layers used by those definitions.
- `config/`: one-episode pick smoke configurations.
- `registry.py`: opt-in factory registration used by `experiments/main.py`.
- `render_all_arm_picks.sh`: compatibility sweep runner.

Run a single variant from the repository root:

```bash
python experiments/main.py \
  --config-dir experiments/robots/config \
  --config-name level1_pick_xarm7_robotiq
```
