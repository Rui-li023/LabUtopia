# Rendering Configurations

Publication-only and fixed visual inspection configurations live here. Run from
the repository root using Hydra's config path, for example:

```bash
python experiments/main.py --config-dir experiments/config --config-name render/level1_pick_franka_beaker --headless
python experiments/main.py --config-dir experiments/config --config-name render/level3_pick_visual_gallery --headless
python experiments/main.py --config-dir experiments/config --config-name render/level3_pick_visual_mixed_lights --headless
```

Use `--config-dir experiments/config`; the experimental runner also adds the
production `config/` directory to Hydra's search path so visual presets can
inherit their source tasks. The global package directive keeps task, robot and
camera settings at the top level.

The multi-arm pick configurations live in `experiments/robots/config/` and are
loaded only by the experimental runner.
`level3_pick_visual.yaml` remains a training/evaluation randomization profile.

## Reference-guided pipette replay

This preset replays recorded joint commands and evaluates the physical placement
again. It does not generate motion from a trained model. Point it at a successful
collection's `dataset` directory:

```bash
LABUTOPIA_PIPETTE_DATASET=/absolute/path/to/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level2_pipette_to_rack --headless \
  --single-camera 0 --render-width 1280 --render-height 960
```

The preset inherits the real-geometry task without changing its success
thresholds. A trained policy requires the task's `infer` checkpoint or server
configuration, which is not supplied by this render preset.

The collection config now uses the enlarged **260 mm** pipette. Historical
220 mm recordings must also set the matching scene; their success results do
not validate the larger object:

```bash
LABUTOPIA_PIPETTE_ASSET=assets/chemistry_lab/lab_002/realistic_pipette/lab_002_pipette.usda \
LABUTOPIA_PIPETTE_DATASET=/absolute/path/to/220mm/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level2_pipette_to_rack --headless
```

The asset override alone is not sufficient for historical trajectories: restore
the recording's robot, gripper and physics settings too. Prefer its saved
resolved config when those settings differ from the current task.

## Three-pipette Level 4 replay (current requested variant)

Use `render/level4_three_pipettes_to_rack` with a dataset recorded by
`level4_three_pipettes_to_rack` and the same physics settings. All three tools
must pass the unchanged placement and 60-step stability gates. The existing
four-slot rack remains unchanged, with one empty slot. Four-tool datasets are
not compatible and are rejected before replay state restoration.

```bash
LABUTOPIA_PIPETTE_DATASET=/absolute/path/to/three-pipette/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level4_three_pipettes_to_rack --headless \
  --single-camera 0 --render-width 960 --render-height 720 \
  --camera-look-at -0.75 -0.6 1.8 0.18 -0.05 0.87 --camera-focal-length 2.0
```

## Four-pipette whole-episode replay (retained historical variant)

Use a dataset collected with the current four-slot stand and matching physics
configuration. This front-wide view keeps all four tools and the stand visible:

```bash
LABUTOPIA_PIPETTE_DATASET=/absolute/path/to/four-pipette/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level4_four_pipettes_to_rack --headless \
  --single-camera 0 --render-width 960 --render-height 720 \
  --camera-look-at -0.75 -0.6 1.8 0.18 -0.05 0.87 --camera-focal-length 2.0
```

The success test still requires all four released tools to remain stably seated
for 60 consecutive steps. Rendering does not bypass contact physics or use a
trained VLA model. Keep visual presets here and collection parameters in the
top-level task configs; every nested preset must use `# @package _global_`.

The acrylic-only contact demonstration is saved at
`outputs/pipette_reliability/level4_acrylic_contact_wide_replay/video/episode_0_success.mp4`
(960×720, 179.267 seconds). It is one successful diagnostic replay, not a new
independent evaluation sample or evidence of zero grip rotation. Current task
defaults now include the scoped acrylic contact material; old recordings must
continue to use their own saved physical settings.

## Original round-bottom flask / metal tripod replay

This preset keeps the original lab USD appearance and uses only the corrected
static support colliders. Use a dataset from the matching original-flask task,
not the previously rebuilt `level2_glassware` asset:

```bash
LABUTOPIA_FLASK_DATASET=/absolute/path/to/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level2_flask_to_triangle --headless
```

It renders the first recorded episode from a view that shows the ring beneath
the flask after release. This is action replay, not a trained VLA policy.
The original stopper and starting support now use the requested cork PBR
material; the glass, metal tripod, geometry and physics are unchanged by that
visual layer.

## Round-bottom flask: actual cork insertion

Use newly collected episodes from the corrected `level2_stopper_to_flask` task:

```bash
LABUTOPIA_STOPPER_DATASET=/absolute/path/to/dataset \
python experiments/main.py --config-dir experiments/config --config-name render/level2_stopper_to_flask --headless
```

This close view replays recorded actions, not a trained policy. The preset keeps
the task's released-insertion depth, alignment and 60-step stability checks.
Old recordings using the smaller stopper or different colliders are not suitable
for validating the repaired task. Keep their saved config if inspecting them.
