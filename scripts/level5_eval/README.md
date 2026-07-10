# Level-5 mobile-manipulation VLA eval harness

Full-body (move+pick) remote VLA evaluation for the Level-5 tasks. The policy
drives all 12 Ridgebase DOFs: base action dims 0:3 are BODY-frame deltas
integrated closed-loop onto the measured base pose (`_apply_action11`), arm
dims are absolute joints. See `docs/level5_inference_report.md` for the
close_pick results and pitfalls.

## Flow

```bash
# 1. serve on the cluster (1 GPU per model, ma4)
bash scripts/level5_eval/serve.sh submit lingbot
bash scripts/level5_eval/serve.sh status

# 2. tunnel (discovers POD_IP from the serve log)
bash scripts/level5_eval/tunnel.sh start lingbot

# 3. eval — main.py takes NO hydra overrides; episode count lives in the config
python main.py --config-name=level5_close_pick_lingbot --headless --no-video

# 4. metrics (grasp success + nav progress)
bash scripts/level5_eval/metrics.sh <run_log>
```

## Port map (pod / local tunnel)

| model   | pod port | local port | action chunk |
|---------|----------|------------|--------------|
| openpi  | 9081     | 18081      | 10           |
| lingbot | 9082     | 18082      | 50           |
| gr00t   | 9084     | 18084      | 40           |
| smolvla | 9085     | 18085      | 50           |

## Configs

`config/level5_close_pick_<model>.yaml` — `infer.type: remote_mobile`,
`base_delta_actions: true`, `image_color: rgb`, `n_obs_steps: 1`.

obs_names (camera_data key = camera NAME for plain-rgb cameras):

- openpi / gr00t / lingbot: `front->observation/image, top->observation/wrist_image, wrist->observation/wrist_image_2`
- smolvla: `front->observation/image, top->observation/image_2, wrist->observation/wrist_image`

## Pitfalls (hard-won)

- **State gripper ×2**: the mobile collector records `state[10] = finger1 × 2`;
  `MobileRemoteInferenceEngine` mirrors this. Sending raw finger1 halves the
  gripper state and breaks grasping (smolvla 0% → 10% after the fix).
- **Tunnel host**: forward through the workspace host
  (`lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn`), NOT the `ma4science`
  scheduler alias (RemoteForward bad-signature kills the session).
- **One Isaac at a time**: never run two `main.py` in parallel.
- **Serve entries** live in each model repo on the cluster:
  `code/<repo>/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh`
  (override `CKPT`/`PORT` via env for other L5 tasks).
