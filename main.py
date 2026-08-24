import argparse
import os
import random
import sys

from isaacsim import SimulationApp
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--config-name', type=str, default='level3_heat_liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    parser.add_argument('--max-episodes', type=int,
                       help='Override the configured episode count for this run')
    parser.add_argument('--max-attempts', type=int,
                       help='Stop after this many attempts, including failed episodes')
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up simulation app based on arguments
simulation_config = {
    # Respect the CLI switch and allow remote/headless runners to opt in via
    # an environment variable without needing a virtual X server.
    "headless": args.headless or os.environ.get("LABUTOPIA_HEADLESS", "").lower()
    in {"1", "true", "yes"},
    "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
}

simulation_app = SimulationApp(simulation_config)

import hydra
import subprocess
import threading
from omegaconf import OmegaConf
import cv2
import numpy as np

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from isaacsim.core.utils import extensions

extensions.enable_extension("omni.physx.bundle")
extensions.enable_extension("omni.usdphysics.ui")

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller
from robots.franka.rmpflow_controller import RMPFlowController
from controllers.atomic_actions.atomic_base_controller import AtomicBaseController



def _convert_to_h264(src_path: str, is_success: bool):
    """Re-encode a video to H264 in-place using ffmpeg (runs in background thread)."""
    if is_success:
        tmp_path = src_path + ".success.h264.tmp.mp4"
        dest_path = src_path.replace(".mp4", "_success.mp4")
    else:
        tmp_path = src_path + ".failure.h264.tmp.mp4"
        dest_path = src_path.replace(".mp4", "_failure.mp4")
    logger.info(f"Converting video to H264: {src_path}")
    ret = subprocess.run(
        [
            "ffmpeg", "-y", "-i", src_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            tmp_path,
        ],
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if ret.returncode == 0:
        os.replace(tmp_path, dest_path)
        os.remove(src_path)
        logger.success(f"Video saved: {dest_path}")
    else:
        stderr_msg = ret.stderr.decode(errors="replace") if ret.stderr else ""
        logger.error(f"H264 conversion failed for: {src_path}\n{stderr_msg}")
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


_convert_threads: list[threading.Thread] = []


def release_and_convert(writer: cv2.VideoWriter, output_path: str, is_success: bool):
    """Release cv2 writer and kick off H264 conversion in a background thread."""
    writer.release()
    t = threading.Thread(target=_convert_to_h264, args=(output_path, is_success), daemon=False)
    t.start()
    _convert_threads.append(t)


def _round_values(values):
    try:
        return [round(float(value), 4) for value in values]
    except (TypeError, ValueError):
        return values


def _object_quaternion(state, task):
    object_path = state.get("object_path")
    if not object_path:
        return None
    pose = task.object_utils.get_world_pose(object_path)
    if not pose:
        return None
    return [round(float(value), 3) for value in pose["orientation"]]

def main():
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name)
    cfg = hydra.compose(config_name=args.config_name)
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be at least 1")
        cfg.max_episodes = args.max_episodes
    if args.max_attempts is not None and args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")
    logger.info(f"Config loaded: {args.config_name}, run dir: {cfg.multi_run.run_dir}")

    # Optional global RNG seed. Off unless the config sets `seed`, so existing
    # runs are unchanged. With it, two runs of the same config draw the same
    # object placements / material cycle / lighting samples — which is what makes
    # an A/B (e.g. baseline vs unseen-lighting) a controlled comparison instead
    # of a comparison of two different random scene sequences.
    _seed = getattr(cfg, "seed", None)
    if _seed is not None:
        random.seed(int(_seed))
        np.random.seed(int(_seed))
        logger.info(f"[seed] global RNG seeded with {int(_seed)}")

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    # Override configuration based on command line arguments
    if args.no_video:
        save_video = False
        show_video = False
    else:
        save_video = True
        # A headless run can still render and save camera frames, but OpenCV has no
        # display server for imshow(). Keep remote diagnostics recording enabled
        # without trying to create a GUI window.
        show_video = not simulation_config["headless"]

    # Scene BEFORE robot. A scene USD may already contain the arm we want to
    # drive (the sim2real bench has both of its Frankas baked in); the robot
    # constructor only binds to an existing prim if that prim is already on the
    # stage, otherwise it references a fresh one and clobbers the authored base
    # transform. No shipped scene contains a robot, so this reorder is a no-op
    # for every pre-existing config.
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")

    robot_kwargs = {"position": np.array(cfg.robot.position)}
    if hasattr(cfg.robot, "default_joint_positions"):
        robot_kwargs["default_joint_positions"] = np.array(cfg.robot.default_joint_positions)
    if hasattr(cfg.robot, "usd_path"):
        robot_kwargs["usd_path"] = str(cfg.robot.usd_path)
    if hasattr(cfg.robot, "prim_path"):
        # Bind to an arm that already exists in the scene instead of spawning one.
        # Orientation is deliberately NOT passed: leaving it unset preserves the
        # base yaw authored in the scene, which RMPFlowController then reads via
        # get_world_pose() and feeds to set_robot_base_pose().
        robot_kwargs["prim_path"] = str(cfg.robot.prim_path)
    robot = create_robot(cfg.robot.type, **robot_kwargs)
    logger.info(f"Robot created: {cfg.robot.type} at {getattr(robot, 'prim_path', cfg.robot.get('prim_path', '/World/Franka'))}")

    # Configure gripper control mode if specified
    gripper_cfg = getattr(cfg.robot, "gripper", None)
    if gripper_cfg:
        mode = str(getattr(gripper_cfg, "control_mode", "position"))
        # backward compat: force_mode: true → control_mode: "force"
        if getattr(gripper_cfg, "force_mode", False) and mode == "position":
            mode = "force"
        if mode != "position":
            robot.set_gripper_control_mode(
                mode=mode,
                closing_force=float(getattr(gripper_cfg, "closing_force", 20.0)),
                closing_speed=float(getattr(gripper_cfg, "closing_speed", 0.2)),
            )
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    # Position-only collection: make RMPFlow roll out an internal virtual robot
    # (ignore measured joint state) so its position targets advance at planned
    # speed while the real arm tracks them with pure position PD — the identical
    # control law replay and inference use. Must be set BEFORE the controller is
    # constructed (atomic controllers build their RMPFlowController in __init__).
    # Applies to collect AND to replay of a position-only dataset (the replay
    # config inherits the flag): controllers with scripted phases in replay
    # (e.g. pour's scripted pick) must drive the arm with the exact same
    # control law the data was collected with, or the hand-off joint state
    # diverges from what the recorded actions assume.
    if bool(getattr(cfg, "collect_position_only", False)):
        RMPFlowController.ignore_robot_state_updates = True
        AtomicBaseController.record_commanded_gripper = True
        logger.info("[collect_position_only] RMPFlow virtual-robot rollout ON, "
                    "velocity feed-forward stripped, gripper channel = commanded")

    # Long-horizon tasks (level4) chain many atomic actions, so per-episode
    # action noise compounds across the sequence; disable it via config.
    if not bool(getattr(cfg, "action_randomization", True)):
        AtomicBaseController.randomization_enabled = False
        logger.info("[action_randomization=false] per-episode action noise "
                    "neutralized (deterministic mid-range parameters)")

    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    
    video_writer = None
    video_output_path = None
    _replay_count_logged = False
    # An episode can end two ways: the controller returns done (handled in the
    # `if done:` branch below) or the task hits max_steps and only flips
    # reset_needed. The latter used to skip the whole end-of-episode block, so
    # timed-out episodes printed no failure reason and never cleared the
    # per-episode robot diagnostics (they leaked into every later episode).
    _episode_finalized = False
    _steps_since_reset = 0
    arm_diagnostics_enabled = bool(os.environ.get("LABUTOPIA_ARM_DIAGNOSTICS"))

    def _reset_episode_diagnostics(rb) -> None:
        if not arm_diagnostics_enabled:
            return
        _am = getattr(rb, "_argmax_diag", (None, None, None))
        logger.info(f"[EP-ARM-ERR] mode={cfg.mode} ep={task_controller.episode_num} "
                    f"max_arm_track_err={round(float(getattr(rb, '_ep_max_arm_err', 0.0)), 4)} "
                    f"at_frame={_am[0]}/{getattr(rb, '_frame_diag', 0)} joint={_am[1]} per_joint_err={_am[2]}")
        rb._ep_max_arm_err = 0.0
        rb._frame_diag = 0
        rb._argmax_diag = (None, None, None)
        rb._closed_once = False

    _arm_stiffness_scale = float(getattr(cfg, "arm_stiffness_scale", 1.0))

    def _apply_arm_gains():
        # Scale the arm position-drive gains so pure joint-position control (no
        # velocity feed-forward) tracks briskly instead of crawling. Re-applied
        # after each reset because re-init restores USD defaults.
        if _arm_stiffness_scale == 1.0:
            return
        try:
            ac = robot.get_articulation_controller()
            kps, kds = ac.get_gains()
            if kps is None:
                return
            kps = list(kps); kds = list(kds)
            for j in range(min(7, len(kps))):
                kps[j] = kps[j] * _arm_stiffness_scale
                kds[j] = kds[j] * (_arm_stiffness_scale ** 0.5)
            ac.set_gains(kps=kps, kds=kds)
            logger.info(f"[gains] arm kp x{_arm_stiffness_scale} -> kp[:7]={[round(float(k),1) for k in kps[:7]]}")
        except Exception as e:
            logger.warning(f"[gains] failed to scale arm gains: {e}")

    def _log_reset_object_poses(label: str) -> None:
        if not os.environ.get("LABUTOPIA_RESET_DEBUG"):
            return
        root_path = getattr(task, "current_obj_path", None)
        if not root_path:
            return
        poses = {}
        for path in (root_path, f"{root_path}/mesh"):
            pose = task.object_utils.get_world_pose(path)
            if pose is not None:
                poses[path] = np.round(pose["position"], 4).tolist()
        logger.info(f"[reset-debug] {label} poses={poses}")

    task.reset()
    _apply_arm_gains()
    if os.environ.get("LABUTOPIA_PICK_DEBUG"):
        _debug_kp, _debug_kd = robot.get_articulation_controller().get_gains()
        logger.info(
            f"[ROBOT-DOF] names={list(robot.dof_names or [])} "
            f"gripper_indices={list(robot.get_gripper_joint_indices())} "
            f"kp={[round(float(v), 3) for v in _debug_kp]} "
            f"kd={[round(float(v), 3) for v in _debug_kd]}"
        )

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            task_controller.reset_needed = True
            
        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    release_and_convert(
                        video_writer,
                        video_output_path,
                        task_controller.is_success(),
                    )
                    video_writer = None
                    video_output_path = None

                # Determine how many episodes to run depending on mode.
                # Replay defaults to the whole dataset — the top-level
                # `max_episodes` is a collect/infer knob and is NOT a cap here.
                # Set `replay.max_episodes` to cap a replay run (or
                # `replay.episode_indices` to pick a subset); the effective
                # count is logged once so a run never silently differs from
                # what the config appears to ask for.
                if cfg.mode == "replay":
                    n_dataset = len(task_controller._replay_loader)
                    _replay_cap = getattr(getattr(cfg, "replay", None), "max_episodes", None)
                    max_episodes = min(n_dataset, int(_replay_cap)) if _replay_cap else n_dataset
                    if not _replay_count_logged:
                        source = ("replay.max_episodes" if _replay_cap
                                  else f"whole dataset (top-level max_episodes={cfg.max_episodes} "
                                       "does not apply to replay)")
                        logger.info(f"[Replay] dataset has {n_dataset} episodes; running "
                                    f"{max_episodes} — capped by {source}")
                        _replay_count_logged = True
                else:
                    max_episodes = cfg.max_episodes

                # An episode that ended on max_steps never reached the `if done:`
                # branch, so finalize it here: report why it failed and clear the
                # per-episode diagnostics before they leak into the next episode.
                # `_steps_since_reset` guards the very first pass (scene setup, and
                # replay's initial reset), where no episode has run yet — without it
                # every run logged a spurious "Episode 0 failed" and every parsed
                # stat came out one failure too high.
                if not _episode_finalized and _steps_since_reset > 0:
                    # Log the standard "failed" line too: a timed-out episode is a
                    # failed one, and the campaign log parsers count these lines.
                    logger.warning(f"Episode {task_controller.episode_num} failed. "
                                   f"(ended on max_steps="
                                   f"{getattr(getattr(cfg, 'task', None), 'max_steps', '?')})")
                    _reset_episode_diagnostics(robot)
                    task_controller.print_failure_reason()
                episode_just_finished = _steps_since_reset > 0
                _episode_finalized = False
                _steps_since_reset = 0

                # Check if we've completed all episodes BEFORE setting up the next one.
                # In infer/replay `episode_num` is only incremented by reset(), which
                # has not run yet for the episode that just ended — count it here, or
                # the run does max_episodes+1 episodes and drops the last one from the
                # stats. (In collect, `episode_num` is the data collector's count of
                # WRITTEN episodes, which is already final.)
                # Real collection deliberately caps SUCCESSFUL datasets, so failed
                # attempts do not count. Smoke tests use MockCollector and need a
                # finite attempt cap; its current failed attempt is not reflected in
                # _episode_num until reset(), hence the explicit +1 here.
                count_attempts = bool(
                    getattr(getattr(task_controller, "data_collector", None), "counts_attempts", False)
                )
                current_episode = int(episode_just_finished)
                episodes_done = (
                    task_controller.episode_num + current_episode
                    if cfg.mode != "collect" or count_attempts
                    else task_controller.episode_num
                )
                attempts_done = task_controller._episode_num + current_episode
                attempts_exhausted = (
                    args.max_attempts is not None and attempts_done >= args.max_attempts
                )
                if (
                    episodes_done >= max_episodes
                    or attempts_exhausted
                    or getattr(task_controller, "_replay_done", False)
                ):
                    if cfg.mode == "collect" and episode_just_finished:
                        # A real collector reaches max_episodes as soon as the final
                        # successful sample is written. Shutdown therefore happens
                        # before reset(), which normally updates and prints the
                        # controller's attempt statistics. Emit that missing final
                        # line without resetting task/controller state during teardown.
                        final_attempts = task_controller._episode_num + 1
                        final_successes = task_controller.success_count + int(
                            task_controller.is_success()
                        )
                        final_rate = final_successes / max(final_attempts, 1) * 100.0
                        logger.info(
                            "Episode Stats: Success Rate = "
                            f"{final_successes}/{final_attempts} ({final_rate:.2f}%)"
                        )
                    if cfg.mode != "collect" and not getattr(task_controller, "_is_initial_replay_reset", False):
                        # reset() is what counts the episode and prints "Episode Stats",
                        # so the final episode is otherwise missing from the tally.
                        task_controller.reset()
                    if attempts_exhausted and episodes_done < max_episodes:
                        logger.info(
                            f"Attempt limit reached ({attempts_done}/{args.max_attempts}). "
                            "Shutting down."
                        )
                    else:
                        logger.info(f"All {max_episodes} episodes completed. Shutting down.")
                    task_controller.close()
                    # Join the ffmpeg transcode threads BEFORE tearing down the
                    # app: closing first killed the last episode's conversion and
                    # left a half-written *.tmp.mp4 instead of its video.
                    for t in _convert_threads:
                        t.join()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break

                # In replay mode: restore scene FIRST, then reset controller
                # This ensures correct episode numbering and environment setup
                if cfg.mode == "replay":
                    _log_reset_object_poses("before task reset")
                    init_state = task_controller.get_current_init_state()
                    task.reset_with_init_state(init_state)
                    _log_reset_object_poses("after task reset")
                    task_controller.reset()
                else:
                    _log_reset_object_poses("before task reset")
                    task_controller.reset()
                    task.reset()
                    _log_reset_object_poses("after task reset")
                _apply_arm_gains()

                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
            _steps_since_reset += 1
            if action is not None:
                # collect_position_only: apply the scripted action as a PURE joint
                # POSITION command (drop RMPFlow's velocity/effort feed-forward), so
                # collection uses the exact same control law as replay & inference
                # (which only have joint positions). Makes the recorded joints a
                # complete control signal -> collect == replay == policy execution.
                if bool(getattr(cfg, "collect_position_only", False)) \
                        and getattr(action, "joint_positions", None) is not None:
                    # Position-only control law in collect AND in the scripted
                    # phases of replay (recorded replay actions carry no
                    # velocities, so this is a no-op for them). Velocity-ONLY
                    # actions (joint_positions=None, e.g. the pour controller's
                    # wrist rotation on a velocity-switched DOF) are the task's
                    # defining actuation — leave them intact.
                    action.joint_velocities = None
                    action.joint_efforts = None
                # Sync BEFORE potentially stripping the finger channels below —
                # velocity/force modes read the finger target to decide open/close.
                robot.sync_gripper_from_action(action)
                # Replay parity for velocity/force gripper modes: during collect
                # the scripted (RMPFlow) action position-commands only the 7 arm
                # DOFs — fingers are driven solely by apply_gripper_effort(). The
                # replayed 9-DOF action would ALSO position-slam the fingers
                # (drive spring to 0 crushes/ejects what collect's gentle
                # velocity close held), so strip the finger channels and let
                # apply_gripper_effort() reproduce collect's exact actuation.
                if (cfg.mode == "replay"
                        and getattr(robot, "_gripper_control_mode", "position") != "position"
                        and getattr(action, "joint_positions", None) is not None
                        and len(action.joint_positions) > 7):
                    # None = "leave this DOF uncommanded" (dims must stay = DOF count)
                    action.joint_positions = list(action.joint_positions[:7]) + \
                        [None] * (len(action.joint_positions) - 7)
                    if action.joint_velocities is not None and len(action.joint_velocities) > 7:
                        action.joint_velocities = list(action.joint_velocities[:7]) + \
                            [None] * (len(action.joint_velocities) - 7)
                robot.get_articulation_controller().apply_action(action)
            robot.apply_gripper_effort()
            # Optional collect-vs-replay diagnostics. Disabled by default because
            # the per-frame queries and traces are useful for debugging, not normal
            # collection or evaluation.
            if arm_diagnostics_enabled:
                try:
                    # Per-frame arm-tracking error (arm-motion consistency). Track
                    # the frame and joint where the maximum occurs.
                    robot._frame_diag = getattr(robot, "_frame_diag", 0) + 1
                    if action is not None and getattr(action, "joint_positions", None) is not None:
                        _action_positions = list(action.joint_positions)
                        _action_indices = getattr(action, "joint_indices", None)
                        if _action_indices is None:
                            _action_indices = range(len(_action_positions))
                        _command_by_dof = {
                            int(index): float(value)
                            for index, value in zip(_action_indices, _action_positions)
                            if value is not None
                        }
                        _arm_indices = robot.get_arm_joint_indices()
                        _actual_positions = robot.get_joint_positions()
                        _perr = [
                            abs(float(_actual_positions[index]) - _command_by_dof[index])
                            for index in _arm_indices
                            if index in _command_by_dof
                        ]
                    else:
                        _perr = []
                    if _perr:
                        _err = max(_perr)
                        robot._last_err = _err
                        if _err > getattr(robot, "_ep_max_arm_err", 0.0):
                            robot._ep_max_arm_err = _err
                            robot._argmax_diag = (
                                robot._frame_diag,
                                _perr.index(_err),
                                [round(error, 3) for error in _perr],
                            )
                        if not getattr(robot, "_closed_once", False) and robot._frame_diag % 25 == 0:
                            logger.info(
                                f"[APPROACH] mode={cfg.mode} ep={task_controller.episode_num} "
                                f"frame={robot._frame_diag} arm_err={round(_err, 4)} "
                                f"max_joint={_perr.index(_err)} "
                                f"per_joint={[round(error, 3) for error in _perr]}"
                            )

                    # Track grasp/release transitions in both position and effort
                    # gripper modes.
                    _gs = 0
                    try:
                        if int(robot.get_gripper_state()) == 1:
                            _gs = 1
                    except (TypeError, ValueError):
                        pass
                    if (
                        action is not None
                        and getattr(action, "joint_positions", None) is not None
                        and len(action.joint_positions) >= 8
                    ):
                        _legacy_gripper_target = action.joint_positions[7]
                        if _legacy_gripper_target is not None and float(_legacy_gripper_target) <= 0.02:
                            _gs = 1
                    _prev = getattr(robot, "_prev_gs_diag", 0)
                    _ft = getattr(robot, "_finger_trace_left", 0)
                    if _gs == 1 and _prev == 0:
                        robot._finger_trace_left = 60
                    elif _ft > 0:
                        robot._finger_trace_left = _ft - 1
                        if _ft % 5 == 0:
                            _joint_positions = robot.get_joint_positions()
                            _gripper_indices = robot.get_gripper_joint_indices()
                            _fp = {
                                str(robot.dof_names[index]): round(float(_joint_positions[index]), 4)
                                for index in _gripper_indices
                            }
                            logger.info(
                                f"[FINGER] mode={cfg.mode} ep={task_controller.episode_num} "
                                f"t-close={60 - _ft} fingers={_fp}"
                            )

                    if _gs == 1 and _prev == 0:  # grasp instant
                        robot._closed_once = True
                        logger.info(
                            f"[GRASP-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"obj_pos={_round_values(state.get('object_position'))} "
                            f"obj_quat={_object_quaternion(state, task)} "
                            f"arm_err_now={round(float(getattr(robot, '_last_err', 0.0)), 4)} "
                            f"arm_err_peak_approach={round(float(robot._ep_max_arm_err), 4)}"
                        )
                    elif _gs == 0 and _prev == 1:  # release instant
                        logger.info(
                            f"[RELEASE-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"obj_pos={_round_values(state.get('object_position'))} "
                            f"obj_quat={_object_quaternion(state, task)} "
                            f"arm_err_max={round(float(getattr(robot, '_ep_max_arm_err', 0.0)), 4)}"
                        )
                    robot._prev_gs_diag = _gs
                except Exception:
                    logger.exception("Arm diagnostics failed")
            if done:
                if is_success:
                    logger.success(f"Episode {task_controller.episode_num} succeeded.")
                else:
                    logger.warning(f"Episode {task_controller.episode_num} failed.")
                _reset_episode_diagnostics(robot)
                task_controller.print_failure_reason()
                task.on_task_complete(is_success)
                _episode_finalized = True
                continue
            
            if save_video or show_video:
                camera_images = []
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                
                if camera_images:
                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    if show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            video_output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(video_output_path, fourcc, 60.0, (width, height))
                            if not video_writer.isOpened():
                                logger.error(f"Failed to open VideoWriter for {video_output_path}")
                                video_writer = None
                        if video_writer is not None:
                            video_writer.write(combined_img)


if __name__ == "__main__":
    main()
