import os
import sys
import argparse
from loguru import logger
from isaacsim import SimulationApp

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
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up simulation app based on arguments
simulation_config = {
    "headless": False,
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

def main():
    hydra.initialize(config_path=args.config_dir, job_name=args.config_name)
    cfg = hydra.compose(config_name=args.config_name)
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")
    logger.info(f"Config loaded: {args.config_name}, run dir: {cfg.multi_run.run_dir}")

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
        show_video = True

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

    task.reset()
    _apply_arm_gains()

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            task_controller.reset_needed = True
            
        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    release_and_convert(video_writer, video_output_path, is_success)
                    video_writer = None
                    video_output_path = None

                # Determine how many episodes to run depending on mode
                if cfg.mode == "replay":
                    max_episodes = len(task_controller._replay_loader)
                else:
                    max_episodes = cfg.max_episodes

                # Check if we've completed all episodes BEFORE setting up the next one
                if task_controller.episode_num >= max_episodes or getattr(task_controller, "_replay_done", False):
                    logger.info(f"All {max_episodes} episodes completed. Shutting down.")
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    for t in _convert_threads:
                        t.join()
                    break

                # In replay mode: restore scene FIRST, then reset controller
                # This ensures correct episode numbering and environment setup
                if cfg.mode == "replay":
                    init_state = task_controller.get_current_init_state()
                    task.reset_with_init_state(init_state)
                    task_controller.reset()
                else:
                    task_controller.reset()
                    task.reset()
                _apply_arm_gains()

                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
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
            # CONSISTENCY-DIAG (no object binding): localize where collect vs
            # replay diverge by checking (1) object position at the grasp and
            # release instants, and (2) arm-tracking error every frame (actual
            # vs commanded joints), reported as a per-episode max.
            try:
                def _round(v):
                    try:
                        return [round(float(x), 4) for x in v]
                    except Exception:
                        return v
                def _oquat():
                    try:
                        _op = state.get('object_path')
                        if _op:
                            _q = task.object_utils.get_world_pose(_op).get('orientation')
                            return [round(float(x), 3) for x in _q]
                    except Exception:
                        return None
                    return None
                # Per-frame arm-tracking error (arm-motion consistency). Track the
                # frame & joint where the max occurs to localize the spike.
                robot._frame_diag = getattr(robot, "_frame_diag", 0) + 1
                if action is not None and getattr(action, "joint_positions", None) is not None \
                        and len(action.joint_positions) >= 7:
                    _cmd_arm = [float(v) for v in action.joint_positions[:7]]
                    _act_arm = [float(v) for v in robot.get_joint_positions()[:7]]
                    _perr = [abs(a - c) for a, c in zip(_act_arm, _cmd_arm)]
                    _err = max(_perr)
                    robot._last_err = _err
                    if _err > getattr(robot, "_ep_max_arm_err", 0.0):
                        robot._ep_max_arm_err = _err
                        robot._argmax_diag = (robot._frame_diag, _perr.index(_err),
                                              [round(e, 3) for e in _perr])
                    # Approach trace: instantaneous error every 25 frames BEFORE the
                    # first gripper close (shows how the arm converges to the grasp).
                    if not getattr(robot, "_closed_once", False) and robot._frame_diag % 25 == 0:
                        logger.info(
                            f"[APPROACH] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"frame={robot._frame_diag} arm_err={round(_err, 4)} "
                            f"max_joint={_perr.index(_err)} per_joint={[round(e, 3) for e in _perr]}"
                        )
                # Closed if EITHER the tracked gripper state says closed (collect &
                # force-replay) OR the commanded finger target is closed (position
                # replay, where the finger is in action.joint_positions[7]). The OR
                # makes the grasp/release transitions fire in every mode.
                _gs = 0
                try:
                    if int(robot.get_gripper_state()) == 1:
                        _gs = 1
                except Exception:
                    pass
                if action is not None and getattr(action, "joint_positions", None) is not None \
                        and len(action.joint_positions) >= 8 and float(action.joint_positions[7]) <= 0.02:
                    _gs = 1
                _prev = getattr(robot, "_prev_gs_diag", 0)
                # Finger-close trace: measured finger DOFs every 5 frames for 60
                # frames after each close transition (collect vs replay contact).
                _ft = getattr(robot, "_finger_trace_left", 0)
                if _gs == 1 and _prev == 0:
                    robot._finger_trace_left = 60
                elif _ft > 0:
                    robot._finger_trace_left = _ft - 1
                    if _ft % 5 == 0:
                        _fp = [round(float(v), 4) for v in robot.get_joint_positions()[7:9]]
                        logger.info(f"[FINGER] mode={cfg.mode} ep={task_controller.episode_num} "
                                    f"t-close={60 - _ft} fingers={_fp}")
                if _gs == 1 and _prev == 0:  # grasp instant
                    robot._closed_once = True
                    logger.info(
                        f"[GRASP-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                        f"obj_pos={_round(state.get('object_position'))} obj_quat={_oquat()} "
                        f"arm_err_now={round(float(getattr(robot, '_last_err', 0.0)), 4)} "
                        f"arm_err_peak_approach={round(float(robot._ep_max_arm_err), 4)}"
                    )
                elif _gs == 0 and _prev == 1:  # release instant
                    logger.info(
                        f"[RELEASE-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                        f"obj_pos={_round(state.get('object_position'))} obj_quat={_oquat()} "
                        f"arm_err_max={round(float(getattr(robot, '_ep_max_arm_err', 0.0)), 4)}"
                    )
                robot._prev_gs_diag = _gs
            except Exception:
                pass
            if done:
                if is_success:
                    logger.success(f"Episode {task_controller.episode_num} succeeded.")
                else:
                    logger.warning(f"Episode {task_controller.episode_num} failed.")
                _am = getattr(robot, "_argmax_diag", (None, None, None))
                logger.info(f"[EP-ARM-ERR] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"max_arm_track_err={round(float(getattr(robot, '_ep_max_arm_err', 0.0)), 4)} "
                            f"at_frame={_am[0]}/{getattr(robot, '_frame_diag', 0)} joint={_am[1]} per_joint_err={_am[2]}")
                robot._ep_max_arm_err = 0.0
                robot._frame_diag = 0
                robot._argmax_diag = (None, None, None)
                robot._closed_once = False
                task_controller.print_failure_reason()
                task.on_task_complete(is_success)
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
