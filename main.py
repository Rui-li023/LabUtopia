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
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
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
        stderr=subprocess.DEVNULL,
    )
    if ret.returncode == 0:
        os.replace(tmp_path, dest_path)
        os.remove(src_path)
        logger.success(f"Video saved: {dest_path}")
    else:
        logger.error(f"H264 conversion failed for: {src_path}")
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

    robot_kwargs = {"position": np.array(cfg.robot.position)}
    if hasattr(cfg.robot, "default_joint_positions"):
        robot_kwargs["default_joint_positions"] = np.array(cfg.robot.default_joint_positions)
    robot = create_robot(cfg.robot.type, **robot_kwargs)
    logger.info(f"Robot created: {cfg.robot.type}")
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    
    video_writer = None
    video_output_path = None
    task.reset()
    
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
                if task_controller.episode_num >= max_episodes:
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

                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
            if done:
                if is_success:
                    logger.success(f"Episode {task_controller.episode_num} succeeded.")
                else:
                    logger.warning(f"Episode {task_controller.episode_num} failed.")
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
                        video_writer.write(combined_img)


if __name__ == "__main__":
    main()
