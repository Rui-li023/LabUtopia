import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import os
import yaml
import hydra
import torch
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import quat_to_euler_angles
import omni.usd
from collections import deque
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer
from omni.isaac.core.utils.types import ArticulationAction

from utils.a_star import plan_navigation_path, load_grid
from utils.object_utils import ObjectUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_assets_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['assets']

def generate_random_start_position():
    """Generate random start position within bounds"""
    x_range = [-7.92, 9.17]
    y_range = [-1.52, 4.17]
    
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    
    return [x, y]

def get_valid_start_position_and_path(nav_scene, target_position, max_attempts=50):
    """Try to get valid start position and corresponding path"""
    for _ in range(max_attempts):
        start_position = generate_random_start_position()
        task_info = {
            "asset": nav_scene,
            "start": start_position,
            "end": target_position
        }
        path_result = plan_navigation_path(task_info)
        
        if path_result is not None:
            return start_position, path_result
    return None, None

def main():
    
    cfg = hydra.initialize(config_path="config", job_name="nav_pick")
    cfg = hydra.compose(config_name="nav_pick")

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")

    
    camera_configs = [
        {
            "prim_path": "/World/Ridgebase/base_link/Camera_01",
            "name": "cam_01",
            "resolution": (84, 84),
        },
        {
            "prim_path": "/World/Ridgebase/base_link/Camera",
            "name": "cam_02", 
            "resolution": (84, 84),
        },
        {
            "prim_path": "/World/Ridgebase/panda_hand/Camera",
            "name": "cam_03",
            "resolution": (84, 84),
        }
    ]

    
    robot_path = "/World/Ridgebase"
    add_reference_to_stage(
        usd_path="assets/robots/ridgeback_franka.usd",
        prim_path=robot_path
    )
    
    
    cameras = []
    for cam_cfg in camera_configs:
        camera = Camera(
            prim_path=cam_cfg["prim_path"],
            name=cam_cfg["name"],
            resolution=cam_cfg["resolution"]
        )
        camera.initialize()
        cameras.append(camera)

    stage = omni.usd.get_context().get_stage()
    ObjectUtils.get_instance(stage)
    
    
    assets = load_assets_config("config/navigation/navigation_assets.yaml")
    nav_scene = assets[0]
    
    add_reference_to_stage(
        usd_path=os.path.abspath(nav_scene['scene_asset_path']), 
        prim_path="/World"
    )
    
    add_reference_to_stage(
        usd_path="assets/objects/beaker.usd",
        prim_path="/World/Desk1/beaker"
    )

    target_position = [-3.0, -0.46]
    start_position, path_result = get_valid_start_position_and_path(nav_scene, target_position)
    
    if start_position is None:
        print("Cannot find valid start position and path!")
        simulation_app.close()
        return

    
    robot = Robot(
        prim_path=robot_path,
        name="ridgebase",
        position=np.array([start_position[0], start_position[1], 0.0])
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfg.infer.policy_model_path, map_location=device)
    policy_config = OmegaConf.load(cfg.infer.policy_config_path)
    
    policy = hydra.utils.instantiate(policy_config.policy)
    policy.load_state_dict(checkpoint['state_dicts']['model'])
    policy.eval()
    policy.to(device)

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(cfg.infer.normalizer_path, map_location=device))
    normalizer.to(device)
    policy.set_normalizer(normalizer)

    
    action_queue = deque()
    
    n_obs_steps = policy_config.n_obs_steps
    obs_history = {
        cam_cfg["name"]: deque(maxlen=n_obs_steps)
        for cam_cfg in camera_configs
    }
    pose_history = deque(maxlen=n_obs_steps)

    
    world.reset()
    robot.initialize()
    initial_beaker_position = None
    success_counter = 0
    REQUIRED_SUCCESS_STEPS = 120
    reset_needed = False
    frame_skip = 5
    frame_count = 0

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            reset_needed = True
            
        if world.is_playing():
            if reset_needed:
                start_position, path_result = get_valid_start_position_and_path(nav_scene, target_position)
                if start_position is None:
                    continue
                    
                world.reset()
                robot.set_world_pose(
                    position=np.array([start_position[0], start_position[1], 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
                
                robot.initialize()
                initial_beaker_position = None
                reset_needed = False
                success_counter = 0
                
                action_queue.clear()
                
                obs_history = {k: deque(maxlen=n_obs_steps) for k in obs_history.keys()}
                pose_history = deque(maxlen=n_obs_steps)
                frame_count = 0
                continue
            
            if frame_count < frame_skip:
                frame_count += 1
                continue
            
            
            current_beaker_position = ObjectUtils.get_instance().get_geometry_center(object_path="/World/beaker")
            if initial_beaker_position is None:
                initial_beaker_position = current_beaker_position.copy()

            
            joint_positions = robot.get_joint_positions()
            pose_history.append(joint_positions)
            
            for camera in cameras:
                frame = camera.get_current_frame()
                rgb_data = frame["rgba"][..., :3].transpose(2, 0, 1)
                obs_history[camera.name].append(rgb_data)

            
            histories_complete = (
                len(pose_history) == n_obs_steps and
                all(len(hist) == n_obs_steps for hist in obs_history.values())
            )

            
            if len(action_queue) == 0 and histories_complete and len(obs_history) > 0:
                
                obs_dict = {
                    cam_name: torch.from_numpy(np.stack(list(hist))).float().to(device) / 255.0
                    for cam_name, hist in obs_history.items()
                }
                obs_dict['agent_pose'] = torch.from_numpy(
                    np.stack(list(pose_history))
                ).float().to(device)
                
                
                obs_dict = {k: v.unsqueeze(0) if v.ndim == 3 else v 
                          for k, v in obs_dict.items()}
                
                
                with torch.no_grad():
                    prediction = policy.predict_action(obs_dict)
                    predicted_positions = prediction['action'][0].cpu().numpy()
                    
                
                for pos in predicted_positions:
                    action_queue.append(pos)
            
            
            if len(action_queue) > 0:
                current_action = action_queue.popleft()
                if current_action is not None:
                    action = ArticulationAction(
                        joint_positions=current_action,
                        joint_indices=np.arange(12),
                        joint_velocities=None
                    )
                    robot.apply_action(action)
                else:
                    print("Warning: current_action is None, skip this step.")
            
            
            if current_beaker_position[2] > initial_beaker_position[2] + 0.1:
                success_counter += 1
            else:
                success_counter = 0

            if success_counter >= REQUIRED_SUCCESS_STEPS:
                print("Task completed successfully!")
                reset_needed = True

    simulation_app.close()

if __name__ == "__main__":
    main()
