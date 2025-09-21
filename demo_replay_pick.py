import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import os
import h5py
import yaml
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
import omni.usd
from utils.object_utils import ObjectUtils

def load_assets_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['assets']

def get_total_episodes(dataset_path):
    """Get total number of episodes in the dataset"""
    with h5py.File(dataset_path, 'r') as f:
        return len([k for k in f.keys() if k.startswith('episode_')])

def load_episode_data(h5_file, episode_id):
    """Load trajectory data for specific episode"""
    episode_group = h5_file[f'episode_{episode_id:04d}']
    agent_poses = episode_group['agent_pose'][:]
    robot_init_pos = episode_group['robot_init_position'][:]
    beaker_position = episode_group['beaker_position'][:]
    success = episode_group.attrs.get('success', False)
    return agent_poses, robot_init_pos[0], beaker_position[0], success

def main():
    
    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    
    
    assets = load_assets_config("config/navigation/navigation_assets.yaml")
    nav_scene = assets[0]
    
    
    add_reference_to_stage(
        usd_path=os.path.abspath(nav_scene['scene_asset_path']), 
        prim_path="/World"
    )
    
    
    robot_path = "/World/Ridgebase"
    add_reference_to_stage(
        usd_path="assets/robots/ridgeback_franka.usd",
        prim_path=robot_path
    )
    
    
    add_reference_to_stage(
        usd_path="assets/objects/beaker.usd",
        prim_path="/World/Desk1/beaker"
    )
    
    stage = omni.usd.get_context().get_stage()
    object_utils = ObjectUtils.get_instance(stage)

    
    dataset_path = "outputs/nav_dataset/pick_data.hdf5"
    total_episodes = get_total_episodes(dataset_path)
    print(f"Found {total_episodes} episodes in dataset")
    
    h5_file = h5py.File(dataset_path, 'r')
    current_episode = 0
    
    while simulation_app.is_running():
        
        agent_poses, init_position, beaker_position, success = load_episode_data(h5_file, current_episode)
        print(f"Replaying episode {current_episode}, success: {success}")
        
        
        robot = Robot(
            prim_path=robot_path,
            name="ridgebase",
            position=np.array([init_position[0], init_position[1], 0.0])
        )
        
        
        world.reset()
        robot.initialize()
        
        
        current_step = 0
        while simulation_app.is_running():
            if world.is_playing():
                if current_step < len(agent_poses):
                    
                    target_joint_positions = agent_poses[current_step]
                    
                    
                    action = ArticulationAction(
                        joint_positions=target_joint_positions,
                        joint_velocities=None
                    )
                    robot.apply_action(action)
                    current_step += 1
                else:
                    print(f"Episode {current_episode} replay completed!")
                    break
            
            world.step(render=True)
        
        
        current_episode += 1
        if current_episode >= total_episodes:
            print("All episodes replayed!")
            break
    
    h5_file.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
