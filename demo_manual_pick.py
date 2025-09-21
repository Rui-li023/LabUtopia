import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import os
import yaml
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.usd
from omni.isaac.core.articulations import Articulation, ArticulationSubset
from omni.isaac.sensor import Camera

from controllers.robot_controllers.ridgebase.ridgebase_controller import RidgebaseController
from controllers.atomic_actions.pick_controller import PickController
from controllers.robot_controllers.grapper_manager import Gripper
from utils.a_star import plan_navigation_path, real_to_grid, load_grid
from utils.object_utils import ObjectUtils
from omni.isaac.core.utils.types import ArticulationAction
from data_collectors.pick_data_collector import PickDataCollector

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

def generate_waypoints_from_path(path_result):
    """Generate waypoints from path result"""
    merged_path_real, _ = path_result
    waypoints = []
    for i in range(len(merged_path_real)):
        x, y, r = merged_path_real[i]
        if i < len(merged_path_real) - 1:
            next_x, next_y, _ = merged_path_real[i + 1]
            theta = np.arctan2(next_y - y, next_x - x)
        else:
            theta = waypoints[-1][2] if waypoints else 0.0
        waypoints.append([x, y, theta])
    return waypoints

def main():

    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    data_collector = PickDataCollector(max_episodes=1)

    
    cameras = []
    camera_configs = [
        {
            "prim_path": "/World/Ridgebase/base_link/Camera_01",
            "name": "cam_01",
            "resolution": (640, 640),
            "position": None,  
            "orientation": None  
        },
        {
            "prim_path": "/World/Ridgebase/base_link/Camera",
            "name": "cam_02",
            "resolution": (640, 640),
            "position": None,
            "orientation": None
        },
        {
            "prim_path": "/World/Ridgebase/panda_hand/Camera",
            "name": "hand_cam",
            "resolution": (640, 640),
            "position": None,
            "orientation": None
        }
    ]

    

    robot_path = "/World/Ridgebase"
    add_reference_to_stage(
        usd_path="assets/robots/ridgeback_franka.usd",
        prim_path=robot_path
    )
    
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
    
    grid, W, H = load_grid(nav_scene['barrier_image_path'])
    gripper_control = Gripper()

    target_position = [-3.0, -0.46]  

    beaker_position = np.array([-3.285, 0.02, 0.48])      
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
    nav_controller = RidgebaseController(
        robot_articulation=robot,
        max_linear_speed=0.02,
        max_angular_speed=1.5,
        position_threshold=0.05,
        angle_threshold=0.02
    )
    
    cspace_controller = RMPFlowController(
            name="rmp_flow",
            robot_articulation=robot
        )
    pick_controller = PickController(
        name="pick_controller",
        cspace_controller=cspace_controller
    )
    franka_subset = ArticulationSubset(
        robot,
        ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
    )
    
    waypoints = generate_waypoints_from_path(path_result)
    nav_controller.set_waypoints(waypoints, np.pi/2)

    world.reset()
    robot.initialize()
    navigation_done = False
    pick_started = False
    pick_success = False
    reset_needed = False
    initial_beaker_position = None  
    
    frame_id = 0
    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped():
            reset_needed = True
        
        if world.is_playing():
            if reset_needed:
                start_position, path_result = get_valid_start_position_and_path(nav_scene, target_position)
                if start_position is None:
                    print("Reset failed: Cannot find valid start position and path!")
                    continue
                world.reset()
                robot.set_world_pose(
                    position=np.array([start_position[0], start_position[1], 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
            
                waypoints = generate_waypoints_from_path(path_result)
                
                robot.initialize()
                pick_controller.reset()
                nav_controller.set_waypoints(waypoints, np.pi/2)
                initial_beaker_position = None  
                reset_needed = False
                navigation_done = False
                pick_started = False
                pick_success = False
                frame_id = 0
            
            frame_id += 1
            if frame_id < 5:  
                continue
            elif frame_id > 3000:  
                print("Exceeded maximum frame limit - Failed")
                data_collector.clear_cache()  
                reset_needed = True
                frame_id = 0
                continue
        
            
            joint_positions = robot.get_joint_positions()
            robot_position = robot.get_world_pose()[0]
            current_beaker_position = ObjectUtils.get_instance().get_geometry_center(object_path="/World/beaker")
            
            
            camera_data = {}
            for camera in cameras:
                frame = camera.get_current_frame()
                camera_data[camera.name + "_rgb"] = frame["rgba"][..., :3]  
            
            
            if initial_beaker_position is None:
                initial_beaker_position = current_beaker_position.copy()
            
            
            data_collector.cache_step(
                joint_positions,
                robot_position,
                current_beaker_position,
                **camera_data  
            )
            
            if not navigation_done:

                position, orientation = robot.get_world_pose()
                euler_angles = quat_to_euler_angles(orientation)
                current_pose = np.array([position[0], position[1], euler_angles[2]])
                
                action, done = nav_controller.get_action(current_pose)
                if action is not None:
                    robot.apply_action(action)
                
                if done:
                    print("Navigation completed, starting pick task!")
                    navigation_done = True
                    pick_started = True
                    
            elif pick_started and not pick_success:
                
                pose = ObjectUtils.get_instance().get_object_xform_position(object_path="/World/Ridgebase/panda_link0")
                quat = ObjectUtils.get_instance().get_transform_quat(object_path="/World/Ridgebase/panda_link0", w_first=True)
                # quat = robot.get_world_pose()[1]
                cspace_controller.rmp_flow.set_robot_base_pose(pose, quat)
                pick_controller.set_robot_position(pose)

                joint_positions = franka_subset.get_joint_positions()
                end_effector_position = ObjectUtils.get_instance().get_object_xform_position(object_path="/World/Ridgebase/endeffector")
                pick_position = current_beaker_position.copy()
                action = pick_controller.forward(
                    picking_position=pick_position,
                    current_joint_positions=joint_positions,
                    object_name="beaker",
                    object_size=np.array([0.06, 0.06, 0]),
                    gripper_control=gripper_control,
                    gripper_position=end_effector_position,
                    end_effector_orientation=R.from_euler('xyz', np.radians([-90, 90, 0])).as_quat(),
                )
                if action is not None:
                    if action.joint_indices is not None:
                        indices = action.joint_indices
                        action = ArticulationAction(
                            joint_positions=action.joint_positions, joint_velocities=action.joint_velocities, joint_indices=indices
                        )
                        robot.apply_action(action)
                    else:
                        franka_subset.apply_action(action.joint_positions, action.joint_velocities)
                
                if pick_controller.is_done():
                    print("Pick task completed!")
                    
                    pick_success = (current_beaker_position[2] - initial_beaker_position[2]) > 0.1
                    
                    if pick_success:
                        print("Pick successful - object lifted!")
                        if data_collector.save_episode(episode_id=0, success=True):
                            print(f"Successfully saved episode {data_collector.current_episode-1}")
                        else:
                            print("Max episodes reached or save failed")
                    else:
                        print("Pick failed - object not lifted enough")
                        data_collector.clear_cache()  
                    
                    if data_collector.is_collection_complete():
                        print("Data collection complete!")
                        data_collector.close()  
                        break
                    
                    reset_needed = True
                    frame_id = 0
    
    simulation_app.close()

if __name__ == "__main__":
    main()