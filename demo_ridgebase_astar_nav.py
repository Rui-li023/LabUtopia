import os
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import numpy as np
import yaml
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

from controllers.robot_controllers.ridgebase.ridgebase_controller import RidgebaseController
from utils.a_star import plan_navigation_path, real_to_grid, load_grid

def load_assets_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['assets']

def generate_random_points(x_bounds, y_bounds, grid, attempts=100):
    """Generate random start and end points in free space"""
    W = len(grid[0])
    H = len(grid)
    
    for _ in range(attempts):
        start_x = np.random.uniform(x_bounds[0], x_bounds[1])
        start_y = np.random.uniform(y_bounds[0], y_bounds[1])
        end_x = np.random.uniform(x_bounds[0], x_bounds[1])
        end_y = np.random.uniform(y_bounds[0], y_bounds[1])
        
        i_start, j_start = real_to_grid(
            start_x, start_y, x_bounds, y_bounds, (W, H)
        )
        i_end, j_end = real_to_grid(
            end_x, end_y, x_bounds, y_bounds, (W, H)
        )
        
        if grid[i_start][j_start] == 0 and grid[i_end][j_end] == 0:
            return [start_x, start_y], [end_x, end_y]
            
    raise RuntimeError("Failed to find valid start and end points")

def main():
    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene")
    
    robot_path = "/World/Ridgebase"
    add_reference_to_stage(
        usd_path="assets/robots/ridgeback_franka.usd",
        prim_path=robot_path
    )
    
    from omni.isaac.core.robots import Robot
    robot = Robot(
        prim_path=robot_path,
        name="ridgebase",
        position=np.array([0.0, 0.0, 0.0])
    )
    
    controller = RidgebaseController(
        robot_articulation=robot,
        max_linear_speed=0.02,
        max_angular_speed=1.5,
        position_threshold=0.08,
        angle_threshold=0.1
    )
    
    assets = load_assets_config("config/navigation/navigation_assets.yaml")
    nav_scene = assets[0]  

    add_reference_to_stage(
        usd_path=os.path.abspath(nav_scene['scene_asset_path']), 
        prim_path="/World"
    )
    
    grid, W, H = load_grid(nav_scene['barrier_image_path'])
    
    def reset_navigation():
        """Reset navigation task with new start/end points and path"""
        path_result = None
        while path_result is None:
            try:
                start_point, end_point = generate_random_points(
                    nav_scene['x_bounds'],
                    nav_scene['y_bounds'],
                    grid
                )
                
                task_info = {
                    "asset": nav_scene,
                    "start": start_point,
                    "end": end_point
                }
                
                path_result = plan_navigation_path(task_info)
                if path_result is not None:
                    print(f"Path found! Start: {start_point}, End: {end_point}")
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
                    
                    controller.set_waypoints(waypoints)
                    
                    initial_position = np.array([start_point[0], start_point[1], 0.0])
                    initial_orientation = np.array([0.0, 0.0, waypoints[0][2]])  

                    from omni.isaac.core.utils.rotations import euler_angles_to_quat
                    initial_rotation = euler_angles_to_quat(initial_orientation)
                    
                    robot.set_world_pose(position=initial_position)
                    return True
                    
            except RuntimeError as e:
                print(f"Error: {e}")
                return False
        
        return False

    if not reset_navigation():
        simulation_app.close()
        return
        
    reset_need = False
    world.reset()
    robot.initialize()
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            reset_need = True
            
        if world.is_playing():
            if reset_need:
                world.reset()
                robot.initialize()
                if not reset_navigation():  
                    simulation_app.close()
                    break

                reset_need = False
                
            position, orientation = robot.get_world_pose()
            from omni.isaac.core.utils.rotations import quat_to_euler_angles
            euler_angles = quat_to_euler_angles(orientation, extrinsic=False)
            current_pose = np.array([position[0], position[1], euler_angles[2]])
            
            action, done = controller.get_action(current_pose)
            
            if action is not None:
                robot.apply_action(action)
            
            if done or controller.is_path_complete():
                print("Navigation completed! Resetting new navigation task...")
                reset_need = True
    
    simulation_app.close()

if __name__ == "__main__":
    main()
