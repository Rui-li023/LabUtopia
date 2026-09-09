from isaacsim.core.api.robots.robot import Robot

from robots.franka.franka import Franka
from robots.piper.piper import Piper
from robots.ridgebase_franka.ridgebase import Ridgebase

_robot_registry: dict[str, type[Robot]] = {}


def register_robot(name: str, robot_class: type[Robot]) -> None:
    _robot_registry[name] = robot_class


def create_robot(robot_type: str, *args, **kwargs) -> Robot:
    if robot_type not in _robot_registry:
        raise ValueError(f"Unknown robot type: '{robot_type}'. Available: {list(_robot_registry.keys())}")
    return _robot_registry[robot_type](*args, **kwargs)


register_robot("franka", Franka)
register_robot("ridgebase", Ridgebase)
register_robot("piper", Piper)
