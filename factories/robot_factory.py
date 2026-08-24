from isaacsim.core.api.robots.robot import Robot

from robots.abb_gofa_robotiq.abb_gofa_robotiq import ABBGoFaRobotiq
from robots.arx_r5.arx_r5 import ArxR5
from robots.arx_x5.arx_x5 import ArxX5
from robots.doosan_m0609_robotiq.doosan_m0609_robotiq import DoosanM0609Robotiq
from robots.doosan_m1013_robotiq.doosan_m1013_robotiq import DoosanM1013Robotiq
from robots.fanuc_crx10ia_robotiq.fanuc_crx10ia_robotiq import FanucCRX10iARobotiq
from robots.fanuc_lrmate200id_robotiq.fanuc_lrmate200id_robotiq import (
    FanucLRMate200iDRobotiq,
)
from robots.fr3.fr3 import FR3
from robots.franka.franka import Franka
from robots.gen3_robotiq.gen3_robotiq import Gen3Robotiq
from robots.iiwa14_robotiq.iiwa14_robotiq import Iiwa14Robotiq
from robots.jaco2.jaco2 import Jaco2
from robots.piper.piper import Piper
from robots.ridgebase_franka.ridgebase import Ridgebase
from robots.rizon4_robotiq.rizon4_robotiq import Rizon4Robotiq
from robots.split_aloha_fl.split_aloha_fl import SplitAlohaFrontLeft
from robots.techman_tm5_900_robotiq.techman_tm5_900_robotiq import TechmanTM5900Robotiq
from robots.unitree_z1.unitree_z1 import UnitreeZ1
from robots.ur3e_robotiq.ur3e_robotiq import UR3eRobotiq
from robots.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq
from robots.ur10e_robotiq.ur10e_robotiq import UR10eRobotiq
from robots.widowx_vx300s.widowx_vx300s import WidowXVX300s
from robots.xarm6_robotiq.xarm6_robotiq import XArm6Robotiq
from robots.xarm7_robotiq.xarm7_robotiq import XArm7Robotiq
from robots.yaskawa_hc10_robotiq.yaskawa_hc10_robotiq import YaskawaHC10Robotiq

_robot_registry: dict[str, type[Robot]] = {}


def register_robot(name: str, robot_class: type[Robot]):
    _robot_registry[name] = robot_class


def create_robot(robot_type: str, *args, **kwargs) -> Robot:
    if robot_type not in _robot_registry:
        raise ValueError(f"Unknown robot type: '{robot_type}'. Available: {list(_robot_registry.keys())}")
    return _robot_registry[robot_type](*args, **kwargs)


register_robot("franka", Franka)
register_robot("fr3", FR3)
register_robot("ridgebase", Ridgebase)
register_robot("piper", Piper)
register_robot("arx_x5", ArxX5)
register_robot("arx_r5", ArxR5)
register_robot("widowx_vx300s", WidowXVX300s)
register_robot("ur5e_robotiq", UR5eRobotiq)
register_robot("ur3e_robotiq", UR3eRobotiq)
register_robot("ur10e_robotiq", UR10eRobotiq)
register_robot("xarm6_robotiq", XArm6Robotiq)
register_robot("xarm7_robotiq", XArm7Robotiq)
register_robot("gen3_robotiq", Gen3Robotiq)
register_robot("iiwa14_robotiq", Iiwa14Robotiq)
register_robot("doosan_m0609_robotiq", DoosanM0609Robotiq)
register_robot("doosan_m1013_robotiq", DoosanM1013Robotiq)
register_robot("fanuc_lrmate200id_robotiq", FanucLRMate200iDRobotiq)
register_robot("jaco2", Jaco2)
register_robot("rizon4_robotiq", Rizon4Robotiq)
register_robot("split_aloha_fl", SplitAlohaFrontLeft)
register_robot("abb_gofa_robotiq", ABBGoFaRobotiq)
register_robot("fanuc_crx10ia_robotiq", FanucCRX10iARobotiq)
register_robot("yaskawa_hc10_robotiq", YaskawaHC10Robotiq)
register_robot("techman_tm5_900_robotiq", TechmanTM5900Robotiq)
register_robot("unitree_z1", UnitreeZ1)
