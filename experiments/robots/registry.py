"""Register the robot variants used by the experimental arm gallery."""

from experiments.robots.definitions.abb_gofa_robotiq.abb_gofa_robotiq import ABBGoFaRobotiq
from experiments.robots.definitions.arx_r5.arx_r5 import ArxR5
from experiments.robots.definitions.arx_x5.arx_x5 import ArxX5
from experiments.robots.definitions.doosan_m0609_robotiq.doosan_m0609_robotiq import DoosanM0609Robotiq
from experiments.robots.definitions.doosan_m1013_robotiq.doosan_m1013_robotiq import DoosanM1013Robotiq
from experiments.robots.definitions.fanuc_crx10ia_robotiq.fanuc_crx10ia_robotiq import FanucCRX10iARobotiq
from experiments.robots.definitions.fanuc_lrmate200id_robotiq.fanuc_lrmate200id_robotiq import (
    FanucLRMate200iDRobotiq,
)
from experiments.robots.definitions.fr3.fr3 import FR3
from experiments.robots.definitions.gen3_robotiq.gen3_robotiq import Gen3Robotiq
from experiments.robots.definitions.iiwa14_robotiq.iiwa14_robotiq import Iiwa14Robotiq
from experiments.robots.definitions.jaco2.jaco2 import Jaco2
from experiments.robots.definitions.rizon4_robotiq.rizon4_robotiq import Rizon4Robotiq
from experiments.robots.definitions.split_aloha_fl.split_aloha_fl import SplitAlohaFrontLeft
from experiments.robots.definitions.techman_tm5_900_robotiq.techman_tm5_900_robotiq import TechmanTM5900Robotiq
from experiments.robots.definitions.unitree_z1.unitree_z1 import UnitreeZ1
from experiments.robots.definitions.ur10e_robotiq.ur10e_robotiq import UR10eRobotiq
from experiments.robots.definitions.ur3e_robotiq.ur3e_robotiq import UR3eRobotiq
from experiments.robots.definitions.ur5e_robotiq.ur5e_robotiq import UR5eRobotiq
from experiments.robots.definitions.widowx_vx300s.widowx_vx300s import WidowXVX300s
from experiments.robots.definitions.xarm6_robotiq.xarm6_robotiq import XArm6Robotiq
from experiments.robots.definitions.xarm7_robotiq.xarm7_robotiq import XArm7Robotiq
from experiments.robots.definitions.yaskawa_hc10_robotiq.yaskawa_hc10_robotiq import YaskawaHC10Robotiq
from factories.robot_factory import register_robot

EXPERIMENTAL_ROBOTS = {
    "abb_gofa_robotiq": ABBGoFaRobotiq,
    "arx_r5": ArxR5,
    "arx_x5": ArxX5,
    "doosan_m0609_robotiq": DoosanM0609Robotiq,
    "doosan_m1013_robotiq": DoosanM1013Robotiq,
    "fanuc_crx10ia_robotiq": FanucCRX10iARobotiq,
    "fanuc_lrmate200id_robotiq": FanucLRMate200iDRobotiq,
    "fr3": FR3,
    "gen3_robotiq": Gen3Robotiq,
    "iiwa14_robotiq": Iiwa14Robotiq,
    "jaco2": Jaco2,
    "rizon4_robotiq": Rizon4Robotiq,
    "split_aloha_fl": SplitAlohaFrontLeft,
    "techman_tm5_900_robotiq": TechmanTM5900Robotiq,
    "unitree_z1": UnitreeZ1,
    "ur10e_robotiq": UR10eRobotiq,
    "ur3e_robotiq": UR3eRobotiq,
    "ur5e_robotiq": UR5eRobotiq,
    "widowx_vx300s": WidowXVX300s,
    "xarm6_robotiq": XArm6Robotiq,
    "xarm7_robotiq": XArm7Robotiq,
    "yaskawa_hc10_robotiq": YaskawaHC10Robotiq,
}


def register_experimental_robots() -> None:
    for name, robot_class in EXPERIMENTAL_ROBOTS.items():
        register_robot(name, robot_class)
