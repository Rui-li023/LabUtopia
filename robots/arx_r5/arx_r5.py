# SPDX-License-Identifier: Apache-2.0

"""ARX R5 - same kinematic family as the X5.

Identical joint layout, gripper and ``gripper_center`` frame; only joint1's range
differs (+-2.62/2.17 rad against the X5's -2.87/3.14) and the meshes. Everything
arm-specific therefore lives in the generated Lula files and the USD, so this is a
thin subclass rather than a copy.
"""

import os

from robots.arx_x5.arx_x5 import ArxX5


class ArxR5(ArxX5):
    """ARX R5 robotic arm."""

    def __init__(self, prim_path: str = "/World/ArxR5", name: str = "arx_r5", usd_path=None, **kwargs) -> None:
        if usd_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            usd_path = os.path.join(repo_root, "assets/robots/arx_r5.usd")
        super().__init__(prim_path=prim_path, name=name, usd_path=usd_path, **kwargs)

    @property
    def motion_config(self) -> dict:
        """R5's own Lula description; the X5's joint1 range does not apply here."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(package_dir, "rmpflow")
        return {
            "end_effector_frame_name": "gripper_center",
            "maximum_substep_size": 0.00334,
            "robot_description_path": os.path.join(rmpflow_dir, "robot_descriptor.yaml"),
            "urdf_path": os.path.join(package_dir, "arx_r5.urdf"),
            "rmpflow_config_path": os.path.join(rmpflow_dir, "arx_r5_rmpflow_common.yaml"),
        }
