"""Read-only wrist tracking and collision diagnostics; never change drive gains."""

import numpy as np
from isaacsim.core.api.sensors import RigidContactView


class ArmTelemetry:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.links = [f"{robot.prim_path}/panda_link{i}" for i in range(4, 8)]
        self.links.append(f"{robot.prim_path}/panda_hand")
        self.filters = [f"{robot.prim_path}/panda_link{i}" for i in range(3, 8)]
        self.filters += ["/World/conical_bottle02/mesh", "/World/beaker1/mesh"]
        self.contacts = RigidContactView(
            prim_paths_expr=self.links,
            filter_paths_expr=[self.filters for _ in self.links],
            name="conical_wrist_contact_audit",
            disable_stablization=False,
            max_contact_count=4096,
        )
        self.contacts.initialize()

    def observe(self, dt: float) -> dict:
        view = self.robot._articulation_view
        action = self.robot.get_applied_action()
        return {
            "links": self.links,
            "filters": self.filters,
            "filtered_contacts_N": np.asarray(self.contacts.get_contact_force_matrix(dt=dt)).tolist(),
            "net_contacts_N": np.asarray(self.contacts.get_net_contact_forces(dt=dt)).tolist(),
            "measured_joint_efforts": np.asarray(view.get_measured_joint_efforts()).tolist(),
            "maximum_joint_efforts": np.asarray(view.get_max_efforts()).tolist(),
            "link_names": list(view.body_names),
            "link_masses_kg": np.asarray(view.get_body_masses()).tolist(),
            "link_inertias_kgm2": np.asarray(view.get_body_inertias()).tolist(),
            "self_collisions_enabled": np.asarray(view.get_enabled_self_collisions()).tolist(),
            "joint_velocities": np.asarray(self.robot.get_joint_velocities()).tolist(),
            "joint_velocity_limits": np.asarray(view.get_joint_max_velocities()).tolist(),
            "velocity_targets": None if action.joint_velocities is None else np.asarray(action.joint_velocities).tolist(),
        }
