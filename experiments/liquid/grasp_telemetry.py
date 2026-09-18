"""Diagnostic-only solver contacts and mass properties; no action/pose writes."""

import numpy as np
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.prims import RigidPrim
from scipy.spatial.transform import Rotation


class GraspTelemetry:
    def __init__(self, robot_path: str) -> None:
        self.bottle_path = "/World/conical_bottle02/mesh"
        self.body = RigidPrim(
            self.bottle_path, name="conical_mass_audit", reset_xform_properties=False, disable_stablization=False
        )
        self.body.initialize()
        self.contacts = RigidContactView(
            prim_paths_expr=[f"{robot_path}/panda_leftfinger", f"{robot_path}/panda_rightfinger"],
            filter_paths_expr=[[self.bottle_path], [self.bottle_path]],
            name="conical_contact_audit",
            disable_stablization=False,
            max_contact_count=2048,
        )
        self.contacts.initialize()

    def observe(self, hand, metrics, dt: float) -> dict:
        mass = float(self.body.get_masses()[0])
        local_com = np.asarray(self.body.get_coms()[0]).reshape(-1, 3)[0]
        position, orientation = self.body.get_world_poses()
        world_com = position[0] + Rotation.from_quat(orientation[0][[1, 2, 3, 0]]).apply(local_com)
        normal_force, points, normals, distances, counts, starts = self.contacts.get_contact_force_data(dt=dt)
        fingers = []
        for count, start in zip(np.asarray(counts).reshape(-1), np.asarray(starts).reshape(-1)):
            selected = slice(int(start), int(start + count))
            forces = np.asarray(normal_force)[selected].reshape(-1)
            contact_points = np.asarray(points)[selected]
            directions = np.asarray(normals)[selected]
            fingers.append(
                {
                    "count": int(count),
                    "normal_force_sum_N": float(np.abs(forces).sum()),
                    "contact_points_world": contact_points.tolist(),
                    "normals_world": directions.tolist(),
                    "separation_m": np.asarray(distances)[selected].reshape(-1).tolist(),
                    "normal_torque_about_hand_Nm": np.cross(contact_points - hand, directions * forces[:, None])
                    .sum(0)
                    .tolist(),
                }
            )
        return {
            "glass_mass_kg": mass,
            "com_local_physics_m": local_com.tolist(),
            "com_world_m": world_com.tolist(),
            "inertia_kgm2": np.asarray(self.body.get_inertias()).tolist(),
            "body_velocity_m_s_rad_s": np.asarray(self.body.get_velocities()).tolist(),
            "glass_gravity_torque_about_hand_Nm": np.cross(world_com - hand, [0, 0, -9.81 * mass]).tolist(),
            "contact_dt": dt,
            "finger_contacts": fingers,
        }
