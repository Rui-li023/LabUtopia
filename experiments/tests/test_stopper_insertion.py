"""Require genuine released insertion and preserve the original glass asset."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from experiments.tests.test_level2_precision import PlaceController, StopperController


class StopperInsertionSuccessTests(unittest.TestCase):
    def controller(self):
        controller = object.__new__(StopperController)
        controller.cfg = SimpleNamespace(
            task=SimpleNamespace(
                physical_insertion=True,
                insertion_depth=0.0132,
                minimum_insertion_depth=0.010,
                maximum_insertion_depth=0.018,
                placement_xy_tolerance=0.0025,
                placement_tilt_tolerance_deg=8,
            )
        )
        controller.initial_size = np.array([0.036, 0.036, 0.030])
        controller._previous_object_position = None
        controller.state = {
            "object_tip_position": np.array([0.2, -0.2, 0.886]),
            "target_mouth_position": np.array([0.2, -0.2, 0.900]),
            "object_position": np.array([0.2, -0.2, 0.901]),
            "target_position": np.array([0.2, -0.2, 0.830]),
            "target_size": np.array([0.08, 0.08, 0.14]),
            "object_up": np.array([0.0, 0.0, 1.0]),
            "target_up": np.array([0.0, 0.0, 1.0]),
            "gripper_position": np.array([0.2, -0.2, 1.10]),
        }
        return controller

    def test_released_seated_stopper_requires_a_previous_stable_frame(self):
        controller = self.controller()
        self.assertFalse(controller._check_placed_on_target())
        self.assertTrue(controller._check_placed_on_target())
        self.assertEqual(controller._last_failure_reason, "")

    def test_hover_rim_or_excessive_depth_cannot_pass(self):
        for depth in (-0.005, 0, 0.008, 0.021, 0.05):
            controller = self.controller()
            controller.state["object_tip_position"][2] = 0.900 - depth
            controller._check_placed_on_target()
            self.assertFalse(controller._check_placed_on_target(), depth)

    def test_off_axis_tilted_held_moving_or_missing_tip_cannot_pass(self):
        for condition in ("off_axis", "tilted", "held", "moving", "missing"):
            controller = self.controller()
            controller._check_placed_on_target()
            if condition == "off_axis":
                controller.state["object_tip_position"][0] += 0.01
            elif condition == "tilted":
                controller.state["object_up"] = np.array([0.5, 0, np.sqrt(0.75)])
            elif condition == "held":
                controller.state["gripper_position"] = controller.state["object_position"] + [0, 0, 0.02]
            elif condition == "moving":
                controller._previous_object_position[2] -= 0.002
            else:
                controller.state["object_tip_position"] = None
            self.assertFalse(controller._check_placed_on_target(), condition)

    def test_goal_uses_mouth_marker_not_target_bounding_box(self):
        controller = self.controller()
        controller.state["target_position"][:] = 9
        controller.state["target_size"][:] = 9
        np.testing.assert_allclose(controller._goal_position(controller.state), [0.2, -0.2, 0.9018])

    def test_after_release_pose_does_not_chase_the_stopper(self):
        controller = self.controller()
        controller.place_controller = SimpleNamespace(_event=3)
        controller._insertion_hand_target = np.array([0.2, -0.2, 0.92])
        controller._placement_hand_orientation = np.array([0, 1, 0, 0])
        position, _ = controller._insertion_pose({})
        np.testing.assert_array_equal(position, [0.2, -0.2, 0.92])

    def feedback_controller(self):
        controller = self.controller()
        controller.place_controller = SimpleNamespace(_event=1, _events_dt=[0.1] * 6)
        controller._grasp_tip_distance = 0.020
        controller._insertion_alignment_steps = 0
        controller._insertion_approach_ready = False
        controller.state["object_tip_position"] = np.array([0.2, -0.2, 0.9468])
        controller.state["gripper_position"] = controller.state["object_tip_position"] + [0, 0, 0.020]
        controller.rmp_controller = SimpleNamespace(
            articulation_rmp=SimpleNamespace(
                get_active_joints_subset=lambda: SimpleNamespace(get_joint_positions=lambda: np.zeros(7))
            ),
            rmp_flow=SimpleNamespace(
                get_end_effector_pose=lambda joints: (controller.state["gripper_position"].copy(), np.eye(3))
            ),
        )
        return controller

    def test_tilted_stopper_stays_above_lip_and_correction_is_bounded(self):
        controller = self.feedback_controller()
        controller.state["object_up"] = Rotation.from_euler("x", 12, degrees=True).apply([0, 0, 1])
        for _ in range(10):
            position, orientation = controller._insertion_pose(controller.state)
        self.assertFalse(controller._insertion_approach_ready)
        self.assertGreater(position[2], 0.96)
        correction = Rotation.from_quat(orientation[[1, 2, 3, 0]])
        self.assertAlmostEqual(np.degrees(correction.magnitude()), 3)
        self.assertAlmostEqual(controller.place_controller.placement_feature_target[2], 0.8868)

    def test_only_stably_aligned_stopper_can_descend(self):
        controller = self.feedback_controller()
        for _ in range(4):
            position, _ = controller._insertion_pose(controller.state)
            self.assertGreater(position[2], 0.96)
        position, _ = controller._insertion_pose(controller.state)
        self.assertTrue(controller._insertion_approach_ready)
        self.assertAlmostEqual(position[2], 0.9068)

    def test_lost_grasp_stops_feedback_without_chasing(self):
        controller = self.feedback_controller()
        previous, _ = controller._insertion_pose(controller.state)
        controller.state["object_tip_position"] -= [0, 0, 0.1]
        position, _ = controller._insertion_pose(controller.state)
        np.testing.assert_array_equal(position, previous)
        self.assertEqual(controller.place_controller._event, 6)
        self.assertIn("left the gripper", controller.place_controller._phase_failure_reason)

    def test_insertion_position_uses_the_solver_tool_frame(self):
        controller = self.feedback_controller()
        controller._insertion_approach_ready = True
        frame_offset = np.array([0.001, -0.002, 0.007])
        controller.rmp_controller.rmp_flow.get_end_effector_pose = lambda joints: (
            controller.state["gripper_position"] + frame_offset,
            np.eye(3),
        )
        position, _ = controller._insertion_pose(controller.state)
        np.testing.assert_allclose(position, np.array([0.2, -0.2, 0.9068]) + frame_offset)

    def test_transport_ramps_speed_and_forbids_early_descent(self):
        controller = self.feedback_controller()
        controller._transfer_waypoint = np.array([0.2, 0.3, 0.93])
        controller._transfer_step = 0.0
        final = np.array([0.2, -0.2, 0.87])
        previous = controller._transfer_waypoint.copy()
        for index in range(300):
            position = controller._transport_position(final)
            step = np.linalg.norm(controller._transfer_waypoint - previous)
            self.assertLessEqual(step, 0.002 + 1e-12)
            if index == 0:
                self.assertAlmostEqual(step, 0.0001)
                self.assertFalse(controller.place_controller.approach_arrival_allowed)
            previous = controller._transfer_waypoint.copy()
        np.testing.assert_allclose(position, final)
        self.assertTrue(controller.place_controller.approach_arrival_allowed)

    def test_atomic_place_cannot_descend_at_an_intermediate_waypoint(self):
        robot = MagicMock()
        robot.num_gripper_joints = 2
        place = PlaceController("test", MagicMock(), robot=robot, precision=True)
        goal = np.array([0.2, -0.2, 0.9])
        hover = goal + np.array([0, 0, 0.06])
        place.approach_arrival_allowed = False
        place._execute_phase(goal, None, np.zeros(9), MagicMock(), hover, 0.06, 0)
        self.assertEqual(place._event, 0)
        place.approach_arrival_allowed = True
        place._execute_phase(goal, None, np.zeros(9), MagicMock(), hover, 0.06, 0)
        self.assertEqual(place._event, 1)
        place.approach_arrival_allowed = False
        place.reset()
        self.assertTrue(place.approach_arrival_allowed)


class StopperInsertionAssetTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        self.original = Usd.Stage.Open(str(root / "lab_002.usd"))
        self.fixed = Usd.Stage.Open(str(root / "stopper_insertion/lab_002_stopper_insertion.usda"))

    def test_original_flask_visible_geometry_uv_and_material_are_unchanged(self):
        path = "/World/round_bottom_flask_01/mesh_001"
        before, after = [stage.GetPrimAtPath(path) for stage in (self.original, self.fixed)]
        for name in ("points", "faceVertexCounts", "faceVertexIndices", "primvars:st", "primvars:st:indices"):
            self.assertEqual(str(before.GetAttribute(name).Get()), str(after.GetAttribute(name).Get()), name)
        self.assertEqual(
            before.GetRelationship("material:binding").GetTargets(),
            after.GetRelationship("material:binding").GetTargets(),
        )

    def test_neck_colliders_do_not_span_the_open_axis(self):
        path = "/World/round_bottom_flask_01/mesh_001"
        necks = [p for p in self.fixed.GetPrimAtPath(path).GetChildren() if p.GetName().startswith("OpenNeck_")]
        self.assertEqual(len(necks), 96)
        for prim in necks:
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()) * 0.000067
            # Every narrow sector stays outside the true minimum bore radius.
            radial = points[:, [0, 2]]
            direction = radial.mean(axis=0)
            direction /= np.linalg.norm(direction)
            self.assertGreater(float(np.min(radial @ direction)), 0.012)
            self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), "convexHull")
        self.assertFalse(UsdPhysics.CollisionAPI(self.fixed.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get())

    def test_stopper_topology_uv_and_free_dynamics_are_preserved(self):
        path = "/World/rubber_stopper/mesh_003"
        for name in ("faceVertexCounts", "faceVertexIndices", "primvars:st", "primvars:st:indices"):
            self.assertEqual(
                str(self.original.GetPrimAtPath(path).GetAttribute(name).Get()),
                str(self.fixed.GetPrimAtPath(path).GetAttribute(name).Get()),
            )
        for body_path in (path, "/World/round_bottom_flask_01/mesh_001"):
            body = UsdPhysics.RigidBodyAPI(self.fixed.GetPrimAtPath(body_path))
            self.assertTrue(body.GetRigidBodyEnabledAttr().Get())
            self.assertFalse(body.GetKinematicEnabledAttr().Get())
        points = np.asarray(UsdGeom.Mesh(self.fixed.GetPrimAtPath(path)).GetPointsAttr().Get()) * 0.00006
        size = np.ptp(points, axis=0)
        self.assertAlmostEqual(size[1], 0.030, places=6)
        self.assertGreater(size[0], 0.035)
        self.assertLess(size[0], 0.0365)


if __name__ == "__main__":
    unittest.main()
