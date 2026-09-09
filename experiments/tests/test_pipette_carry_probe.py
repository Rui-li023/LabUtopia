"""Test carry execution gates without loading the simulator SDK."""

import ast
import unittest
from itertools import pairwise, product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.pipette.run_pipette_fixture import smooth_joint_path

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


def load_execution_class():
    source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CarryRRTProbe")
    namespace = {"np": np, "Rotation": Rotation, "ArticulationAction": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["CarryRRTProbe"]


class CarryExecutionTests(unittest.TestCase):
    def test_carry_margin_cannot_weaken_default_or_exceed_bounded_buffer(self):
        carry = load_execution_class()
        for margin in (0.001, 0.021, float("nan"), float("inf")):
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "Carry planning margin"):
                carry(None, None, None, None, None, None, 35, None, None, carry_margin=margin)

    def _pose_optimizer(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        names = {
            "screen_joint_path",
            "shortcut_joint_path",
            "prepare_approach_path",
            "rank_pose_ik",
            "improve_pose_path",
            "screened_hold_compensation",
            "screen_pose_continuation",
            "seat_heading_command_state",
            "forecast_seated_pose",
            "grasp_orientation_candidates",
            "choose_seat_grasp",
        }
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"np": np, "pairwise": pairwise, "product": product, "Rotation": Rotation}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        return namespace

    def test_motion_heading_preserves_observed_goal_axis_position_and_arrays(self):
        command = self._pose_optimizer()["seat_heading_command_state"]
        rotation = Rotation.from_euler("xyz", [5, -7, 22], degrees=True)
        state = {
            "target_quaternion": rotation.as_quat(),
            "target_axis": rotation.apply([0, 0, 1]),
            "target_seat_position": np.array([0.2, -0.1, 0.9]),
            "object_quaternion": np.array([0, 0, 0, 1]),
        }
        before = {k: v.copy() for k, v in state.items()}
        self.assertIs(command(state, 0), state)
        for heading in (-20, -10, 5, 20):
            result = command(state, heading)
            self.assertIsNot(result, state)
            np.testing.assert_allclose(
                Rotation.from_quat(result["target_quaternion"]).apply([0, 0, 1]), state["target_axis"]
            )
            for key in state:
                np.testing.assert_array_equal(state[key], before[key])
                if key != "target_quaternion":
                    self.assertIs(result[key], state[key])

    def test_motion_heading_cannot_exceed_bound_or_invent_an_observed_rotation(self):
        command = self._pose_optimizer()["seat_heading_command_state"]
        for heading in (21, -21, float("nan"), float("inf")):
            with self.subTest(heading=heading), self.assertRaises(ValueError):
                command({}, heading)
        for quaternion in ([0, 0, 0, 0], [0, 0, 0, 2], [0, 0, np.nan, 1], [0, 0, 1]):
            with self.subTest(quaternion=quaternion), self.assertRaises(ValueError):
                command({"target_quaternion": quaternion}, 5)

    def test_chooser_prefers_aligned_grip_with_feasible_heading_and_still_requires_hover(self):
        namespace = self._pose_optimizer()
        solver = SimpleNamespace(
            compute_inverse_kinematics=lambda frame, position, orientation, **kw: (
                np.zeros(7),
                bool(position[2] < 0.2 or orientation[3] > 0.08),
            )
        )
        subset = SimpleNamespace(get_joint_positions=lambda: np.zeros(7))
        active = SimpleNamespace(
            cfg=SimpleNamespace(
                task=SimpleNamespace(pipette_preflight_joint_margin=0.1, pipette_grasp_height_offset=0)
            ),
            robot=SimpleNamespace(
                _articulation_view=SimpleNamespace(get_dof_limits=lambda: np.tile([-2, 2], (1, 7, 1)))
            ),
            rmp_controller=SimpleNamespace(
                rmp_flow=SimpleNamespace(get_kinematics_solver=lambda: solver, end_effector_frame_name="eef"),
                articulation_rmp=SimpleNamespace(get_active_joints_subset=lambda: subset),
            ),
            _command_seat_position=lambda state: np.array([0, 0, 1]),
            _release_clearance=lambda: 0,
        )
        state = {
            "object_path": "/tool",
            "object_axis": [1, 0, 0],
            "object_grasp_position": np.zeros(3),
            "object_reference_position": np.zeros(3),
            "object_quaternion": [0, 0, 0, 1],
            "target_quaternion": [0, 0, 0, 1],
        }
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: 0.01)
        records, selected = [], Mock()
        pick = namespace["choose_seat_grasp"](
            active,
            state,
            [1, 0, 0, 0],
            [0, 0, 0],
            [0, 10],
            0,
            35,
            records.append,
            pregrasp_checker=checker,
            pregrasp_height=0.14,
            seat_headings=(0, 5, 10),
            on_heading_selected=selected,
        )
        np.testing.assert_allclose(pick, [1, 0, 0, 0])
        self.assertEqual([r["seat_heading_deg"] for r in records], [0, 5, 10])
        self.assertTrue(all(r["roll_deg"] == 0 for r in records))
        self.assertTrue(records[-1]["hover_feasible"])
        selected.assert_called_once_with(10)
        self.assertEqual(state["target_quaternion"], [0, 0, 0, 1])
        checker.minimum_clearance = lambda q: -0.01
        selected.reset_mock()
        with self.assertRaisesRegex(ValueError, "No candidate grasp"):
            namespace["choose_seat_grasp"](
                active,
                state,
                [1, 0, 0, 0],
                [0, 0, 0],
                [0],
                0,
                35,
                records.append,
                pregrasp_checker=checker,
                pregrasp_height=0.14,
                seat_headings=(0, 5, 10),
                on_heading_selected=selected,
            )
        selected.assert_not_called()

    def test_heading_routes_motion_only_and_does_not_replace_controller_observation(self):
        source = Path(__file__).resolve().parents[2] / "experiments/pipette/run_pipette_fixture.py"
        node = next(
            n
            for n in ast.walk(ast.parse(source.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "forward_with_command_heading"
        )
        command = self._pose_optimizer()["seat_heading_command_state"]
        original = {"target_quaternion": np.array([0, 0, 0, 1])}
        active = SimpleNamespace(state=original, _scene_state=original)
        forward = Mock(return_value=("action", "record"))
        namespace = {"approach_forward": forward, "placement_command_state": lambda state: command(state, 10)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(namespace["forward_with_command_heading"](active, original), ("action", "record"))
        self.assertIs(active.state, original)
        self.assertIs(active._scene_state, original)
        self.assertIsNot(forward.call_args.args[1], original)
        np.testing.assert_array_equal(original["target_quaternion"], [0, 0, 0, 1])

    def test_continuation_screens_every_segment_without_actuation(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (q[:3], np.eye(3)))
        calls = []

        def solve(solver, frame, position, orientation, seed, bounds, **kwargs):
            calls.append((position.copy(), seed.copy()))
            return np.r_[position, seed[3:]], True

        targets = np.array([[0.006, 0, 0], [0.006, 0.006, 0], [0, 0.006, 0]])
        report = screen(
            solver, solve, "eef", np.zeros(7), targets, [1, 0, 0, 0], np.tile([-1, 1], (7, 1)), joint_margin=0.1
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(report["kinematics_only"])
        self.assertEqual(report["samples"], 9)
        np.testing.assert_allclose(report["final_joints"][:3], targets[-1])
        for position, seed in calls:
            self.assertLessEqual(np.linalg.norm(position - seed[:3]), 0.002 + 1e-12)

    def test_continuation_accepts_clear_open_hand_geometry(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (q[:3], np.eye(3)))
        checker = SimpleNamespace(margin=0.002, minimum_clearance=Mock(return_value=0.01))
        report = screen(
            solver,
            lambda s, f, p, o, q, bounds, **kw: (np.r_[p, q[3:]], True),
            "eef",
            np.zeros(7),
            [[0.006, 0, 0]],
            [1, 0, 0, 0],
            np.tile([-1, 1], (7, 1)),
            joint_margin=0.1,
            geometry_checker=checker,
        )
        self.assertTrue(report["accepted"])
        self.assertFalse(report["kinematics_only"])
        self.assertEqual(report["minimum_robot_gap_m"], 0.01)
        self.assertGreaterEqual(checker.minimum_clearance.call_count, 4)

    def test_continuation_rejects_collision_inside_ik_joint_segment(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        # A redundant wrist joint sweeps 0.1 rad between clear IK endpoints.
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (q[:3], np.eye(3)))
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: -0.001 if 0.04 < q[3] < 0.06 else 0.01)
        report = screen(
            solver,
            lambda s, f, p, o, q, bounds, **kw: (np.r_[p, 0.1, q[4:]], True),
            "eef",
            np.zeros(7),
            [[0.002, 0, 0]],
            [1, 0, 0, 0],
            np.tile([-1, 1], (7, 1)),
            joint_margin=0.1,
            geometry_checker=checker,
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["kinematics_only"])
        self.assertIn("placement geometry", report["reason"])
        self.assertEqual(report["minimum_robot_gap_m"], -0.001)
        self.assertEqual(report["geometry"]["start_gap"], 0.01)
        self.assertEqual(report["geometry"]["end_gap"], 0.01)

    def test_continuation_rejects_blocked_start_before_ik(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        solver, solve = Mock(), Mock()
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: -0.001)
        report = screen(
            solver,
            solve,
            "eef",
            np.zeros(7),
            [[0.002, 0, 0]],
            [1, 0, 0, 0],
            np.tile([-1, 1], (7, 1)),
            joint_margin=0.1,
            geometry_checker=checker,
        )
        self.assertFalse(report["accepted"])
        solver.compute_forward_kinematics.assert_not_called()
        solve.assert_not_called()

    def test_continuation_rejects_inaccurate_fk_branch_jump_limits_and_missing_ik(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        targets = np.array([[0.004, 0, 0]])
        for q, ok in (
            (np.zeros(7), True),
            (np.full(7, 0.4), True),
            (np.full(7, 0.95), True),
            (np.zeros(7), False),
            (np.full(7, np.nan), True),
        ):
            with self.subTest(q=q, ok=ok):
                solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (q[:3], np.eye(3)))
                report = screen(
                    solver,
                    lambda *a, q=q, ok=ok, **kw: (q, ok),
                    "eef",
                    np.zeros(7),
                    targets,
                    [1, 0, 0, 0],
                    np.tile([-1, 1], (7, 1)),
                    joint_margin=0.1,
                )
                self.assertFalse(report["accepted"])

    def test_continuation_budget_is_checked_before_any_ik_call(self):
        screen = self._pose_optimizer()["screen_pose_continuation"]
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (q[:3], np.eye(3)))
        solve = Mock()
        report = screen(
            solver, solve, "eef", np.zeros(7), [[1, 0, 0]], [1, 0, 0, 0], np.tile([-2, 2], (7, 1)), joint_margin=0.1
        )
        self.assertFalse(report["accepted"])
        self.assertIn("256", report["reason"])
        solve.assert_not_called()

    def test_pose_optimizer_discards_baseline_and_short_endpoint_with_failed_continuation(self):
        namespace = self._pose_optimizer()
        endpoints = [np.full(7, 0.1), np.full(7, 0.3)]
        namespace["rank_pose_ik"] = lambda *a, **kw: [
            {"joints": q, "joint_cost": float(q[0]), "endpoint_gap": 0.01} for q in endpoints
        ]
        solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q: (np.zeros(3), np.eye(3)))
        checker = SimpleNamespace(solver=solver, margin=0.002, minimum_clearance=lambda q: 0.01)
        endpoint = [None]
        planner = SimpleNamespace(
            set_cspace_target=lambda q: endpoint.__setitem__(0, q.copy()),
            set_random_seed=Mock(),
            compute_path=Mock(side_effect=lambda q, unused: np.array([q, endpoint[0]])),
            set_end_effector_target=Mock(),
        )
        baseline = np.array([np.zeros(7), np.full(7, 0.2)])
        validator = Mock(side_effect=lambda q: {"accepted": bool(q[0] == 0.3)})
        chosen, report = namespace["improve_pose_path"](
            planner,
            checker,
            "eef",
            np.zeros(3),
            [1, 0, 0, 0],
            baseline[0],
            baseline,
            np.tile([-1, 1], (7, 1)),
            smooth_joint_path,
            35,
            joint_margin=0.1,
            endpoint_validator=validator,
        )
        self.assertFalse(report["baseline_continuation"]["accepted"])
        self.assertFalse(report["candidates"][0]["continuation"]["accepted"])
        np.testing.assert_allclose(chosen[-1], endpoints[1])
        self.assertGreater(report["selected_commands"], report["baseline_commands"])
        planner.compute_path.assert_called_once()

    def test_pose_optimizer_returns_no_path_if_every_continuation_fails(self):
        namespace = self._pose_optimizer()
        namespace["rank_pose_ik"] = lambda *a, **kw: []
        checker = SimpleNamespace(solver=None, margin=0.002, minimum_clearance=lambda q: 0.01)
        planner = SimpleNamespace(set_end_effector_target=Mock())
        baseline = np.array([np.zeros(7), np.full(7, 0.2)])
        chosen, report = namespace["improve_pose_path"](
            planner,
            checker,
            "eef",
            np.zeros(3),
            [1, 0, 0, 0],
            baseline[0],
            baseline,
            np.tile([-1, 1], (7, 1)),
            smooth_joint_path,
            35,
            joint_margin=0.1,
            endpoint_validator=lambda q: {"accepted": False},
        )
        self.assertIsNone(chosen)
        self.assertIsNone(report["selected_commands"])

    def test_hold_compensation_is_slow_capped_and_does_not_move_original_target(self):
        compensate = self._pose_optimizer()["screened_hold_compensation"]
        target = np.zeros(7)
        measured = np.full(7, -0.02)
        bias = np.zeros(7)
        checker = SimpleNamespace(margin=0.002, minimum_clearance=Mock(return_value=0.01))
        limits = np.tile([-1.0, 1.0], (7, 1))
        command, updated, verdict = compensate(target, measured, bias, checker, limits, joint_margin=0.1)
        np.testing.assert_allclose(updated, np.full(7, 0.0008))
        np.testing.assert_array_equal(target, np.zeros(7))
        np.testing.assert_array_equal(bias, np.zeros(7))
        self.assertTrue(verdict["accepted"])
        self.assertGreater(checker.minimum_clearance.call_count, 2)  # chord, not endpoint only
        for _ in range(100):
            command, updated, verdict = compensate(target, measured, updated, checker, limits, joint_margin=0.1)
        np.testing.assert_allclose(command, np.full(7, 0.03))

    def test_hold_compensation_rejects_blocked_chord_with_clear_endpoints(self):
        compensate = self._pose_optimizer()["screened_hold_compensation"]
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: -0.001 if -0.015 < q[0] < -0.004 else 0.01)
        with self.assertRaisesRegex(ValueError, "dense clearance"):
            compensate(np.zeros(7), np.full(7, -0.02), np.zeros(7), checker, np.tile([-1, 1], (7, 1)), joint_margin=0.1)

    def test_hold_compensation_rejects_large_error_bad_vectors_and_margin(self):
        compensate = self._pose_optimizer()["screened_hold_compensation"]
        checker = SimpleNamespace(margin=0.002, minimum_clearance=Mock(return_value=0.01))
        bounds = np.tile([-1, 1], (7, 1))
        for target, measured, bias, margin in (
            (np.zeros(7), np.full(7, 0.061), np.zeros(7), 0.1),
            (np.zeros(7), np.zeros(7), np.full(7, 0.031), 0.1),
            (np.full(7, np.nan), np.zeros(7), np.zeros(7), 0.1),
            (np.zeros(6), np.zeros(7), np.zeros(7), 0.1),
            (np.zeros(7), np.zeros(7), np.zeros(7), -1),
            (np.zeros(7), np.zeros(7), np.zeros(7), np.nan),
            (np.zeros(7), np.zeros(7), np.zeros(7), 1),
            (np.full(7, 0.91), np.full(7, 0.88), np.zeros(7), 0.1),
        ):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                compensate(target, measured, bias, checker, bounds, joint_margin=margin)
        checker.minimum_clearance.assert_not_called()

    def test_hold_compensation_retains_physical_margin_when_bias_saturates(self):
        compensate = self._pose_optimizer()["screened_hold_compensation"]
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: 0.01)
        target = np.full(7, 0.89)
        command, bias, verdict = compensate(
            target, np.full(7, 0.88), np.full(7, 0.0298), checker, np.tile([-1, 1], (7, 1)), joint_margin=0.1
        )
        np.testing.assert_allclose(command, np.full(7, 0.9))
        np.testing.assert_allclose(bias, np.full(7, 0.01))
        self.assertTrue(verdict["accepted"])

    def test_pose_optimizer_rejects_blocked_closest_ik_and_uses_shorter_safe_plan(self):
        namespace = self._pose_optimizer()
        start, position = np.zeros(7), np.zeros(3)
        choices = np.zeros((3, 7))
        choices[:, :2] = [[0.2, 0], [0.3, 0.3], [0.6, 0.6]]
        count = [0]

        def solve(*args, **kwargs):
            q = choices[count[0] % 3]
            count[0] += 1
            return q, True

        solver = SimpleNamespace(
            compute_inverse_kinematics=solve,
            compute_forward_kinematics=lambda frame, q: (position + (0.01 if q[0] == 0.6 else 0), np.eye(3)),
        )
        checker = SimpleNamespace(
            margin=0.002, solver=solver, minimum_clearance=lambda q: np.linalg.norm(q[:2] - [0.1, 0]) - 0.02
        )
        target = [None]
        planner = SimpleNamespace(
            set_cspace_target=lambda q: target.__setitem__(0, q.copy()),
            set_random_seed=Mock(),
            compute_path=Mock(side_effect=lambda q, unused: np.asarray([q, target[0]])),
            set_end_effector_target=Mock(),
        )
        baseline = np.zeros((2, 7))
        baseline[1, :2] = [0.8, 0.7]
        chosen, details = namespace["improve_pose_path"](
            planner,
            checker,
            "eef",
            position,
            [1, 0, 0, 0],
            start,
            baseline,
            np.tile([-1, 1], (7, 1)),
            smooth_joint_path,
            35,
            joint_margin=0.1,
        )
        self.assertEqual(count[0], 92)
        self.assertEqual(details["feasible_unique_endpoints"], 2)  # bad actual FK rejected
        self.assertFalse(details["candidates"][0]["accepted"])
        self.assertTrue(details["improved"])
        np.testing.assert_array_equal(chosen[-1], choices[1])
        np.testing.assert_array_equal(baseline[1, :2], [0.8, 0.7])
        self.assertLess(details["selected_commands"], details["baseline_commands"])
        self.assertEqual(planner.compute_path.call_count, 2)  # never bypass native planner
        planner.set_end_effector_target.assert_called_once_with(position, [1, 0, 0, 0])

    def test_pose_optimizer_retains_baseline_when_native_planner_fails(self):
        namespace = self._pose_optimizer()
        joints = np.full(7, 0.2)
        solver = SimpleNamespace(
            compute_inverse_kinematics=lambda *a, **kw: (joints, True),
            compute_forward_kinematics=lambda frame, q: (np.zeros(3), np.eye(3)),
        )
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: 0.01, solver=solver)
        planner = SimpleNamespace(
            set_cspace_target=Mock(),
            set_random_seed=Mock(),
            compute_path=Mock(return_value=None),
            set_end_effector_target=Mock(),
        )
        baseline = np.array([[0.0] * 7, [0.8] * 7])
        chosen, details = namespace["improve_pose_path"](
            planner,
            checker,
            "eef",
            np.zeros(3),
            [1, 0, 0, 0],
            baseline[0],
            baseline,
            np.tile([-1, 1], (7, 1)),
            smooth_joint_path,
            35,
            joint_margin=0.1,
        )
        self.assertIs(chosen, baseline)
        self.assertFalse(details["improved"])
        self.assertEqual(planner.compute_path.call_count, 1)  # duplicate IK solutions deduplicated
        self.assertEqual(planner.set_end_effector_target.call_count, 1)

    def test_ik_ranking_rejects_nonfinite_out_of_bounds_and_inaccurate_results(self):
        rank = self._pose_optimizer()["rank_pose_ik"]
        for joints, actual in (
            (np.full(7, np.nan), np.zeros(3)),
            (np.full(7, 0.95), np.zeros(3)),
            (np.zeros(7), np.full(3, np.nan)),
            (np.zeros(7), np.full(3, 0.01)),
        ):
            solver = SimpleNamespace(
                compute_inverse_kinematics=Mock(return_value=(joints, True)),
                compute_forward_kinematics=Mock(return_value=(actual, np.eye(3))),
            )
            checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: 0.01)
            self.assertEqual(
                rank(
                    solver,
                    checker,
                    "eef",
                    np.zeros(3),
                    [1, 0, 0, 0],
                    np.zeros(7),
                    np.tile([-1, 1], (7, 1)),
                    35,
                    joint_margin=0.1,
                ),
                [],
            )

    def test_approach_shortcut_checks_actual_commands_and_preserves_audit_path(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        names = {"screen_joint_path", "shortcut_joint_path", "prepare_approach_path"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"np": np, "pairwise": pairwise}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        path = np.zeros((4, 7))
        path[:, :2] = [[0, 0], [0, 0.1], [0.1, 0.1], [0.1, 0]]
        original = path.copy()
        limits = np.tile([-1.0, 1.0], (7, 1))
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: np.linalg.norm(q[:2] - [0.05, 0]) - 0.02)
        prepare = namespace["prepare_approach_path"]
        short, details = prepare(path, checker, limits, lambda q: q.copy(), joint_margin=0.1)
        self.assertEqual(len(short), 3)
        np.testing.assert_array_equal(path, original)
        np.testing.assert_array_equal(details["original_path"], original)
        self.assertEqual(details["original_waypoints"], 4)
        self.assertEqual(details["shortened_waypoints"], 3)
        self.assertGreaterEqual(details["minimum_command_gap"], 0.002)
        # A safe geometric shortcut does not excuse a bad interpolation that
        # cuts across the blocked chord or exceeds the actual joint bounds.
        for interpolate in (lambda q: q[[0, -1]], lambda q: np.vstack([q, np.full(7, 0.95)])):
            with self.assertRaisesRegex(ValueError, "interpolation violates"):
                prepare(path, checker, limits, interpolate, joint_margin=0.1)

    def test_seat_chooser_rejects_obstructed_hover_and_tries_opposite_jaws(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        names = {
            "forecast_seated_pose",
            "grasp_orientation_candidates",
            "choose_seat_grasp",
            "seat_heading_command_state",
        }
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"np": np, "Rotation": Rotation, "product": product}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)

        def solve(frame, position, orientation, **kwargs):
            result = np.zeros(7)
            result[0] = 0.5 if abs(orientation[0]) < 0.5 else 0
            return result, True

        solver = SimpleNamespace(compute_inverse_kinematics=solve)
        subset = SimpleNamespace(get_joint_positions=lambda: np.zeros(7))
        active = SimpleNamespace(
            cfg=SimpleNamespace(
                task=SimpleNamespace(pipette_preflight_joint_margin=0.1, pipette_grasp_height_offset=0)
            ),
            robot=SimpleNamespace(
                _articulation_view=SimpleNamespace(get_dof_limits=lambda: np.tile([-2, 2], (1, 7, 1)))
            ),
            rmp_controller=SimpleNamespace(
                rmp_flow=SimpleNamespace(get_kinematics_solver=lambda: solver, end_effector_frame_name="eef"),
                articulation_rmp=SimpleNamespace(get_active_joints_subset=lambda: subset),
            ),
            _command_seat_position=lambda state: np.array([0, 0, 1]),
            _release_clearance=lambda: 0,
        )
        state = {
            "object_path": "/tool",
            "object_axis": [1, 0, 0],
            "object_grasp_position": np.zeros(3),
            "object_reference_position": np.zeros(3),
            "object_quaternion": [0, 0, 0, 1],
            "target_quaternion": [0, 0, 0, 1],
        }
        records = []
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: 0.01 if q[0] > 0.1 else -0.01)
        pick = namespace["choose_seat_grasp"](
            active,
            state,
            [1, 0, 0, 0],
            [0, 0, 0],
            [0],
            0,
            35,
            records.append,
            symmetric=True,
            pregrasp_checker=checker,
            pregrasp_height=0.14,
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["seat_feasible"])
        self.assertFalse(records[0]["hover_feasible"])
        self.assertFalse(records[0]["feasible"])
        self.assertTrue(records[1]["jaws_swapped"])
        self.assertTrue(records[1]["feasible"])
        np.testing.assert_allclose(pick, [0, 0, 0, 1], atol=1e-12)

    def test_jaw_swap_preserves_approach_axis_and_reverses_closing_normal(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grasp_orientation_candidates")
        namespace = {"np": np, "Rotation": Rotation}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        base = Rotation.from_euler("xyz", [15, 70, -20], degrees=True).as_quat()[[3, 0, 1, 2]]
        axis = np.array([2.0, 1.0, 0.0])
        candidates = list(namespace["grasp_orientation_candidates"](base, axis, [0, -10], -10, symmetric=True))
        self.assertEqual([(r, s) for r, s, _ in candidates], [(0, False), (0, True), (-10, False), (-10, True)])
        for i in (0, 2):
            first = Rotation.from_quat(candidates[i][2][[1, 2, 3, 0]])
            second = Rotation.from_quat(candidates[i + 1][2][[1, 2, 3, 0]])
            np.testing.assert_allclose(first.apply([0, 0, 1]), second.apply([0, 0, 1]), atol=1e-12)
            np.testing.assert_allclose(first.apply([0, 1, 0]), -second.apply([0, 1, 0]), atol=1e-12)
        np.testing.assert_array_equal(axis, [2, 1, 0])
        self.assertEqual(len(list(namespace["grasp_orientation_candidates"](base, axis, [0], 0))), 1)
        with self.assertRaises(ValueError):
            list(namespace["grasp_orientation_candidates"](base, [0, 0, 0], [0], 0))

    def test_shortcut_preserves_endpoints_and_cannot_cross_an_obstacle(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("screen_joint_path", "shortcut_joint_path")
        ]
        namespace = {"np": np, "pairwise": pairwise}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        path = np.zeros((4, 7))
        path[:, :2] = [[0, 0], [0, 0.1], [0.1, 0.1], [0.1, 0]]
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: np.linalg.norm(q[:2] - [0.05, 0]) - 0.02)
        limits = np.tile([-1.0, 1.0], (7, 1))
        short = namespace["shortcut_joint_path"](path, checker, limits, joint_margin=0.1)
        self.assertEqual(len(short), 3)
        np.testing.assert_array_equal(short[[0, -1]], path[[0, -1]])
        self.assertTrue(namespace["screen_joint_path"](short, checker, limits, joint_margin=0.1)["accepted"])
        with self.assertRaisesRegex(ValueError, "invalid input"):
            namespace["shortcut_joint_path"](path[[0, -1]], checker, limits, joint_margin=0.1)
        checker.minimum_clearance = lambda q: 0.01
        short = namespace["shortcut_joint_path"](path, checker, limits, joint_margin=0.1)
        self.assertEqual(len(short), 2)

    @unittest.skipIf(Usd is None, "USD libraries unavailable")
    def test_physical_mesh_reader_excludes_visual_and_disabled_shapes_without_writes(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "world_collision_meshes")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/hand")
        for name in ("physical", "visual", "disabled"):
            mesh = UsdGeom.Mesh.Define(stage, "/hand/" + name)
            mesh.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            if name != "visual":
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(name != "disabled")
        usd = SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf)
        before = stage.GetRootLayer().ExportToString()
        groups = namespace["world_collision_meshes"](usd, stage, "/hand")
        self.assertEqual(list(groups), ["/hand/physical"])
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        cube = UsdGeom.Cube.Define(stage, "/hand/unsupported")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            namespace["world_collision_meshes"](usd, stage, "/hand")

    def test_pregrasp_descent_uses_actual_final_arrival_and_rotated_tcp(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PregraspDescent")
        namespace = {"np": np, "Rotation": Rotation}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        commands = []
        motion = SimpleNamespace(forward=lambda **kwargs: commands.append(kwargs))
        rotation = Rotation.from_euler("y", 90, degrees=True)
        target = np.array([0.2, 0.1, 0.75])
        descent = namespace["PregraspDescent"](motion, target, rotation.as_quat()[[3, 0, 1, 2]], [0, 0, 0.0034])
        for _ in range(10):
            _, record = descent.forward(target + np.array([0, 0, 0.08]), rotation.as_matrix())
            self.assertFalse(record["reached"])
        for _ in range(9):
            _, record = descent.forward(target, rotation.as_matrix())
            self.assertFalse(record["reached"])
        _, record = descent.forward(target, rotation.as_matrix())
        self.assertTrue(record["reached"])
        np.testing.assert_allclose(commands[-1]["target_end_effector_position"], [0.2034, 0.1, 0.75])
        _, record = descent.forward(target, Rotation.identity().as_matrix())
        self.assertFalse(record["reached"])
        self.assertEqual(descent.stable, 0)
        descent.steps = 600
        with self.assertRaisesRegex(ValueError, "600-step"):
            descent.forward(target, rotation.as_matrix())

    def test_seat_forecast_rotates_grasp_and_converts_tcp_without_state_writes(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forecast_seated_pose")
        namespace = {"np": np, "Rotation": Rotation}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        active = SimpleNamespace(
            cfg=SimpleNamespace(task=SimpleNamespace(pipette_grasp_height_offset=-0.003)),
            _command_seat_position=lambda state: np.array([0.2, -0.2, 0.92]),
            _release_clearance=lambda: 0.014,
        )
        source_rotation = Rotation.from_euler("y", 90, degrees=True)
        state = {
            "object_quaternion": source_rotation.as_quat(),
            "target_quaternion": Rotation.identity().as_quat(),
            "object_reference_position": np.array([0.1, 0.1, 0.75]),
            "object_grasp_position": np.array([0.15, 0.1, 0.753]),
        }
        before = {k: v.copy() for k, v in state.items()}
        position, orientation = namespace["forecast_seated_pose"](
            active, state, source_rotation.as_quat()[[3, 0, 1, 2]], [0, 0, 0.0034]
        )
        np.testing.assert_allclose(position, [0.2, -0.2, 0.9806], atol=1e-12)
        np.testing.assert_allclose(orientation, [1, 0, 0, 0], atol=1e-12)
        for key in state:
            np.testing.assert_array_equal(state[key], before[key])

    def test_proposal_screen_checks_segment_interiors_and_joint_limits(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "screen_joint_path")
        namespace = {"np": np, "pairwise": pairwise}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        check = namespace["screen_joint_path"]
        path = np.zeros((2, 7))
        path[1, 0] = 0.1
        limits = np.tile([-1.0, 1.0], (7, 1))
        checker = SimpleNamespace(margin=0.002, minimum_clearance=lambda q: abs(q[0] - 0.05) - 0.01)
        self.assertFalse(check(path, checker, limits, joint_margin=0.1)["accepted"])
        checker.minimum_clearance = lambda q: 0.003
        self.assertTrue(check(path, checker, limits, joint_margin=0.1)["accepted"])
        path[1, 0] = 0.95
        self.assertFalse(check(path, checker, limits, joint_margin=0.1)["accepted"])
        with self.assertRaises(ValueError):
            check([[float("nan")] * 7], checker, limits, joint_margin=0.1)

    @unittest.skipIf(Usd is None, "USD libraries unavailable")
    def test_actual_mesh_reader_traverses_instances_without_changing_stage(self):
        source = Path(__file__).resolve().parents[2] / "controllers/pipette_path_planning.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "world_mesh_vertices")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/prototype")
        mesh = UsdGeom.Mesh.Define(stage, "/prototype/mesh")
        mesh.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        instance = UsdGeom.Xform.Define(stage, "/instance")
        instance.GetPrim().GetReferences().AddInternalReference("/prototype")
        instance.GetPrim().SetInstanceable(True)
        instance.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        before = stage.GetRootLayer().ExportToString()
        points = namespace["world_mesh_vertices"](SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, Gf=Gf), stage, "/instance")
        np.testing.assert_allclose(points, [[1, 2, 3], [2, 2, 3], [1, 3, 3], [1, 2, 4]])
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def setUp(self):
        cls = load_execution_class()
        self.motion = cls.__new__(cls)
        self.records = []
        self.motion.emit = self.records.append
        self.motion.index = self.motion.steps = self.motion.stable = 0
        self.motion.done = False
        self.motion.hold_steps = 0
        self.motion.tracking_bias = np.zeros(7)
        self.motion.joint_bounds = np.tile([-3.0, 3.0], (7, 1))
        self.motion.joint_margin = 0.1
        self.motion.checker = SimpleNamespace(minimum_clearance=lambda q: 0.01)
        self.motion.commands = np.zeros((2, 7))
        self.motion.other_objects = {"/other": np.zeros(3)}
        self.motion.orientation = np.array([1, 0, 0, 0])
        self.motion.tcp_target = np.zeros(3)
        self.joints = np.zeros(7)
        self.rotation = np.eye(3)
        self.drift = 0
        subset = SimpleNamespace(get_joint_positions=lambda: self.joints)
        self.active = SimpleNamespace(
            _trace_grasp=lambda state: self.drift,
            cfg=SimpleNamespace(task=SimpleNamespace(max_grasp_rotation_deg=8)),
            rmp_controller=SimpleNamespace(
                articulation_rmp=SimpleNamespace(get_active_joints_subset=lambda: subset),
                rmp_flow=SimpleNamespace(get_end_effector_pose=lambda joints: (np.zeros(3), self.rotation)),
            ),
        )
        self.state = {
            "gripper_position": np.zeros(3),
            "pipettes": [
                {"object_path": "/held", "object_grasp_position": np.ones(3)},
                {"object_path": "/other", "object_grasp_position": np.zeros(3)},
            ],
        }

    def test_actual_arrival_must_be_stable_and_records_only_arm_commands(self):
        for _ in range(9):
            action = self.motion.forward(self.active, self.state)
            self.assertFalse(self.motion.done)
            np.testing.assert_array_equal(action.joint_indices, np.arange(7))
            self.assertEqual(action.joint_positions.shape, (7,))
        self.motion.forward(self.active, self.state)
        self.assertTrue(self.motion.done)

    def test_orientation_and_physical_tcp_errors_prevent_arrival(self):
        for angular, linear in ((3, 0), (0, 0.003)):
            self.motion.stable = 0
            self.rotation = Rotation.from_euler("z", angular, degrees=True).as_matrix()
            self.state["gripper_position"] = np.array([linear, 0, 0])
            for _ in range(12):
                self.motion.forward(self.active, self.state)
            self.assertFalse(self.motion.done)

    def test_joint_tracking_error_prevents_path_advance(self):
        self.joints[:] = 0.04
        self.motion.forward(self.active, self.state)
        self.assertEqual(self.motion.index, 0)
        self.assertFalse(self.motion.done)

    def test_slip_and_other_object_motion_stop_carry(self):
        self.drift = 8.01
        with self.assertRaisesRegex(ValueError, "slip"):
            self.motion.forward(self.active, self.state)
        self.drift = 0
        self.state["pipettes"][1]["object_grasp_position"][0] = 0.002
        with self.assertRaisesRegex(ValueError, "Another pipette"):
            self.motion.forward(self.active, self.state)

    def test_timeout_does_not_claim_arrival(self):
        self.motion.steps = 3000
        with self.assertRaisesRegex(ValueError, "budget"):
            self.motion.forward(self.active, self.state)
        self.assertFalse(self.motion.done)

    def test_static_compensation_is_bounded_and_never_overrides_clearance(self):
        self.joints[:] = 0.04
        for _ in range(30):
            action = self.motion.forward(self.active, self.state)
        self.assertLessEqual(np.max(np.abs(self.motion.tracking_bias)), 0.03)
        self.assertLessEqual(np.max(np.abs(action.joint_positions)), 0.03)
        self.motion.checker.minimum_clearance = lambda q: -0.001
        with self.assertRaisesRegex(ValueError, "lost clearance"):
            self.motion.forward(self.active, self.state)
