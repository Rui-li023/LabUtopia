import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_pick_controller():
    class ArticulationAction:
        def __init__(self, joint_positions=None):
            self.joint_positions = joint_positions

    class BaseRobot:
        pass

    package_name = "_pick_adaptive_test"
    stub_modules = {
        "isaacsim": _package("isaacsim"),
        "isaacsim.core": _package("isaacsim.core"),
        "isaacsim.core.utils": _package("isaacsim.core.utils"),
        "isaacsim.core.utils.stage": types.SimpleNamespace(get_stage_units=lambda: 1.0),
        "isaacsim.core.utils.types": types.SimpleNamespace(ArticulationAction=ArticulationAction),
        "robots": _package("robots"),
        "robots.base_robot": types.SimpleNamespace(
            BaseRobot=BaseRobot,
            GRIPPER_CLOSED=0,
            GRIPPER_OPEN=1,
        ),
        package_name: _package(package_name),
    }

    root = Path(__file__).resolve().parents[1] / "controllers" / "atomic_actions"
    with patch.dict(sys.modules, stub_modules):
        base_name = f"{package_name}.atomic_base_controller"
        base_spec = importlib.util.spec_from_file_location(base_name, root / "atomic_base_controller.py")
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[base_name] = base_module
        base_spec.loader.exec_module(base_module)

        pick_name = f"{package_name}.pick_controller"
        pick_spec = importlib.util.spec_from_file_location(pick_name, root / "pick_controller.py")
        pick_module = importlib.util.module_from_spec(pick_spec)
        sys.modules[pick_name] = pick_module
        pick_spec.loader.exec_module(pick_module)

    return pick_module.PickController


PickController = _load_pick_controller()


class AdaptivePickPhaseTest(unittest.TestCase):
    def make_controller(self):
        controller = object.__new__(PickController)
        controller.adaptive_phase_completion = True
        controller.arrival_stable_steps = 2
        controller._arrival_steps = 0
        controller._transitioned_this_step = False
        controller._phase_failure_reason = ""
        controller._last_target = None
        controller._last_gripper_position = None
        controller._last_position_error = None
        controller._event = 0
        controller._t = 0.0
        controller._events_dt = [0.5] * 7
        return controller

    def test_requires_consecutive_stable_arrivals(self) -> None:
        controller = self.make_controller()

        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(controller._event, 0)
        self.assertEqual(controller._arrival_steps, 1)

        controller._transitioned_this_step = False
        controller._advance_if_arrived(False, [0.02, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(controller._arrival_steps, 0)

        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._transitioned_this_step = False
        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])

        self.assertEqual(controller._event, 1)
        self.assertEqual(controller._arrival_steps, 0)
        self.assertTrue(controller._transitioned_this_step)

    def test_motion_phase_timeout_reports_context(self) -> None:
        controller = self.make_controller()

        controller._advance_if_arrived(False, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._advance_state()
        controller._transitioned_this_step = False
        controller._advance_if_arrived(False, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._advance_state()

        self.assertTrue(controller.is_done())
        self.assertIn("move_above", controller.get_failure_reason())
        self.assertIn("0.200 m", controller.get_failure_reason())

    def test_reset_clears_timeout_and_locked_target(self) -> None:
        controller = self.make_controller()
        controller._is_done = True
        controller._locked_picking_position = [1.0, 2.0, 3.0]
        controller._cspace_controller = None
        controller._default_n_phases = 7
        controller._randomization_sampled = True

        controller.reset()

        self.assertFalse(controller.is_done())
        self.assertEqual(controller._event, 0)
        self.assertEqual(controller.get_failure_reason(), "")
        self.assertIsNone(controller._locked_picking_position)


if __name__ == "__main__":
    unittest.main()
