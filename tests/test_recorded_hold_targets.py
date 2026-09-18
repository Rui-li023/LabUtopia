"""A no-op must record the held drive target, not the measured joint pose."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_record_method():
    source = Path(__file__).resolve().parents[1] / "controllers/atomic_actions/atomic_base_controller.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_record_array")
    namespace = {
        "np": np,
        "ArticulationAction": SimpleNamespace,
        "GRIPPER_MAX_OPEN": 0.05,
        "GRIPPER_CLOSED": 0,
        "AtomicBaseController": SimpleNamespace(record_commanded_gripper=True),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


RECORD = load_record_method()


class RecordedHoldTargetsTests(unittest.TestCase):
    def controller(self, targets):
        return SimpleNamespace(
            _last_record_positions=None,
            _last_gripper_state=1,
            _robot=SimpleNamespace(
                get_applied_action=lambda: SimpleNamespace(joint_positions=targets),
                get_gripper_commanded_opening=lambda: 0.05,
            ),
        )

    def test_startup_noop_records_drive_targets_after_warmup(self):
        targets = np.arange(9, dtype=float) / 10
        for positions in (None, [None] * 9):
            controller = self.controller(targets)
            record = RECORD(controller, SimpleNamespace(joint_positions=positions), targets + 0.003)
            np.testing.assert_array_equal(record[:7], targets[:7])
            self.assertEqual(record[7], 0.0)

    def test_partial_command_preserves_uncommanded_drive_targets(self):
        controller = self.controller(np.ones(9))
        record = RECORD(controller, SimpleNamespace(joint_positions=[None, 0.25, None]), np.zeros(9))
        np.testing.assert_array_equal(record[:7], [1, 0.25, 1, 1, 1, 1, 1])

    def test_existing_recorded_target_takes_precedence_at_handoff(self):
        controller = self.controller(np.ones(9))
        controller._last_record_positions = np.full(8, 0.5)
        record = RECORD(controller, SimpleNamespace(joint_positions=None), np.zeros(9))
        np.testing.assert_array_equal(record[:7], np.full(7, 0.5))

    def test_missing_or_nonfinite_drive_channels_fall_back_to_measurement(self):
        for targets, expected in ((None, np.ones(7)), ([np.nan] + [0.2] * 8, [1] + [0.2] * 6)):
            controller = self.controller(targets)
            record = RECORD(controller, SimpleNamespace(joint_positions=None), np.ones(9))
            np.testing.assert_array_equal(record[:7], expected)


if __name__ == "__main__":
    unittest.main()
