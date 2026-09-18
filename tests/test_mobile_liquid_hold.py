"""Liquid-settling frames record drive targets in the mobile replay layout."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_hold_method():
    source = Path(__file__).resolve().parents[1] / "controllers/mobile_pour_controller.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_MobilePourController"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_record_liquid_hold")
    namespace = {"np": np, "GRIPPER_MAX_OPEN": 0.05}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


HOLD = load_hold_method()


class MobileLiquidHoldTests(unittest.TestCase):
    def record(self, targets, indices):
        result = []
        controller = SimpleNamespace(
            robot=SimpleNamespace(get_applied_action=lambda: SimpleNamespace(joint_positions=targets)),
            all_subset=SimpleNamespace(joint_indices=np.asarray(indices)),
            _record_step=lambda state, action, phase: result.append((action, phase)),
        )
        HOLD(controller, {}, 5)
        return result[0]

    def test_uses_drive_targets_and_joint_subset_order(self):
        targets = np.arange(12, dtype=float) / 100
        indices = list(reversed(range(12)))
        action, phase = self.record(targets, indices)
        np.testing.assert_array_equal(action[:10], targets[indices[:10]])
        self.assertAlmostEqual(action[10], 0.8)
        self.assertEqual(phase, 5)
        self.assertEqual(action.dtype, np.dtype("float64"))

    def test_invalid_hold_cannot_enter_dataset(self):
        with self.assertRaisesRegex(ValueError, "finite targets"):
            self.record([np.nan] * 12, range(12))


if __name__ == "__main__":
    unittest.main()
