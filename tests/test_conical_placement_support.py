"""Measure actual collision support, without treating visual target sheets as tables."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

source = ast.parse((Path(__file__).resolve().parents[1] / "tasks/open_transport_pour_task.py").read_text())
scene = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_OpenTransportPourScene")
measure = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == "_measure_placement_supports")


class ConicalPlacementSupportTests(unittest.TestCase):
    def measure(self, hit):
        query = Mock()
        query.raycast_closest.return_value = hit
        scope = {"np": np, "logger": Mock(), "omni": SimpleNamespace(
            physx=SimpleNamespace(get_physx_scene_query_interface=lambda: query))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[measure], type_ignores=[])), "actual_support_measurement", "exec"), scope)
        owner = SimpleNamespace(object_utils=SimpleNamespace(
            get_geometry_center=lambda object_path: np.array([0.1, -0.5, 0.775])))
        return scope["_measure_placement_supports"](owner), query

    def test_support_uses_collision_height_not_visual_marker(self):
        supports, query = self.measure({"hit": True, "position": [0.1, -0.5, 0.7198], "collision": "/World/Cube"})
        self.assertEqual(set(supports), {"/World/target_plat", "/World/target_plat2"})
        self.assertEqual(supports["/World/target_plat2"][2], 0.7198)
        query.raycast_closest.assert_called_with((0.1, -0.5, 0.790), (0.0, 0.0, -1.0), 0.3)

    def test_missing_collision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "No physical support"):
            self.measure({"hit": False})

    def test_nonfinite_collision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid placement support"):
            self.measure({"hit": True, "position": [0.1, -0.5, float("nan")]})


if __name__ == "__main__":
    unittest.main()
