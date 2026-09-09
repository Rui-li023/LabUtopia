"""The diagnostic must use the canonical controller planning implementation."""

import ast
import unittest
from pathlib import Path


class PlanningMigrationTests(unittest.TestCase):
    def test_script_reexports_all_public_primitives_without_duplicate_algorithms(self):
        root = Path(__file__).resolve().parents[2]
        old = ast.parse((root / "experiments/pipette/pipette_carry_probe.py").read_text())
        new = ast.parse((root / "controllers/pipette_path_planning.py").read_text())
        definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        self.assertFalse(any(isinstance(node, definitions) for node in old.body))
        canonical = {node.name for node in new.body if isinstance(node, definitions) and not node.name.startswith("_")}
        imports = {
            name.name
            for node in old.body
            if isinstance(node, ast.ImportFrom) and node.module == "controllers.pipette_path_planning"
            for name in node.names
        }
        exports = next(
            ast.literal_eval(node.value)
            for node in old.body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        )
        self.assertEqual(imports, canonical)
        self.assertEqual(set(exports), canonical)


if __name__ == "__main__":
    unittest.main()
