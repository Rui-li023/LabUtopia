import math
import unittest
import xml.etree.ElementTree as ET

from scripts.urdf_to_usd.prepare_urdf import _rpy_matrix, rotate_visual_origins


class RotateVisualOriginsTest(unittest.TestCase):
    def test_adds_rotation_to_visuals_only(self) -> None:
        source = """
        <robot name="test">
          <link name="link">
            <visual><geometry><mesh filename="link.glb"/></geometry></visual>
            <collision><geometry><mesh filename="link.glb"/></geometry></collision>
          </link>
        </robot>
        """

        rewritten, count = rotate_visual_origins(source, (-math.pi / 2.0, 0.0, 0.0))
        root = ET.fromstring(rewritten)

        self.assertEqual(count, 1)
        self.assertEqual(root.find(".//visual/origin").get("xyz"), "0 0 0")
        self.assertEqual(root.find(".//visual/origin").get("rpy"), "-1.5707963267948966 0 0")
        self.assertIsNone(root.find(".//collision/origin"))

    def test_composes_with_existing_origin_and_preserves_translation(self) -> None:
        source = """
        <robot name="test">
          <link name="link">
            <visual>
              <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
              <geometry><mesh filename="link.glb"/></geometry>
            </visual>
          </link>
        </robot>
        """
        correction = (-math.pi / 2.0, 0.0, 0.0)

        rewritten, count = rotate_visual_origins(source, correction)
        origin = ET.fromstring(rewritten).find(".//visual/origin")
        actual_rpy = tuple(float(value) for value in origin.get("rpy").split())
        actual_matrix = _rpy_matrix(actual_rpy)
        expected_matrix = (
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        )

        self.assertEqual(count, 1)
        self.assertEqual(origin.get("xyz"), "1 2 3")
        for actual_row, expected_row in zip(actual_matrix, expected_matrix):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
