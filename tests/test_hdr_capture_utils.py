import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.hdr_capture_utils import _CUBE_FACES, _stitch, _write_exr


class HDRCaptureUtilsTest(unittest.TestCase):
    @staticmethod
    def _faces() -> dict[str, np.ndarray]:
        return {
            name: np.full((8, 8, 3), index, dtype=np.float32) for index, (name, _) in enumerate(_CUBE_FACES, start=1)
        }

    def test_stitch_uses_all_faces_without_black_gaps(self) -> None:
        output = _stitch(self._faces(), out_w=64, out_h=32)

        self.assertEqual(output.shape, (32, 64, 3))
        self.assertFalse(np.any(np.all(output == 0, axis=2)))
        self.assertEqual(set(np.unique(output)), set(range(1, 7)))

    def test_stitch_rejects_missing_face(self) -> None:
        faces = self._faces()
        del faces["nz"]

        with self.assertRaisesRegex(ValueError, "missing cube faces: nz"):
            _stitch(faces, out_w=64, out_h=32)

    def test_write_exr_creates_openexr_file(self) -> None:
        image = np.ones((8, 16, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(_write_exr(image, output_dir))

            self.assertEqual(output_path.read_bytes()[:4], bytes.fromhex("762f3101"))

    def test_write_exr_rejects_black_image(self) -> None:
        image = np.zeros((8, 16, 3), dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as output_dir,
            self.assertRaisesRegex(ValueError, "all-black"),
        ):
            _write_exr(image, output_dir)


if __name__ == "__main__":
    unittest.main()
