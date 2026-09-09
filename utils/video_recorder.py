"""Episode video display, recording, and H264 conversion."""

import os
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger


class VideoRecorder:
    def __init__(self, run_dir: str, cameras: Any, *, save: bool, show: bool, fps: float = 60.0) -> None:
        self.run_dir = Path(run_dir)
        self.cameras = cameras
        self.save = save
        self.show = show
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._output_path: str | None = None
        self._threads: list[threading.Thread] = []

    def write(self, state: dict[str, Any], episode_num: int) -> None:
        if not (self.save or self.show):
            return

        images = [cv2.cvtColor(data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for data in state["camera_display"].values()]
        if not images:
            return

        combined = np.hstack(images)
        offset = 0
        for index, image in enumerate(images):
            label = f"Camera {index + 1} ({self.cameras[index].image_type})"
            cv2.putText(combined, label, (offset + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            offset += image.shape[1]

        if self.show:
            cv2.imshow("Camera Views", combined)
            cv2.waitKey(1)
        if self.save:
            self._write_frame(combined, episode_num)

    def finish_episode(self, is_success: bool) -> None:
        if self._writer is None or self._output_path is None:
            return
        self._writer.release()
        thread = threading.Thread(
            target=self._convert_to_h264,
            args=(self._output_path, is_success),
            daemon=False,
        )
        thread.start()
        self._threads.append(thread)
        self._writer = None
        self._output_path = None

    def close(self) -> None:
        for thread in self._threads:
            thread.join()
        cv2.destroyAllWindows()

    def _write_frame(self, image: np.ndarray, episode_num: int) -> None:
        if self._writer is None:
            output_dir = self.run_dir / "video"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._output_path = str(output_dir / f"episode_{episode_num}.mp4")
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                self._output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self._writer.isOpened():
                logger.error(f"Failed to open VideoWriter for {self._output_path}")
                self._writer = None
                self._output_path = None
                return
        self._writer.write(image)

    @staticmethod
    def _convert_to_h264(src_path: str, is_success: bool) -> None:
        status = "success" if is_success else "failure"
        tmp_path = f"{src_path}.{status}.h264.tmp.mp4"
        dest_path = src_path.replace(".mp4", f"_{status}.mp4")
        logger.info(f"Converting video to H264: {src_path}")
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                src_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "fast",
                tmp_path,
            ],
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode == 0:
            os.replace(tmp_path, dest_path)
            os.remove(src_path)
            logger.success(f"Video saved: {dest_path}")
            return

        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        logger.error(f"H264 conversion failed for: {src_path}\n{stderr}")
        with suppress(FileNotFoundError):
            os.remove(tmp_path)
