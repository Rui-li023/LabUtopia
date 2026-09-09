"""Offscreen camera evidence for the Franka physical-contact bench."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom


class BenchRecording:
    def __init__(self, world, output: Path, video: bool):
        self.world, self.output, self.frames = world, output, 0
        self.width, self.height, self.fps = 960, 900, 24
        self.footer_height = 88
        self.telemetry = []
        camera = UsdGeom.Camera.Define(world.stage, "/World/BenchCamera")
        camera.CreateFocalLengthAttr(40)
        camera.CreateHorizontalApertureAttr(20)
        camera.CreateVerticalApertureAttr(18.75)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(1.3, -2.4, 1.4), Gf.Vec3d(-0.10, -0.20, 0.35), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        camera.AddTransformOp().Set(matrix)
        self.product = rep.create.render_product(str(camera.GetPath()), (self.width, self.height))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach(self.product)
        self.font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 24)
        self.process = None
        self.error_log = None
        if video:
            self.error_log = (output / "video_encoder.log").open("w")
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{self.width}x{self.height + self.footer_height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output / "franka_demo.mp4"),
                ],
                stdin=subprocess.PIPE,
                stderr=self.error_log,
            )
        for _ in range(12):
            world.render()

    def frame(self, phase: str, state: dict):
        if self.process is None:
            return
        self.world.render()
        frame = Image.fromarray(self.rgb.get_data()).convert("RGB")
        canvas = Image.new("RGB", (self.width, self.height + self.footer_height), (14, 20, 27))
        canvas.paste(frame, (0, 0))
        label = phase
        for key, text in {
            "initialize": "待机 · 缩小后的仪器与 Franka",
            "approach_": "接近 ",
            "press_": "实体按压 ",
            "retract_": "松开并撤回 ",
            "motor_run": "RUN 启停",
            "motor_power": "电机电源",
            "bath_up": "温度 +",
            "bath_down": "温度 −",  # noqa: RUF001 - typographic minus in the video caption
            "bath_set": "SET 确认",
            "bath_mode": "加热开关",
            "grasp_dial": "双指夹持旋钮",
            "release_dial": "松开旋钮",
            "turn_dial_": "旋钮转至 ",
            "dial": "调速旋钮",
        }.items():
            label = label.replace(key, text)
        draw = ImageDraw.Draw(canvas)
        draw.text((20, self.height + 6), "Franka 接触测试  |  " + label, font=self.font, fill=(205, 239, 237))
        draw.text(
            (20, self.height + 44),
            f"设定 {state['rpm_set']:.1f} rpm    实测 {state['rpm']:.1f} rpm    温度设定 {state['temperature_edit_c']:.0f} °C",
            font=self.font,
            fill=(212, 218, 227),
        )
        self.process.stdin.write(canvas.tobytes())
        self.telemetry.append({"frame": self.frames, "phase": phase, "state": state})
        self.frames += 1

    def snapshot(self, name="franka_scene"):
        for _ in range(12):
            self.world.render()
        Image.fromarray(self.rgb.get_data()).save(self.output / (name + ".png"))

    def close(self):
        if self.process:
            self.process.stdin.close()
            code = self.process.wait(timeout=60)
            self.error_log.close()
            if code:
                raise RuntimeError("Franka video encoder failed")
            (self.output / "franka_demo.json").write_text(json.dumps(self.telemetry, indent=2) + "\n")
        return {
            "frames": self.frames,
            "fps": self.fps,
            "duration_s": self.frames / self.fps,
            "file": str(self.output / "franka_demo.mp4") if self.frames else None,
        }
