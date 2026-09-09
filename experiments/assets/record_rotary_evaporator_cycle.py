"""Record the operating cycle from Isaac's camera, with separate measured telemetry."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom


def record_cycle(device, world, camera, rgb, destination: Path, offset, focus: str = "cycle") -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    device.reset()
    device.controls.home()
    device.set_dial_rpm(40)
    for _ in range(420):
        device.update(1 / 120)
        world.step(render=False)
    if focus == "dial":
        device.press("motor_run")
        for _ in range(180):
            device.update(1 / 120)
            world.step(render=False)
    shots = {
        "dial": ((-0.049, -0.55, 0.680), (-0.049, -0.020, 0.680), 55),
        "full": ((0.8, -3, 1.1), (0, 0, 0.51), 55),
        "upper": ((0.10, -0.72, 0.755), (-0.034, -0.015, 0.696), 50),
        "bath": ((0.52, -0.80, 0.24), (0.22, -0.045, 0.100), 45),
        "lever": ((-0.12, -0.60, 0.34), (-0.005, -0.035, 0.105), 55),
    }

    def shot(name):
        eye, aim, focal = shots[name]
        # Upper panel moves with the lift. Frame its current physical position.
        lift = device.controls.read_state()["lift_mm"] / 1000 if name == "upper" else 0
        delta = np.array([0, 0, lift]) + offset
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*(np.array(eye) + delta)),
                Gf.Vec3d(*(np.array(aim) + delta)),
                Gf.Vec3d(0, 0, 1),
            )
            .GetInverse()
        )
        camera.GetFocalLengthAttr().Set(focal)
        # A frontal orthographic view removes perspective parallax between the
        # recessed fixed scale and the index on the knob's front surface.
        camera.GetProjectionAttr().Set("orthographic" if name == "dial" else "perspective")
        camera.GetHorizontalApertureAttr().Set(1.65 if name == "dial" else 20.0)
        camera.GetVerticalApertureAttr().Set(2.017 if name == "dial" else 24.444)
        UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(matrix)
        for _ in range(8):
            world.render()

    shot("dial" if focus == "dial" else "full")
    pixels = rgb.get_data()
    height, width = pixels.shape[:2]
    footer_height = 64 if focus == "cycle" else 0
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
    fps = 30 if focus == "dial" else 24
    held_button = None

    def press(name):
        nonlocal held_button
        held_button = name
        device.press(name, held=True)

    stages = [
        ("整机待机", 2, "full", lambda: None),
        ("电机电源: 关闭", 1, "upper", lambda: press("motor_power")),
        ("电机电源: 开启", 1, "upper", lambda: press("motor_power")),
        ("启动旋转: 40 rpm", 2, "upper", lambda: press("motor_run")),
        ("旋钮调速: 40 → 80 rpm", 3, "dial", lambda: device.set_dial_rpm(80)),
        ("旋钮调速: 80 → 40 rpm", 3, "dial", lambda: device.set_dial_rpm(40)),
        ("浴锅: 开启加热请求", 1, "bath", lambda: press("bath_mode")),
        ("温度设定: 增加", 1, "bath", lambda: press("bath_up")),
        ("温度设定: 减少", 1, "bath", lambda: press("bath_down")),
        ("温度设定: 确认", 1, "bath", lambda: press("bath_set")),
        ("升降: 抬升至 100 mm", 3, "full", lambda: device.controls.set_lift_mm(100)),
        ("黑色手柄: 俯仰调节", 2, "lever", lambda: device.controls.set_lever_degrees(-12)),
        ("手柄联动: 机头回正", 2, "full", lambda: device.controls.set_lever_degrees(0)),
        ("停止旋转: 保留 40 rpm 设定", 2, "upper", lambda: press("motor_run")),
        ("机构复位: 下降并回正", 3, "full", device.controls.home),
        ("浴锅: 关闭加热请求", 1, "bath", lambda: press("bath_mode")),
        ("侧面总开关: 关闭电源", 2, "bath", lambda: device.set_switch(False)),
        ("整机关闭", 2, "full", lambda: None),
    ]
    if focus == "dial":
        stages = [
            ("Scale and display: 40 rpm", 1, "dial", lambda: None),
            ("Turn clockwise: 40 -> 80 rpm", 3, "dial", lambda: device.set_dial_rpm(80)),
            ("Turn back: 80 -> 40 rpm", 3, "dial", lambda: device.set_dial_rpm(40)),
            ("RUN stop: retain 40 rpm setting", 1, "dial", lambda: press("motor_run")),
        ]
    frame_dir = destination.parent / (destination.stem + "_frames")
    frame_dir.mkdir(exist_ok=True)
    metadata = []
    with destination.with_suffix(".ffmpeg.log").open("w") as error_log:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height + footer_height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                *(["-vf", "crop=iw:ih-104:0:0"] if focus == "dial" else []),
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
                str(destination),
            ],
            stdin=subprocess.PIPE,
            stderr=error_log,
        )
        try:
            for phase_index, (label, seconds, view, action) in enumerate(stages):
                shot(view)
                held_button = None
                logger.info("Rendering phase {}/{}: {}", phase_index + 1, len(stages), label)
                action()
                for frame_index in range(seconds * fps):
                    if held_button and frame_index == fps // 2:
                        device.release(held_button)
                        held_button = None
                    for _ in range(120 // fps):
                        device.update(1 / 120)
                        world.step(render=False)
                    device.refresh_displays()
                    world.render()
                    frame = Image.fromarray(rgb.get_data()).convert("RGB")
                    state = device.state()
                    state["lever_angle_deg"] = float(
                        np.rad2deg(device.articulation.get_joint_positions()[device.controls.lever_index])
                    )
                    if frame_index == seconds * fps - 1:
                        frame.save(frame_dir / f"{phase_index + 1:02d}_{view}.png")
                    if footer_height:
                        # Put a static chapter caption below the complete camera
                        # image. CPU telemetry can lead GPU camera readback and
                        # must not be presented as numbers from this image.
                        captioned = Image.new("RGB", (width, height + footer_height), (14, 20, 27))
                        captioned.paste(frame, (0, 0))
                        draw = ImageDraw.Draw(captioned)
                        draw.text((24, height + 12), label, font=font, fill=(205, 239, 237))
                        frame = captioned
                    process.stdin.write(frame.tobytes())
                    metadata.append({"time_s": len(metadata) / fps, "phase": label, "state": state})
        finally:
            process.stdin.close()
            returncode = process.wait(timeout=60)
        if returncode:
            raise RuntimeError(f"Video encoder failed; see {destination.with_suffix('.ffmpeg.log')}")
    assert not device.rotating and abs(device.controls.read_state()["rpm"]) < 1
    if focus == "cycle":
        assert not device.mains_on
    assert abs(device.controls.read_state()["tilt_deg"]) < 1 and device.controls.read_state()["lift_mm"] < 3
    destination.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return {
        "file": str(destination),
        "frames": len(metadata),
        "fps": fps,
        "duration_s": len(metadata) / fps,
        "keyframes": str(frame_dir),
        "focus": focus,
        "resolution": [width, height + footer_height if footer_height else height - 104],
        "physics_steps": 420 + (180 if focus == "dial" else 0) + len(metadata) * (120 // fps),
        "renderer": "Isaac Sim camera; PhysX motion; on-model readout"
        if focus == "dial"
        else "Isaac Sim camera; PhysX motion; Chinese chapter captions below the camera image",
        "metadata_note": "Telemetry JSON contains producer simulation states. GPU image readback may lag by several frames.",
    }
