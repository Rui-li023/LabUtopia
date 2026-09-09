"""Interact with the articulated evaporator or validate it in Isaac Sim 5.1.

Examples:
    python experiments/assets/simulate_rotary_evaporator.py
    python experiments/assets/simulate_rotary_evaporator.py --headless --validate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true")
mode = parser.add_mutually_exclusive_group()
mode.add_argument(
    "--validate-device", action="store_true", help="Test on-model buttons, displays and switch headlessly"
)
mode.add_argument("--validate-full", action="store_true", help="Test the complete instrument with scripted buttons")
mode.add_argument("--validate", action="store_true")
mode.add_argument("--demo", action="store_true", help="Run the old automatic motion sequence")
mode.add_argument("--validate-interaction", action="store_true", help="Validate interactive physical controls")
mode.add_argument("--ui-test", action="store_true", help="Open the actual panel, exercise its callbacks, then exit")
parser.add_argument(
    "--device-config", type=Path, default=ROOT / "assets/chemistry_lab/rotary_evaporator/interaction_config.json"
)
parser.add_argument("--renderer", choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting")
parser.add_argument("--position", nargs=3, type=float, default=[0, 0, 0], metavar=("X", "Y", "Z"))
parser.add_argument(
    "--record-cycle",
    type=Path,
    help="Save an operating-cycle MP4 alongside --validate-full/--validate-device and --snapshot",
)
parser.add_argument(
    "--record-focus", choices=["cycle", "dial"], default="cycle", help="Record a full cycle or continuous dial close-up"
)
parser.add_argument("--render-samples", type=int, help="Path tracing samples per frame, 1..256")
parser.add_argument("--render-size", nargs=2, type=int, default=[900, 1100], metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--snapshot", type=Path, help="Optional final Isaac Sim camera PNG")
parser.add_argument("--usd", type=Path, default=ROOT / "assets/chemistry_lab/rotary_evaporator/rotary_evaporator.usdc")
parser.add_argument("--report", type=Path, default=ROOT / "outputs/rotary_evaporator/v2/physics_validation.json")
args = parser.parse_args()
args.validate_device = args.validate_device or args.validate_full
if any(size < 128 or size > 4096 or size % 2 for size in args.render_size):
    parser.error("--render-size must use even dimensions within 128..4096")
if args.render_samples is not None and not 1 <= args.render_samples <= 256:
    parser.error("--render-samples must be within 1..256")
if args.record_cycle and not (args.validate_device and args.snapshot):
    parser.error("--record-cycle needs --validate-device or --validate-full, plus --snapshot")
if args.ui_test and args.headless:
    parser.error("--ui-test needs a graphical window; use --validate-interaction for headless tests")
app = SimulationApp(
    {
        "headless": args.headless,
        "width": 640 if args.headless else 1000,
        "height": 640 if args.headless else 1000,
        "renderer": args.renderer,
        "window_width": 1440,
        "window_height": 1000,
        "fast_shutdown": True,
        "samples_per_pixel_per_frame": args.render_samples or (64 if args.snapshot else 16),
        "max_bounces": 12,
        "max_specular_transmission_bounces": 12,
    }
)

# Isaac Sim modules must be imported after SimulationApp initializes the extensions.
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicSphere, GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402


def setup_scene(world: World) -> None:
    """Create a small, portable studio around the reusable asset."""
    world.scene.add(GroundPlane("/World/Ground", z_position=-0.005, size=200, color=np.array([0.22, 0.24, 0.25])))
    stage = world.stage
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(600.0)
    for name, loc, power, size in [
        ("Key", (-0.7, -1, 1.6), 1600, (0.8, 1.2)),
        ("Rim", (0.6, 0.5, 1.3), 1600, (0.5, 1.0)),
    ]:
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateIntensityAttr(power)
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*(np.array(loc) + args.position)),
                Gf.Vec3d(*(np.array([0, 0, 0.5]) + args.position)),
                Gf.Vec3d(0, 0, 1),
            )
            .GetInverse()
        )
        UsdGeom.Xformable(light).AddTransformOp().Set(matrix)
    set_camera_view(
        eye=np.array([0.38, -1.22, 0.81]) + args.position, target=np.array([-0.12, 0, 0.52]) + args.position
    )


def run_interactive(world: World, device: SingleArticulation) -> None:
    # These modules depend on UI and PhysX extensions initialized by SimulationApp.
    from experiments.assets.rotary_evaporator_controls import EvaporatorControls, EvaporatorPanel

    controls = EvaporatorControls(device)
    electronics = inspector = viewport_controls = None
    if not args.validate_interaction and not args.ui_test:
        from experiments.assets.rotary_evaporator_device import EvaporatorDevice
        from experiments.assets.rotary_evaporator_device_ui import DeviceInspector, DeviceViewport

        electronics = EvaporatorDevice(controls, world.stage, config_path=args.device_config)
        if not args.headless:
            inspector = DeviceInspector(electronics)
            viewport_controls = DeviceViewport(electronics)
    panel = None if args.headless else EvaporatorPanel(controls, electronics)
    try:
        if args.validate_interaction or args.ui_test:
            from experiments.assets.validate_rotary_evaporator_interaction import validate_interaction

            result = validate_interaction(controls, world, panel)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
            logger.info("Interactive control validation: {}", args.report)
            return
        logger.info("Interactive controls ready: J/L tilt, U/O lift, G lock, P spin, H home, X stop")
        needs_reset = False
        steps = 0
        while app.is_running():
            if not world.is_playing():
                needs_reset = needs_reset or world.is_stopped()
                if viewport_controls:
                    viewport_controls.cancel()
                    inspector.window.frame.enabled = False
                panel.active = False
                panel.window.frame.enabled = False
                panel.status.text = "PAUSED - press Play in the timeline"
                world.render()
                continue
            if needs_reset:
                world.reset()
                controls.controller.set_gains(kps=controls.original_kp.copy(), kds=controls.original_kd.copy())
                controls.free_hinge = False
                controls.capture_current_pose()
                if electronics:
                    electronics.reset()
                needs_reset = False
            panel.active = True
            panel.window.frame.enabled = True
            if electronics:
                viewport_controls.active = True
                inspector.window.frame.enabled = True
                electronics.update(1 / 120)
            else:
                controls.update(1 / 120)
            world.step(render=steps % 2 == 0)
            if steps % 12 == 0:
                panel.refresh()
                if inspector:
                    inspector.refresh()
            steps += 1
    finally:
        if viewport_controls:
            viewport_controls.close()
        if inspector:
            inspector.close()
        if panel:
            panel.close()


def main() -> None:
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
    add_reference_to_stage(str(args.usd.resolve()), "/World/Evaporator")
    setup_scene(world)
    device = world.scene.add(
        SingleArticulation("/World/Evaporator", name="evaporator", position=np.array(args.position))
    )
    # A joint attached to the world uses a world-space anchor. Relocating the
    # reference alone does not update it, so match the base's authored pose.
    fixed = UsdPhysics.FixedJoint(world.stage.GetPrimAtPath("/World/Evaporator/Joints/fixed_base"))
    base_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(world.stage.GetPrimAtPath("/World/Evaporator/base"))
    fixed.CreateLocalPos0Attr(Gf.Vec3f(base_matrix.ExtractTranslation()))
    fixed.CreateLocalRot0Attr(Gf.Quatf(base_matrix.ExtractRotationQuat()))
    probe = None
    if args.validate:
        probe = world.scene.add(
            DynamicSphere(
                "/World/CavityProbe",
                name="cavity_probe",
                position=np.array([0.295, 0.065, 0.53]) + np.array(args.position),
                radius=0.008,
                mass=0.015,
                color=np.array([0.9, 0.1, 0.05]),
            )
        )
    if args.validate_device:
        world.scene.add(
            DynamicSphere(
                "/World/ButtonProbe",
                name="button_probe",
                position=np.array([0.7, -0.7, 0.8]) + args.position,
                radius=0.004,
                mass=0.02,
                color=np.array([0.8, 0.2, 0.1]),
            )
        )
    if args.snapshot:
        # Camera annotators require an initialized SimulationApp.
        import omni.replicator.core as rep
        from PIL import Image

        camera = UsdGeom.Camera.Define(world.stage, "/World/ProductCamera")
        camera.CreateFocalLengthAttr(55.0)
        camera.CreateHorizontalApertureAttr(20.0)
        camera.CreateVerticalApertureAttr(24.444)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        matrix = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*(np.array([0.8, -3, 1.1]) + args.position)),
                Gf.Vec3d(*(np.array([0, 0, 0.51]) + args.position)),
                Gf.Vec3d(0, 0, 1),
            )
            .GetInverse()
        )
        UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
        product = rep.create.render_product("/World/ProductCamera", tuple(args.render_size))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
    world.reset()
    if args.validate_device:
        UsdPhysics.RigidBodyAPI(world.stage.GetPrimAtPath("/World/ButtonProbe")).CreateKinematicEnabledAttr(True)
    names = list(device.dof_names)
    logger.info("Articulation DOFs: {}", names)
    assert {"lift_joint", "tilt_joint", "spin_joint"}.issubset(names), names
    if args.validate_device:
        from experiments.assets.rotary_evaporator_controls import EvaporatorControls
        from experiments.assets.rotary_evaporator_device import EvaporatorDevice
        from experiments.assets.validate_rotary_evaporator_calibration import validate_dial_calibration
        from experiments.assets.validate_rotary_evaporator_device import validate_device

        controls = EvaporatorControls(device)
        electronics = EvaporatorDevice(controls, world.stage, config_path=args.device_config)
        result = validate_device(electronics, world)
        result["dial_calibration"] = validate_dial_calibration(electronics, world)
        result["physics_steps"] += result["dial_calibration"]["physics_steps"]
        if args.validate_full:
            from experiments.assets.validate_rotary_evaporator_full import validate_full_device

            def capture_state(label):
                for _ in range(12):
                    world.render()
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb.get_data()).save(args.snapshot.with_name(label + ".png"))

            result["full_instrument"] = validate_full_device(
                electronics, world, capture_state if args.snapshot else None
            )
            result["physics_steps"] += result["full_instrument"]["physics_steps"]

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        if args.snapshot:
            electronics.set_switch(True)
            for _ in range(120):
                electronics.update(1 / 120)
                world.step(render=False)
            electronics.press("motor_run")
            for _ in range(240):
                electronics.update(1 / 120)
                world.step(render=False)
            for _ in range(20):
                world.render()
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb.get_data()).save(args.snapshot)
            # Additional offline detail renders show the actual running device,
            # physical button travel, and a powered-down screen.
            for label, eye, target in [
                ("upper_controls", (0.18, -0.85, 0.77), (-0.035, -0.03, 0.63)),
                ("bath_controls", (0.60, -1.15, 0.32), (0.225, -0.02, 0.12)),
                ("button_pressed", (0.08, -0.48, 0.78), (-0.05, -0.035, 0.711)),
                ("power_off", (0.18, -0.85, 0.77), (-0.035, -0.03, 0.63)),
            ]:
                if label == "button_pressed":
                    electronics.press("motor_run", held=True)
                elif label == "power_off":
                    electronics.release("motor_run")
                    electronics.set_switch(False)
                for _ in range(120):
                    electronics.update(1 / 120)
                    world.step(render=False)
                matrix = (
                    Gf.Matrix4d()
                    .SetLookAt(
                        Gf.Vec3d(*(np.array(eye) + args.position)),
                        Gf.Vec3d(*(np.array(target) + args.position)),
                        Gf.Vec3d(0, 0, 1),
                    )
                    .GetInverse()
                )
                UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(matrix)
                for _ in range(20):
                    world.render()
                Image.fromarray(rgb.get_data()).save(args.snapshot.with_name("isaac_" + label + ".png"))
        if args.record_cycle:
            from experiments.assets.record_rotary_evaporator_cycle import record_cycle

            result["recording"] = record_cycle(
                electronics, world, camera, rgb, args.record_cycle, args.position, focus=args.record_focus
            )
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        logger.info("Device validation: {}", args.report)
        world.stop()
        return
    if args.validate_interaction or args.ui_test or (not args.headless and not args.demo and not args.validate):
        run_interactive(world, device)
        world.stop()
        return
    indices = {name: names.index(name) for name in names}
    lift_index, tilt_index, spin_index = (indices[n] for n in ["lift_joint", "tilt_joint", "spin_joint"])
    initial_base = device.get_world_pose()[0].copy()
    assert np.allclose(initial_base, args.position, atol=0.001), (initial_base, args.position)
    checkpoints = []
    failures = []
    # The first stage is an idle-with-gravity test. Then move, rotate, and return.
    phases = [
        ("rest", 0, 0, 0),
        ("raise", 0.115, 0, 0),
        ("tilt_out", 0.115, -18, 0),
        ("rotate", 0.115, -18, 2 * np.pi),
        ("tilt_back", 0.115, 0, 0),
        ("lower", 0.002, 0, 0),
        ("tilt_in", 0.002, 4, 0),
        ("return", 0, 0, 0),
    ]
    if args.validate:
        phases += [
            ("lift_upper_stop", 0.16, 0, 0),
            ("tilt_lower_stop", 0.10, -30, 0),
            ("tilt_upper_stop", 0.10, 12, 0),
            ("final_rest", 0, 0, 0),
        ]
    keep_running = True
    while keep_running and app.is_running():
        for label, height, angle_degrees, speed in phases:
            command = np.array([height, np.deg2rad(angle_degrees)], dtype=np.float32)
            device.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=np.array([lift_index, tilt_index]))
            )
            device.apply_action(
                ArticulationAction(
                    joint_velocities=np.array([speed], dtype=np.float32), joint_indices=np.array([spin_index])
                )
            )
            for _ in range(360):
                if not app.is_running():
                    return
                world.step(render=not args.headless)
                q = device.get_joint_positions()
                qd = device.get_joint_velocities()
                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                    raise AssertionError(f"Nonfinite joint state during {label}: {q}, {qd}")
                if np.max(np.abs(qd)) > 100:
                    raise AssertionError(f"Unstable velocity during {label}: {qd}")
            q = device.get_joint_positions()
            qd = device.get_joint_velocities()
            lift_error = abs(float(q[lift_index]) - np.clip(height, 0, 0.12))
            tilt_error = abs(float(q[tilt_index]) - np.deg2rad(np.clip(angle_degrees, -20, 5)))
            speed_error = abs(float(qd[spin_index]) - speed)
            record = {
                "phase": label,
                "lift_m": float(q[lift_index]),
                "tilt_degrees": float(np.rad2deg(q[tilt_index])),
                "spin_rad_s": float(qd[spin_index]),
                "lift_error_m": lift_error,
                "tilt_error_degrees": float(np.rad2deg(tilt_error)),
                "spin_velocity_error_rad_s": speed_error,
            }
            checkpoints.append(record)
            logger.info("{}", record)
            if lift_error > 0.003 or tilt_error > np.deg2rad(1.5) or speed_error > 0.15:
                failures.append(record)
            if label == "rest" and probe is not None:
                pos, _ = probe.get_world_pose()
                logger.info("Cavity probe settled at {}", pos)
                pos = pos - np.array(args.position)
                # The bath lip is at 0.396 m. A filled hull would stop the sphere there.
                if not (0.235 < pos[2] < 0.250 and abs(pos[0] - 0.295) < 0.02 and abs(pos[1] - 0.065) < 0.02):
                    failures.append({"cavity_probe_failed": pos.tolist()})
        keep_running = not (args.headless or args.validate)
    if args.snapshot:
        for _ in range(12):
            world.step(render=True)
        pixels = rgb.get_data()
        if pixels.size == 0:
            raise AssertionError("Isaac Sim camera returned no image")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.snapshot)
        logger.info("Isaac Sim rendered snapshot: {}", args.snapshot)
    base_drift = float(np.linalg.norm(device.get_world_pose()[0] - initial_base))
    if base_drift > 0.001:
        failures.append({"base_drift_m": base_drift})
    report = {
        "passed": not failures,
        "isaac_sim": "5.1",
        "physics_dt_s": 1 / 120,
        "physics_steps": len(checkpoints) * 360,
        "dof_names": names,
        "fixed_base_drift_m": base_drift,
        "asset_position_m": args.position,
        "checkpoints": checkpoints,
        "failures": failures,
        "cavity_probe_position_m": probe.get_world_pose()[0].tolist() if probe else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Report: {}", args.report)
    if failures:
        raise AssertionError(f"Validation failed: {failures}")
    world.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception("Evaporator validation/demo failed")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"passed": False, "error": repr(error)}, indent=2) + "\n")
        # Kit fast shutdown otherwise exits with code zero, hiding failed tests.
        # The report has been closed and the error logged before exiting.
        os._exit(1)
    finally:
        app.close()
