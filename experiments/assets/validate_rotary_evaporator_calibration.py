"""Check engraved dial geometry against solved joints and visible LCD segments."""

from __future__ import annotations

import copy
import math

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics

from experiments.assets.rotary_evaporator_displays import DIGITS


def validate_dial_calibration(device, world) -> dict:
    """Exercise the existing articulation without creating a renderer or window."""
    steps, checks = 0, []
    original = copy.deepcopy(device.config)
    stage, root = world.stage, device.root

    def tick(count: int) -> None:
        nonlocal steps
        for _ in range(count):
            device.update(1 / 120)
            world.step(render=False)
            steps += 1
            assert np.isfinite(device.articulation.get_joint_positions()).all()
            assert np.isfinite(device.articulation.get_joint_velocities()).all()

    def record(name: str, **values) -> None:
        checks.append({"check": name, **values, "state": device.state()})

    def mesh_center(path: str, body: str) -> np.ndarray:
        prim = stage.GetPrimAtPath(path)
        assert prim and prim.IsA(UsdGeom.Mesh), path
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        assert len(points) and np.isfinite(points).all(), path
        cache = UsdGeom.XformCache()
        relative = (
            cache.GetLocalToWorldTransform(prim)
            * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body)).GetInverse()
        )
        return np.asarray(relative.Transform(Gf.Vec3d(*points.mean(axis=0))))

    def read_display() -> int:
        chars = ""
        for index in range(3):
            visible = set()
            for segment in "abcdefg":
                path = f"{root}/lift/Display_rpm/Digits/d{index}_{segment}"
                prim = stage.GetPrimAtPath(path)
                assert prim, path
                if UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
                    visible.add(segment)
            matches = [char for char in "0123456789" if set(DIGITS[char]) == visible]
            assert len(matches) == 1, (index, sorted(visible))
            chars += matches[0]
        return int(chars)

    # Remove each body's current pose: the pointer may already have turned when
    # this check starts. These centroids come from exported meshes, not the
    # runtime conversion helper or its requested angular destination.
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root + "/Joints/speed_dial_joint"))
    assert joint and joint.GetAxisAttr().Get() == "Y"
    assert joint.GetBody0Rel().GetTargets() == [stage.GetPrimAtPath(root + "/lift").GetPath()]
    assert joint.GetBody1Rel().GetTargets() == [stage.GetPrimAtPath(root + "/speed_dial").GetPath()]
    for rotation in (joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()):
        assert np.linalg.norm(np.asarray(rotation.GetImaginary())) < 1e-7, rotation
    assert joint.GetLowerLimitAttr().Get() == 0 and joint.GetUpperLimitAttr().Get() == 300
    pointer = mesh_center(root + "/speed_dial/Visuals/Controller___dial_index", root + "/speed_dial")
    pointer -= np.asarray(joint.GetLocalPos1Attr().Get())
    assert np.linalg.norm(pointer[[0, 2]]) > 0.005, pointer
    zero_angle = math.atan2(pointer[0], pointer[2])
    tick_angles, tick_values, errors = [], [], []
    for value in range(0, 121, 5):
        tick_center = mesh_center(root + f"/lift/Visuals/Controller___speed_tick_{value}", root + "/lift")
        radial = tick_center - np.asarray(joint.GetLocalPos0Attr().Get())
        angle = math.degrees(math.atan2(radial[0], radial[2]) - zero_angle)
        angle = 0.0 if abs(angle) < 0.001 else angle % 360
        # The printed values independently specify 0..120 over this 300-degree
        # sweep. Never call the implementation's angle/rpm mapping in this test.
        expected = value * 2.5
        error = abs(angle - expected)
        assert error < 0.01, (value, angle, expected, radial.tolist(), pointer.tolist())
        tick_angles.append(angle)
        tick_values.append(value)
        errors.append(error)
    assert np.all(np.diff(tick_angles) > 0)
    record(
        "all_25_exported_ticks_match_pointer_geometry",
        tick_count=len(tick_angles),
        max_angular_error_deg=max(errors),
        measured_tick_angles_deg=tick_angles,
    )

    def settle(requested: float, expected: int) -> None:
        dial = device.config["dial"]
        current = math.degrees(float(device.articulation.get_joint_positions()[device.indices["speed_dial"]]))
        destination = float(np.interp(expected, tick_values, tick_angles))
        speed = dial.get("max_speed_deg_s", 60)
        duration = abs(destination - current) / speed + 2 * speed / dial.get("max_accel_deg_s2", 180) + 0.8
        device.set_dial_rpm(requested)
        tick(math.ceil(duration * 120))
        device.refresh_displays()
        angle = math.degrees(float(device.articulation.get_joint_positions()[device.indices["speed_dial"]]))
        engraved_value = float(np.interp(angle, tick_angles, tick_values))
        displayed = read_display()
        assert abs(angle - destination) < 0.3, (requested, expected, angle, destination)
        assert abs(engraved_value - expected) < 0.15, (engraved_value, expected)
        assert abs(device.rpm_set - engraved_value) < 0.15, (device.rpm_set, engraved_value)
        assert displayed == expected, (displayed, expected, device.state())
        record(
            "solved_pointer_and_display_match_engraving",
            requested_rpm=requested,
            engraved_rpm=expected,
            observed_pointer_rpm=engraved_value,
            measured_angle_deg=angle,
            displayed_rpm=displayed,
        )

    def click_run() -> None:
        previous = len(device.events)
        device.press("motor_run")
        tick(240)
        assert len(device.events) == previous + 1, device.events

    try:
        assert original["displays"]["rpm"]["mode"] == "live"
        assert original["displays"]["rpm"]["source"] == "rpm_set", original["displays"]["rpm"]
        assert original["dial"]["min_rpm"] == 0 and original["dial"]["max_rpm"] == 120
        device.reset()
        tick(120)
        for value in (0, 20, 40, 60, 80, 100, 120):
            settle(value, value)
            assert not device.rotating and abs(device.controls.read_state()["rpm"]) < 0.5

        settle(60, 60)
        click_run()
        assert device.rotating and abs(device.controls.read_state()["rpm"] - 60) < 0.5, device.state()
        assert read_display() == 60
        record("running_display_keeps_calibrated_60_rpm_setting")
        click_run()
        assert not device.rotating and abs(device.controls.read_state()["rpm"]) < 0.5, device.state()
        assert abs(device.state()["dial_angle_deg"] - tick_angles[12]) < 0.3
        assert read_display() == 60
        record("stopped_motor_keeps_60_rpm_pointer_and_display")

        actual_display = copy.deepcopy(original)
        actual_display["displays"]["rpm"]["source"] = "rpm_actual"
        device.reload_config(actual_display)
        tick(30)
        assert read_display() == 0 and abs(device.rpm_set - 60) < 0.15
        click_run()
        assert read_display() == 60 and device.rotating
        click_run()
        assert read_display() == 0 and not device.rotating
        record("optional_actual_speed_display_tracks_motor_separately")

        limited = copy.deepcopy(original)
        limited["dial"].update(min_rpm=30, max_rpm=90)
        device.reload_config(limited)
        settle(0, 30)
        settle(120, 90)
        record("hot_reload_limits_clamp_without_stretching_engraved_scale")
    finally:
        device.reload_config(original)
        device.reset()
        dial = original["dial"]
        duration = (
            300 / dial.get("max_speed_deg_s", 60)
            + 2 * dial.get("max_speed_deg_s", 60) / dial.get("max_accel_deg_s2", 180)
            + 0.8
        )
        tick(math.ceil(duration * 120))

    return {"passed": True, "isaac_sim": "5.1", "physics_steps": steps, "checks": checks}
