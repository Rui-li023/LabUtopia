"""Bounded integration checks against Isaac Sim's actual articulation solver."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from omni.physx import get_physx_scene_query_interface
from pxr import Tf, Usd, UsdGeom

from experiments.assets.rotary_evaporator_device import BUTTONS
from experiments.assets.rotary_evaporator_displays import DIGITS


def validate_device(device, world) -> dict:
    steps, checks = 0, []

    def tick(count=120):
        nonlocal steps
        for _ in range(count):
            device.update(1 / 120)
            world.step(render=False)
            steps += 1
            assert np.isfinite(device.articulation.get_joint_positions()).all()

    def record(name, **extra):
        checks.append({"check": name, **extra, "state": device.state()})

    def click(name):
        initial = len(device.events)
        assert device.activate_path(f"{device.root}/{name}/Visuals/button") == name
        tick(60)
        assert len(device.events) == initial + 1, (name, device.events, device.articulation.get_joint_positions())
        assert abs(device.articulation.get_joint_positions()[device.indices[name]]) < 0.0003

    tick(180)
    assert len(device.indices) == 8
    device.refresh_displays()
    dirty_paths = []

    def changed(notice, _sender):
        dirty_paths.extend(str(path) for path in (*notice.GetChangedInfoOnlyPaths(), *notice.GetResyncedPaths()))

    subscription = Tf.Notice.Register(Usd.Notice.ObjectsChanged, changed, world.stage)
    try:
        for _ in range(20):
            device.refresh_displays()
    finally:
        subscription.Revoke()
    assert not dirty_paths, dirty_paths
    record("unchanged_display_emits_no_usd_edits")
    # Real PhysX picking must reach every cap, with no housing hiding its collision shape.
    for name in (*BUTTONS, "speed_dial", "mains_switch"):
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(world.stage.GetPrimAtPath(device.root + "/" + name))
        center = np.array(transform.ExtractTranslation())
        outward = np.array([1, 0, 0] if name == "mains_switch" else [0, -1, 0])
        hit = get_physx_scene_query_interface().raycast_closest(tuple(center + outward * 0.08), tuple(-outward), 0.15)
        assert hit["hit"] and f"/{name}/Collisions/" in hit["collision"], (name, hit)
    record("all_eight_controls_physx_pickable")
    # Move a kinematic fingertip against the cap: only collision response presses the button.
    probe = world.scene.get_object("button_probe")
    center = np.array(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(world.stage.GetPrimAtPath(device.root + "/motor_run"))
        .ExtractTranslation()
    )
    initial = len(device.events)
    for offset in np.linspace(-0.020, -0.0040, 100):
        probe.set_world_pose(position=center + np.array([0, offset, 0]))
        tick(1)
    tick(30)
    contact_travel = float(device.articulation.get_joint_positions()[device.indices["motor_run"]])
    assert len(device.events) == initial + 1 and device.rotating, (contact_travel, device.events)
    probe.set_world_pose(position=center + np.array([0, -0.3, 0]))
    tick(90)
    click("motor_run")
    tick(120)
    assert not device.rotating
    record("external_fingertip_contact_operates_button", travel_m=contact_travel)
    # Build Kit's gesture overlay headlessly, and verify camera-space picking plus
    # its press/release callbacks. This creates no desktop window.
    from pxr import Gf

    from experiments.assets.rotary_evaporator_device_ui import DeviceInspector, DeviceViewport

    overlay = DeviceViewport(device)
    inspector = DeviceInspector(device)
    for _ in range(3):
        world.render()
    point = (overlay.viewport.view * overlay.viewport.projection).Transform(Gf.Vec3d(*center))
    mouse = (point[0], point[1])
    path = overlay.pick(mouse)
    assert path and "/motor_run/" in path, path
    gesture = SimpleNamespace(sender=SimpleNamespace(gesture_payload=SimpleNamespace(mouse=mouse)))
    overlay.on_begin(gesture)
    tick(30)
    overlay.on_end(gesture)
    tick(90)
    assert device.rotating
    inspector.editor.set_value(json.dumps(device.config))
    assert inspector.apply()
    inspector.refresh()
    overlay.close()
    inspector.close()
    click("motor_run")
    tick(120)
    record("headless_viewport_ray_gesture_and_config_editor")
    click("motor_run")
    tick(120)
    assert device.rotating and abs(device.controls.read_state()["rpm"] - 60) < 2
    record("run_button_starts_real_rotation")
    click("motor_run")
    tick(120)
    assert not device.rotating and abs(device.controls.read_state()["rpm"]) < 1
    record("run_button_stops_real_rotation")
    # A held button cannot retrigger. It returns under its own spring after release.
    initial = len(device.events)
    device.press("motor_run", held=True)
    tick(180)
    travel = float(device.articulation.get_joint_positions()[device.indices["motor_run"]])
    assert len(device.events) == initial + 1 and 0.0012 < travel < 0.0026, travel
    device.release("motor_run")
    tick(90)
    assert abs(device.articulation.get_joint_positions()[device.indices["motor_run"]]) < 0.0003
    record("spring_travel_release_and_hold_debounce", travel_m=travel)
    click("motor_power")
    tick(120)
    assert not device.motor_enabled and not device.rotating and abs(device.controls.read_state()["rpm"]) < 1
    click("motor_run")
    assert not device.rotating
    click("motor_power")
    assert device.motor_enabled and not device.rotating
    record("motor_power_interlock_no_auto_restart")
    click("bath_up")
    assert device.temperature_edit_c == 41 and device.temperature_set_c == 40
    click("bath_set")
    assert device.temperature_set_c == 41
    click("bath_down")
    click("bath_set")
    assert device.temperature_set_c == 40
    click("bath_mode")
    assert device.heater_enabled
    record("all_bath_buttons")
    click("motor_run")
    tick(180)
    # Sample the real articulation and authored display on every physics step.
    # The old endpoint-only test missed a dial completing its turn in one frame.
    for requested in (90, 30, 90):
        before = device.state()
        device.set_dial_rpm(requested)
        assert device.rpm_set == before["rpm_set"]
        angles, speeds, readings = [before["dial_angle_deg"]], [], []
        distance = abs(requested - before["rpm_set"]) / 120 * 300
        speed = device.config["dial"].get("max_speed_deg_s", 60)
        accel = device.config["dial"].get("max_accel_deg_s2", 180)
        count = math.ceil((distance / speed + 2 * speed / accel + 0.6) * 120)
        for frame in range(count):
            tick(1)
            state = device.state()
            angles.append(state["dial_angle_deg"])
            speeds.append(state["rpm"])
            mapped = state["dial_angle_deg"] / 300 * 120
            assert abs(state["rpm_set"] - mapped) < 0.3, state
            if frame % 2 == 0:
                chars = ""
                for digit in range(3):
                    mask = "".join(
                        seg
                        for seg in "abcdefg"
                        if world.stage.GetPrimAtPath(f"{device.root}/lift/Display_rpm/Digits/d{digit}_{seg}")
                        .GetAttribute("visibility")
                        .Get()
                        != "invisible"
                    )
                    chars += next(char for char in "0123456789" if set(DIGITS[char]) == set(mask))
                value = int(chars)
                displayed_source = "rpm" if device.config["displays"]["rpm"]["source"] == "rpm_actual" else "rpm_set"
                assert abs(value - state[displayed_source]) < 1.2, (value, state)
                readings.append(value)
        direction = 1 if requested > before["rpm_set"] else -1
        delta = np.diff(angles)
        assert max(abs(delta)) <= speed / 120 * 1.15 + 0.02, max(abs(delta))
        assert min(delta * direction) > -0.01, min(delta * direction)
        assert len(set(round(v, 1) for v in angles)) > 40
        assert max(abs(np.diff(readings))) <= 1, readings
        assert abs(angles[-1] - requested / 120 * 300) < 1
        assert abs(speeds[-1] - requested) < 1
        record(
            "continuous_dial_and_sequential_display",
            requested_rpm=requested,
            max_step_degrees=float(max(abs(delta))),
            distinct_display_values=len(set(readings)),
            sampled_physics_steps=count,
        )
    record("dial_joint_controls_motor_speed", dial_degrees=angles[-1])
    device.activate_path(device.root + "/mains_switch/Collisions/rocker")
    tick(300)
    assert not device.mains_on and not device.rotating and not device.heater_enabled
    assert abs(device.controls.read_state()["rpm"]) < 1
    click("motor_run")
    assert not device.rotating and not device.events[-1]["accepted"]
    record("mains_rocker_cuts_motor_and_heater")
    device.set_switch(True)
    tick(120)
    assert device.mains_on and not device.rotating
    record("power_restore_does_not_restart")
    # Rebind a real button and replace both texture/background and display color.
    original = device.config_path
    config = json.loads(original.read_text())
    for display in config["displays"].values():
        display["texture"] = str(original.parent / display["texture"])
    config["buttons"]["bath_mode"] = {"action": "toggle_rotation"}
    config["displays"]["rpm"].update(
        mode="image", texture=str(original.parent / "textures/custom_example.png"), color=[0.1, 0.9, 0.9]
    )
    config["switch"]["action"] = "heater_power"
    with tempfile.TemporaryDirectory(prefix="evaporator_config_") as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config))
        device.config_path = path
        device.reload_config()
        click("bath_mode")
        assert device.rotating
        texture = (
            world.stage.GetPrimAtPath(device.root + "/Materials/LCD_rpm/MDL")
            .GetAttribute("inputs:diffuse_texture")
            .Get()
        )
        assert texture.path.endswith("custom_example.png")
        assert (
            world.stage.GetPrimAtPath(device.root + "/lift/Display_rpm/Digits/d0_a").GetAttribute("visibility").Get()
            == "invisible"
        )
        device.set_switch(False)
        tick(120)
        assert device.mains_on and device.rotating and not device.heater_enabled
        device.set_switch(True)
        tick(120)
        assert device.heater_enabled and device.mains_on
        record("hot_reload_button_switch_and_display_texture")
        config["buttons"]["bath_mode"]["action"] = "invalid_action"
        path.write_text(json.dumps(config))
        try:
            device.reload_config()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid config was accepted")
        assert device.config["buttons"]["bath_mode"]["action"] == "toggle_rotation"
        record("invalid_config_preserves_previous_bindings")
    # The lower black lever has its own hinge. A drive command moves that
    # joint first; the head follows the measured lever angle through controls.
    device.rotating = False
    device.controls.home()
    tick(240)
    lever_index = device.controls.lever_index
    device.controls.set_lever_degrees(-12)
    tick(240)
    lever_angle = math.degrees(float(device.articulation.get_joint_positions()[lever_index]))
    assert abs(lever_angle + 12) < 0.6, lever_angle
    assert abs(device.controls.read_state()["tilt_deg"] + 12) < 0.7, device.state()
    record("black_lever_drives_head_tilt", lever_degrees=lever_angle)
    device.controls.set_tilt_degrees(2)
    tick(240)
    assert abs(math.degrees(float(device.articulation.get_joint_positions()[lever_index])) - 2) < 0.6
    record("head_controls_synchronize_lower_lever")
    device.controls.home()
    tick(240)
    # Contact-equivalent external torque on the real lever joint, bypassing the
    # command API; the passive detent and head must follow measured displacement.
    for _ in range(90):
        device.update(1 / 120)
        device.articulation.apply_action(
            ArticulationAction(
                joint_efforts=np.array([-0.09], dtype=np.float32),
                joint_indices=np.array([lever_index]),
            )
        )
        world.step(render=False)
        steps += 1
    tick(180)
    lever_angle = math.degrees(float(device.articulation.get_joint_positions()[lever_index]))
    assert -20.4 <= lever_angle < -2, lever_angle
    assert abs(device.controls.read_state()["tilt_deg"] - lever_angle) < 1.0
    record("external_lever_torque_moves_head", lever_degrees=lever_angle)
    device.controls.home()
    tick(240)
    # Camera ray and native gesture callbacks operate the lower paddle too.
    overlay = DeviceViewport(device)
    for _ in range(3):
        world.render()
    paddle = (
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(world.stage.GetPrimAtPath(device.root + "/tilt_lever/Collisions/paddle"))
        .ExtractTranslation()
    )
    hit = get_physx_scene_query_interface().raycast_closest(tuple(paddle + Gf.Vec3d(0, -0.15, 0)), (0, 1, 0), 0.3)
    assert hit["hit"] and "/tilt_lever/Collisions/paddle" in hit["collision"], hit
    point = (overlay.viewport.view * overlay.viewport.projection).Transform(paddle)
    mouse = (point[0], point[1])
    assert "/tilt_lever/" in (overlay.pick(mouse) or "")
    gesture.sender.gesture_payload.mouse = mouse
    overlay.on_begin(gesture)
    assert overlay.held == "tilt_lever"
    gesture.sender.gesture_payload.mouse = (mouse[0], mouse[1] + 0.125)
    overlay.on_change(gesture)
    tick(240)
    overlay.on_end(gesture)
    overlay.close()
    assert abs(device.controls.read_state()["tilt_deg"] + 10) < 0.8, device.state()
    record("native_viewport_black_lever_drag")
    device.controls.home()
    tick(240)
    device.config_path = original
    device.reload_config()
    device.reset()
    tick(120)
    return {
        "passed": True,
        "isaac_sim": "5.1",
        "physics_steps": steps,
        "dof_names": list(device.articulation.dof_names),
        "checks": checks,
    }
