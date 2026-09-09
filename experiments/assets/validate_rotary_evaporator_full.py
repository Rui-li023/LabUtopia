"""Complete operating-cycle checks using the device script and actual PhysX joints."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import carb.input
import numpy as np
from pxr import UsdGeom, UsdPhysics

from experiments.assets.rotary_evaporator_controls import EvaporatorPanel
from experiments.assets.rotary_evaporator_device import BUTTONS
from experiments.assets.rotary_evaporator_displays import DIGITS


def validate_full_device(device, world, capture=None) -> dict:
    steps, checks = 0, []
    original = copy.deepcopy(device.config)
    c = device.controls

    def tick(count=120):
        nonlocal steps
        for _ in range(count):
            device.update(1 / 120)
            world.step(render=False)
            steps += 1
            q = device.articulation.get_joint_positions()
            assert np.isfinite(q).all(), q
            assert np.isfinite(device.articulation.get_joint_velocities()).all()

    def record(label, **values):
        checks.append({"check": label, **values, "state": device.state()})

    def click(button):
        previous = len(device.events)
        device.press(button)
        tick(60)
        assert len(device.events) == previous + 1, (button, device.events)

    def digits(name, expected):
        body = "lift" if name == "rpm" else "base"
        for index, char in enumerate(expected):
            for segment in "abcdefg":
                prim = world.stage.GetPrimAtPath(f"{device.root}/{body}/Display_{name}/Digits/d{index}_{segment}")
                visible = prim.GetAttribute("visibility").Get() != "invisible"
                assert visible == (segment in DIGITS[char]), (name, index, segment, expected)

    def snapshot(label):
        if capture:
            capture(label)

    # All six caps must work under external fingertip contact, not just a motor command.
    probe = world.scene.get_object("button_probe")
    for name in BUTTONS:
        device.reset()
        tick(150)
        center = np.array(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(world.stage.GetPrimAtPath(device.root + "/" + name))
            .ExtractTranslation()
        )
        previous = len(device.events)
        for offset in np.linspace(-0.02, -0.004, 100):
            probe.set_world_pose(position=center + np.array([0, offset, 0]))
            tick(1)
        tick(30)
        travel = device.state()["buttons"][name]["travel_m"]
        assert len(device.events) == previous + 1 and device.events[-1]["button"] == name, (name, travel, device.events)
        assert 0.0012 < travel < 0.0026, (name, travel)
        probe.set_world_pose(position=center + np.array([0, -0.3, 0]))
        tick(90)
        assert abs(device.state()["buttons"][name]["travel_m"]) < 0.0003
        record("external_contact_" + name, contact_travel_m=travel)
    device.reset()
    tick(150)
    click("motor_run")
    for requested, expected in [(0, 0), (30, 30), (60, 60), (120, 120), (500, 120), (-50, 0)]:
        dial = device.config["dial"]
        distance = abs(expected - device.rpm_set) / 120 * 300
        duration = (
            distance / dial.get("max_speed_deg_s", 60)
            + 2 * dial.get("max_speed_deg_s", 60) / dial.get("max_accel_deg_s2", 180)
            + 1
        )
        device.set_dial_rpm(requested)
        tick(int(np.ceil(duration * 120)))
        assert abs(c.read_state()["rpm"] - expected) < 1.5, device.state()
        assert abs(device.rpm_set - expected) < 1, device.state()
        digits("rpm", f"{round(c.read_state()['rpm']):03d}")
        record("speed_range", requested_rpm=requested, expected_rpm=expected)
    device.set_dial_rpm(60)
    tick(480)
    for _ in range(10):
        click("motor_run")
        tick(120)
        assert not device.rotating and abs(c.read_state()["rpm"]) < 1
        click("motor_run")
        tick(120)
        assert device.rotating and abs(c.read_state()["rpm"] - 60) < 1.5
    record("ten_start_stop_cycles")
    # Temperature bounds, confirmation and explicit external sensor input.
    config = copy.deepcopy(original)
    config["buttons"]["bath_up"]["value"] = 200
    config["buttons"]["bath_down"]["value"] = -200
    device.reload_config(config)
    click("bath_up")
    assert device.temperature_edit_c == 95
    click("bath_set")
    assert device.temperature_set_c == 95
    digits("bath", "950")
    click("bath_down")
    click("bath_set")
    assert device.temperature_set_c == 5
    digits("bath", "050")
    config["displays"]["bath"]["source"] = "temperature_measured_c"
    device.reload_config(config)
    device.temperature_measured_c = 37.5
    tick(60)
    digits("bath", "375")
    record("temperature_limits_confirm_and_sensor_display")
    snapshot("sensor_display_37_5")
    # Render two genuinely replaced screen textures, with both live digit sets hidden.
    for display in config["displays"].values():
        display["mode"] = "image"
        display["texture"] = "textures/custom_example.png"
    device.reload_config(config)
    tick(30)
    for name, body in [("rpm", "lift"), ("bath", "base")]:
        assert (
            world.stage.GetPrimAtPath(f"{device.root}/{body}/Display_{name}/Digits/d0_a")
            .GetAttribute("visibility")
            .Get()
            == "invisible"
        )
    snapshot("custom_displays")
    record("both_custom_screen_images")
    device.reload_config(original)
    device.reset()
    tick(150)
    # The actual combined native control panel must work with electronics enabled.
    panel = EvaporatorPanel(c, device)
    try:
        panel.rpm_model.set_value(75)
        panel.invoke("Spin / stop")
        tick(300)
        assert device.rotating and abs(c.read_state()["rpm"] - 75) < 1.5
        panel.lift_model.set_value(100)
        panel.tilt_model.set_value(-12)
        tick(420)
        assert abs(c.read_state()["lift_mm"] - 100) < 3 and abs(c.read_state()["tilt_deg"] + 12) < 1
        snapshot("raised_running")
        panel.invoke("Unlock hinge")
        tick(300)
        assert c.free_hinge and not device.rotating and abs(c.read_state()["rpm"]) < 1
        panel.invoke("Push down")
        tick(100)
        panel.invoke("Lock here")
        locked = c.read_state()["tilt_deg"]
        tick(180)
        assert abs(c.read_state()["tilt_deg"] - locked) < 0.5
        panel.invoke("Home")
        tick(420)
        assert c.read_state()["lift_mm"] < 3 and abs(c.read_state()["tilt_deg"]) < 1
        key = SimpleNamespace(type=carb.input.KeyboardEventType.KEY_PRESS, input=carb.input.KeyboardInput.P)
        assert panel.on_keyboard(key) is False
        tick(300)
        assert device.rotating
        panel.invoke("STOP / hold")
        tick(240)
        assert not device.rotating and abs(c.read_state()["rpm"]) < 1
        record("combined_panel_keyboard_hinge_and_home")
        world.pause()
        frozen = device.articulation.get_joint_positions().copy()
        for _ in range(5):
            world.render()
        assert np.allclose(device.articulation.get_joint_positions(), frozen, atol=1e-7)
        world.play()
        tick(120)
        record("timeline_pause_resume")
        # Exercise a real Stop/Reset after running with a held button.
        click("motor_run")
        device.press("bath_up", held=True)
        tick(60)
        world.stop()
        # DynamicSphere.reset writes velocities; temporarily make the test probe
        # dynamic while World resets it, then restore its kinematic contact role.
        probe_physics = UsdPhysics.RigidBodyAPI(world.stage.GetPrimAtPath("/World/ButtonProbe"))
        probe_physics.CreateKinematicEnabledAttr(False)
        world.reset()
        probe_physics.CreateKinematicEnabledAttr(True)
        device.reset()
        tick(180)
        assert not device.rotating and abs(c.read_state()["rpm"]) < 1
        assert all(not value for value in device.pressed.values())
        assert all(value == 0 for value in device.pending.values())
        assert abs(c.read_state()["lift_mm"]) < 3 and abs(c.read_state()["tilt_deg"]) < 1
        record("stop_reset_clears_held_inputs_and_motion")
    finally:
        panel.close()
    click("motor_run")
    click("bath_mode")
    tick(240)
    assert device.rotating and device.heater_enabled
    snapshot("running_heater_requested")
    device.set_switch(False)
    tick(300)
    assert not device.mains_on and not device.rotating and not device.heater_enabled
    digits("rpm", "   ")
    digits("bath", "   ")
    snapshot("shutdown")
    record("complete_shutdown_both_screens_off")
    device.set_switch(True)
    tick(180)
    assert not device.rotating and not device.heater_enabled
    device.reset()
    tick(150)
    return {
        "passed": True,
        "physics_steps": steps,
        "checks": checks,
        "scope": "Rigid mechanics and scripted electronics; thermal, vacuum and evaporation physics are not implemented.",
    }
