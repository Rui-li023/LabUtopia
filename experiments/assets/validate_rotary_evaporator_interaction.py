"""Physical interaction regression, invoked by simulate_rotary_evaporator.py."""

from __future__ import annotations

from types import SimpleNamespace

import carb.input
import carb.settings
import numpy as np
from isaacsim.core.prims import RigidPrim
from loguru import logger
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from omni.physx.bindings import _physx as physx_bindings
from pxr import Gf


def validate_interaction(controls, world, panel=None) -> dict:
    """Exercise actual drives, release/relock, external body torque and UI callbacks."""
    checks = []
    steps = 0

    def advance(count=360):
        nonlocal steps
        for _ in range(count):
            controls.update(1 / 120)
            world.step(render=panel is not None and steps % 12 == 0)
            steps += 1
            if panel is not None and steps % 24 == 0:
                panel.refresh()
            assert np.all(np.isfinite(controls.device.get_joint_positions()))

    def record(label):
        result = {"phase": label, **controls.read_state()}
        checks.append(result)
        logger.info("{}", result)
        return result

    # The same model callbacks used by sliders are tested in the GUI run.
    if panel:
        panel.lift_model.set_value(100.0)
        panel.tilt_model.set_value(-12.0)
    else:
        controls.set_lift_mm(100)
        controls.set_tilt_degrees(-12)
    advance(480)
    state = record("slider_targets")
    assert abs(state["lift_mm"] - 100) < 2
    assert abs(state["tilt_deg"] + 12) < 0.5

    if panel:
        panel.rpm_model.set_value(60.0)
    else:
        controls.set_rpm(60)
    advance(240)
    state = record("speed_control")
    assert abs(state["rpm"] - 60) < 1

    if panel:
        panel.invoke("STOP / hold")
    else:
        controls.stop()
    advance(120)
    state = record("stop_and_hold")
    assert abs(state["rpm"]) < 0.2

    controls.set_tilt_degrees(-10)
    advance(180)
    if panel:
        panel.invoke("Unlock hinge")
    else:
        controls.unlock()
    kp, _ = controls.controller.get_gains()
    assert kp[controls.indices["tilt_joint"]] == 0
    advance(180)
    before = record("unlocked_under_gravity")

    # An external world-space torque on the rigid body, not an angle command.
    body = RigidPrim(prim_paths_expr="/World/Evaporator/tilt", name="hinge_force_probe", reset_xform_properties=False)
    body.initialize()
    for _ in range(60):
        body.apply_forces_and_torques_at_pos(torques=np.array([[0, 4.0, 0]], dtype=np.float32), is_global=True)
        advance(1)
    after = record("external_body_torque")
    assert after["tilt_deg"] - before["tilt_deg"] > 3, (before, after)
    assert -20.3 <= after["tilt_deg"] <= 5.3
    # Exercise the same PhysX ray interaction used by Shift+left-drag, on the
    # condenser collider. Pull upward, against the free hinge's gravity.
    advance(180)
    initial_drag = record("before_mouse_drag")
    positions, orientations = body.get_world_poses()
    rotation = Gf.Rotation(Gf.Quatd(float(orientations[0][0]), Gf.Vec3d(*[float(v) for v in orientations[0][1:]])))
    handle = np.array(positions[0], dtype=float) + np.array(rotation.TransformDir(Gf.Vec3d(-0.138, -0.032, 0.237)))
    origin = handle + np.array([0, -1.0, 0])
    interface = get_physx_interface()
    event = physx_bindings.PhysicsInteractionEvent
    settings = carb.settings.get_settings()
    desired = {
        physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED: True,
        physx_bindings.SETTING_MOUSE_GRAB: True,
        physx_bindings.SETTING_MOUSE_GRAB_IGNORE_INVISBLE: False,
        physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE: True,
        physx_bindings.SETTING_MOUSE_PICKING_FORCE: 1.0,
    }
    old_settings = {key: settings.get(key) for key in desired}
    for key, value in desired.items():
        settings.set(key, value)
    hit = get_physx_scene_query_interface().raycast_closest(tuple(origin), (0, 1, 0), 2.0)
    logger.info("Mouse ray handle hit: {}", hit)
    assert hit["hit"] and "condenser" in hit["collision"], hit
    try:
        direction = np.array([0.0, 1.0, 0.0])
        interface.update_interaction(tuple(origin), tuple(direction), event.MOUSE_DRAG_BEGAN)
        advance(1)
        for frame in range(120):
            target = handle + np.array([0, 0, 0.14 * (frame + 1) / 120])
            direction = target - origin
            direction /= np.linalg.norm(direction)
            interface.update_interaction(tuple(origin), tuple(direction), event.MOUSE_DRAG_CHANGED)
            advance(1)
        dragged = record("native_physx_mouse_drag")
        assert dragged["tilt_deg"] - initial_drag["tilt_deg"] > 3, (initial_drag, dragged)
    finally:
        interface.update_interaction(tuple(origin), tuple(direction), event.MOUSE_DRAG_ENDED)
        for key, value in old_settings.items():
            if value is None:
                settings.destroy_item(key)
            else:
                settings.set(key, value)
    if panel:
        panel.invoke("Lock here")
    else:
        controls.lock_here()
    locked = controls.read_state()["tilt_deg"]
    advance(240)
    state = record("relock_current_angle")
    assert abs(state["tilt_deg"] - locked) < 0.5
    assert not state["free_hinge"]

    if panel:
        panel.invoke("Push up")
    else:
        controls.push(-1)
    advance(120)
    state = record("physical_push_button")
    assert state["tilt_deg"] < locked - 2
    assert controls.push_remaining == 0
    controls.lock_here()

    # Exercise keyboard dispatch, including the enable/disable switch.
    if panel:
        event = SimpleNamespace(type=carb.input.KeyboardEventType.KEY_PRESS, input=carb.input.KeyboardInput.U)
        old = controls.target_lift
        panel.keyboard_enabled = False
        panel.on_keyboard(event)
        assert controls.target_lift == old
        panel.keyboard_enabled = True
        panel.on_keyboard(event)
        assert abs(controls.target_lift - min(0.12, old + 0.01)) < 1e-6
        panel.invoke("Home")
    else:
        controls.home()
    advance(480)
    state = record("home_without_teleport")
    assert abs(state["lift_mm"]) < 2
    assert abs(state["tilt_deg"]) < 0.5
    assert abs(state["rpm"]) < 0.2

    controls.set_lift_mm(1000)
    controls.set_tilt_degrees(-1000)
    controls.set_rpm(1000)
    assert controls.target_lift == 0.12
    assert controls.target_tilt == np.deg2rad(-20)
    assert controls.target_rpm == 120
    controls.home()
    advance(30)
    return {
        "passed": True,
        "physics_steps": steps,
        "ui_callbacks_tested": panel is not None,
        "checks": checks,
        "external_force_is_physical": True,
    }
