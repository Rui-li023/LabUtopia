"""Interactive controls for the evaporator. Import after SimulationApp starts.

Targets use mm, degrees and rpm at the UI boundary; only physical articulation
commands are sent to Isaac Sim. No USD transforms or joint poses are teleported.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import carb.input
import carb.settings
import numpy as np
import omni.appwindow
import omni.ui as ui
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from omni.physx.bindings import _physx as physx_bindings


class EvaporatorControls:
    """Rate-limited joint targets, compliant hinge mode, and physical push pulses."""

    def __init__(self, articulation: SingleArticulation) -> None:
        self.device = articulation
        self.indices = {name: articulation.get_dof_index(name) for name in ("lift_joint", "tilt_joint", "spin_joint")}
        self.controller = articulation.get_articulation_controller()
        kp, kd = self.controller.get_gains()
        self.original_kp = np.array(kp, copy=True)
        self.original_kd = np.array(kd, copy=True)
        self.free_hinge = False
        self.push_torque = 0.0
        self.push_remaining = 0.0
        self.target_lift = 0.0
        self.target_tilt = 0.0
        self.target_rpm = 0.0
        self.command = np.zeros(3, dtype=np.float32)
        self.capture_current_pose()
        self.lever_index = articulation.get_dof_index("tilt_lever_joint")
        self.lever_target = self.target_tilt
        self.lever_last_main = self.target_tilt
        self.lever_input_remaining = 0.0
        self.lever_settle_remaining = 0.5

    @staticmethod
    def _bounded(value: float, lower: float, upper: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Control targets must be finite")
        return float(np.clip(value, lower, upper))

    def read_state(self) -> dict[str, float | bool]:
        q = self.device.get_joint_positions()
        qd = self.device.get_joint_velocities()
        return {
            "lift_mm": float(q[self.indices["lift_joint"]] * 1000),
            "tilt_deg": float(np.rad2deg(q[self.indices["tilt_joint"]])),
            "rpm": float(qd[self.indices["spin_joint"]] * 60 / (2 * np.pi)),
            "free_hinge": self.free_hinge,
        }

    def capture_current_pose(self) -> None:
        state = self.read_state()
        self.target_lift = float(np.clip(state["lift_mm"] / 1000, 0, 0.12))
        self.target_tilt = float(np.clip(np.deg2rad(state["tilt_deg"]), np.deg2rad(-20), np.deg2rad(5)))
        self.target_rpm = 0.0
        self.command[:] = [self.target_lift, self.target_tilt, 0]
        self.push_torque = self.push_remaining = 0.0

    def set_lift_mm(self, millimetres: float) -> None:
        self.target_lift = self._bounded(millimetres, 0, 120) / 1000

    def set_tilt_degrees(self, degrees: float) -> None:
        goal = np.deg2rad(self._bounded(degrees, -20, 5))
        if self.free_hinge:
            self.lock_here()
        self.target_tilt = float(goal)

    def set_lever_degrees(self, degrees: float) -> None:
        """Drive the lower black lever; measured travel commands the head hinge."""
        if self.free_hinge:
            self.lock_here()
        self.lever_target = math.radians(self._bounded(degrees, -20, 5))
        self.lever_last_main = self.target_tilt
        self.lever_input_remaining = 0.8
        self.lever_settle_remaining = 0.0

    def _update_lever(self, dt: float) -> None:
        actual = float(self.device.get_joint_positions()[self.lever_index])
        self.lever_input_remaining = max(0.0, self.lever_input_remaining - dt)
        self.lever_settle_remaining = max(0.0, self.lever_settle_remaining - dt)
        if self.free_hinge:
            self.lever_target = float(self.device.get_joint_positions()[self.indices["tilt_joint"]])
            self.lever_settle_remaining = 0.5
            self.lever_input_remaining = 0.0
        elif abs(self.target_tilt - self.lever_last_main) > 1e-6:
            # Panel, scripts, home and stop take precedence over a previous drag.
            self.lever_target = self.target_tilt
            self.lever_settle_remaining = 0.5
            self.lever_input_remaining = 0.0
        elif self.lever_input_remaining > 0:
            self.target_tilt = float(np.clip(actual, math.radians(-20), math.radians(5)))
        elif self.lever_settle_remaining == 0 and abs(actual - self.lever_target) > math.radians(0.7):
            # Robot contact / native PhysX dragging moves the same physical joint.
            self.lever_target = float(np.clip(actual, math.radians(-20), math.radians(5)))
            self.target_tilt = self.lever_target
        self.lever_last_main = self.target_tilt
        self.device.apply_action(
            ArticulationAction(
                joint_positions=np.array([self.lever_target], dtype=np.float32),
                joint_indices=np.array([self.lever_index]),
            )
        )

    def set_rpm(self, rpm: float) -> None:
        self.target_rpm = self._bounded(rpm, 0, 120)

    def unlock(self) -> None:
        """Release the hinge position drive; contacts and joint limits stay active."""
        kp, kd = self.original_kp.copy(), self.original_kd.copy()
        kp[self.indices["tilt_joint"]] = 0.0
        kd[self.indices["tilt_joint"]] = 0.25
        self.controller.set_gains(kps=kp, kds=kd)
        self.free_hinge = True
        self.target_rpm = 0.0
        self.push_remaining = self.push_torque = 0.0
        self._apply_effort(0)

    def lock_here(self) -> None:
        """Re-engage the drive at the current angle, without snapping to an old target."""
        angle = float(self.device.get_joint_positions()[self.indices["tilt_joint"]])
        self.target_tilt = float(np.clip(angle, np.deg2rad(-20), np.deg2rad(5)))
        self.command[1] = self.target_tilt
        self.controller.set_gains(kps=self.original_kp.copy(), kds=self.original_kd.copy())
        self.free_hinge = False
        self.push_remaining = self.push_torque = 0.0
        self._apply_effort(0)
        self.device.apply_action(
            ArticulationAction(
                joint_positions=np.array([self.target_tilt], dtype=np.float32),
                joint_indices=np.array([self.indices["tilt_joint"]]),
            )
        )

    def toggle_lock(self) -> None:
        if self.free_hinge:
            self.lock_here()
        else:
            self.unlock()

    def push(self, direction: float) -> None:
        """A 3 N m hinge torque for 0.35 s; movement is produced by PhysX."""
        direction = self._bounded(direction, -1, 1)
        if not self.free_hinge:
            self.unlock()
        self.push_torque = direction * 3.0
        self.push_remaining = 0.35

    def stop(self) -> None:
        """Stop rotation and hold both positioning axes where they are now."""
        self.capture_current_pose()
        self.lock_here()
        self.command[2] = 0.0
        self.update(0.0)

    def home(self) -> None:
        self.lock_here()
        self.target_lift = self.target_tilt = self.target_rpm = 0.0

    def toggle_spin(self) -> None:
        self.target_rpm = 0.0 if self.target_rpm > 0 else 60.0

    def _apply_effort(self, torque: float) -> None:
        self.device.apply_action(
            ArticulationAction(
                joint_efforts=np.array([torque], dtype=np.float32),
                joint_indices=np.array([self.indices["tilt_joint"]]),
            )
        )

    def update(self, dt: float) -> None:
        """Call before each physics step, with simulation dt in seconds."""
        if hasattr(self, "lever_index"):
            self._update_lever(dt)
        goal = np.array([self.target_lift, self.target_tilt, self.target_rpm * 2 * np.pi / 60])
        # 40 mm/s, 15 degrees/s, 60 rpm/s; use simulation time, not UI frame rate.
        rate = np.array([0.04, np.deg2rad(15), 2 * np.pi])
        self.command += np.clip(goal - self.command, -rate * dt, rate * dt)
        if self.free_hinge:
            self.command[1] = self.device.get_joint_positions()[self.indices["tilt_joint"]]
        self.device.apply_action(
            ArticulationAction(
                joint_positions=self.command[:2].copy(),
                joint_indices=np.array([self.indices["lift_joint"], self.indices["tilt_joint"]]),
            )
        )
        self.device.apply_action(
            ArticulationAction(
                joint_velocities=np.array([self.command[2]], dtype=np.float32),
                joint_indices=np.array([self.indices["spin_joint"]]),
            )
        )
        self._apply_effort(self.push_torque if self.push_remaining > 0 else 0.0)
        self.push_remaining = max(0.0, self.push_remaining - dt)


class EvaporatorPanel:
    """Native Isaac Sim window and keyboard bindings for the physical device."""

    def __init__(self, controls: EvaporatorControls, electronics=None) -> None:
        self.controls = controls
        self.electronics = electronics
        self.syncing = False
        self.keyboard_enabled = True
        self.active = True
        self.window = ui.Window("Rotary Evaporator | Interactive", width=370, height=660)
        self.window.position_x = 18
        self.window.position_y = 75
        self.actions: dict[str, Callable[[], None]] = {
            "Tilt up": lambda: controls.set_tilt_degrees(-20),
            "Tilt down": lambda: controls.set_tilt_degrees(5),
            "Raise": lambda: controls.set_lift_mm(120),
            "Lower": lambda: controls.set_lift_mm(0),
            "Unlock hinge": controls.unlock,
            "Lock here": controls.lock_here,
            "Push up": lambda: controls.push(-1),
            "Push down": lambda: controls.push(1),
            "Spin / stop": controls.toggle_spin,
            "Home": controls.home,
            "STOP / hold": controls.stop,
        }
        if electronics:
            self.actions["Spin / stop"] = lambda: electronics.press("motor_run")
            self.actions["Home"] = lambda: self._device_stop(home=True)
            self.actions["STOP / hold"] = self._device_stop
        with self.window.frame, ui.VStack(spacing=5):
            ui.Label("ROTARY EVAPORATOR", height=24, style={"font_size": 19})
            ui.Label("Physical hinge controls", height=18)
            self.status = ui.Label("", height=20)
            self.feedback = ui.Label("", height=38, word_wrap=True)
            ui.Separator(height=2)
            ui.Label("Hinge angle (degrees)", height=18)
            self.tilt_model = self._slider(-20, 5, lambda v: controls.set_tilt_degrees(v))
            self._buttons("Tilt up", "Tilt down")
            ui.Label("Lift height (mm)", height=18)
            self.lift_model = self._slider(0, 120, lambda v: controls.set_lift_mm(v))
            self._buttons("Raise", "Lower")
            ui.Label("Flask speed (rpm)", height=18)
            self.rpm_model = self._slider(
                0, 120, lambda v: electronics.set_dial_rpm(v) if electronics else controls.set_rpm(v)
            )
            self._buttons("Spin / stop", "Home")
            ui.Separator(height=2)
            self._buttons("Unlock hinge", "Lock here")
            self._buttons("Push up", "Push down")
            self._buttons("STOP / hold")
            ui.Label(
                "Unlock, then Shift + left-drag the motor housing.\nJ/L tilt | U/O lift | G lock | P spin | H home | X stop",
                height=42,
                word_wrap=True,
            )
            with ui.HStack(height=20):
                key_model = ui.CheckBox(width=22).model
                key_model.set_value(True)
                key_model.add_value_changed_fn(lambda m: setattr(self, "keyboard_enabled", m.as_bool))
                ui.Label("Keyboard shortcuts enabled")
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_keyboard)
        # PhysX's built-in Shift-drag applies force to the picked rigid body.
        self.settings = carb.settings.get_settings()
        desired = {
            physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED: True,
            physx_bindings.SETTING_MOUSE_GRAB: True,
            # Collision proxies are hidden while their visual meshes stay visible.
            physx_bindings.SETTING_MOUSE_GRAB_IGNORE_INVISBLE: False,
            physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE: True,
            physx_bindings.SETTING_MOUSE_PICKING_FORCE: 1.0,
        }
        self.previous_mouse_settings = {key: self.settings.get(key) for key in desired}
        for key, value in desired.items():
            self.settings.set(key, value)
        self.refresh()

    def _device_stop(self, home=False):
        self.electronics.rotating = False
        self.controls.home() if home else self.controls.stop()

    def _slider(self, low: float, high: float, action: Callable[[float], None]) -> ui.AbstractValueModel:
        with ui.HStack(height=22, spacing=6):
            model = ui.FloatSlider(min=low, max=high, step=0.1).model
            ui.FloatField(model=model, width=70, precision=1)
        model.add_value_changed_fn(lambda m: action(m.as_float) if not self.syncing else None)
        return model

    def _buttons(self, *labels: str) -> None:
        with ui.HStack(height=27, spacing=6):
            for label in labels:
                ui.Button(label, clicked_fn=lambda name=label: self.invoke(name))

    def invoke(self, action: str) -> None:
        self.actions[action]()
        self.refresh()

    def on_keyboard(self, event, *_args) -> bool:
        if (
            not self.active
            or not self.keyboard_enabled
            or event.type
            not in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            )
        ):
            return True
        if event.type == carb.input.KeyboardEventType.KEY_REPEAT and event.input not in (
            carb.input.KeyboardInput.J,
            carb.input.KeyboardInput.L,
            carb.input.KeyboardInput.U,
            carb.input.KeyboardInput.O,
        ):
            return True
        c = self.controls
        key_actions = {
            carb.input.KeyboardInput.J: lambda: c.set_tilt_degrees(np.rad2deg(c.target_tilt) - 2),
            carb.input.KeyboardInput.L: lambda: c.set_tilt_degrees(np.rad2deg(c.target_tilt) + 2),
            carb.input.KeyboardInput.U: lambda: c.set_lift_mm(c.target_lift * 1000 + 10),
            carb.input.KeyboardInput.O: lambda: c.set_lift_mm(c.target_lift * 1000 - 10),
            carb.input.KeyboardInput.G: c.toggle_lock,
            carb.input.KeyboardInput.P: self.actions["Spin / stop"],
            carb.input.KeyboardInput.H: self.actions["Home"],
            carb.input.KeyboardInput.X: self.actions["STOP / hold"],
        }
        callback = key_actions.get(event.input)
        if callback is None:
            return True
        callback()
        self.refresh()
        return False

    def refresh(self, paused: bool = False) -> None:
        state = self.controls.read_state()
        mode = "FREE: drag housing / apply forces" if self.controls.free_hinge else "DRIVEN: position hold"
        self.status.text = ("PAUSED | " if paused else "") + mode
        self.feedback.text = (
            f"Actual  {state['tilt_deg']:+.1f} deg  |  {state['lift_mm']:.1f} mm\nRotation  {state['rpm']:.1f} rpm"
        )
        self.syncing = True
        self.tilt_model.set_value(float(np.rad2deg(self.controls.target_tilt)))
        self.lift_model.set_value(float(self.controls.target_lift * 1000))
        self.rpm_model.set_value(float(self.electronics.rpm_target if self.electronics else self.controls.target_rpm))
        self.syncing = False

    def close(self) -> None:
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        for key, value in self.previous_mouse_settings.items():
            if value is None:
                self.settings.destroy_item(key)
            else:
                self.settings.set(key, value)
        self.window.destroy()
