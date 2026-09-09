"""Device electronics driven by real spring-button, dial and rocker joint states.

Import after SimulationApp initializes. Call update before every physics step.
Mouse presses apply forces, exactly the same threshold path as robot contact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from pxr import UsdGeom, UsdShade

from experiments.assets.rotary_evaporator_calibration import dial_angle_to_rpm, rpm_to_dial_angle
from experiments.assets.rotary_evaporator_displays import configure_display, set_value

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/rotary_evaporator/interaction_config.json"
BUTTONS = ("motor_power", "motor_run", "bath_mode", "bath_down", "bath_up", "bath_set")
ACTIONS = {
    "toggle_motor_power",
    "toggle_rotation",
    "start_rotation",
    "stop_rotation",
    "toggle_heater",
    "temperature_delta",
    "confirm_temperature",
    "set_rpm",
    "lift_delta",
    "tilt_delta",
    "toggle_hinge",
    "home",
    "stop",
    "none",
}


class EvaporatorDevice:
    def __init__(self, controls, stage, root: str = "/World/Evaporator", config_path: Path = DEFAULT_CONFIG):
        self.controls = controls
        self.articulation = controls.device
        self.stage, self.root = stage, root
        self.config_path = Path(config_path).resolve()
        self.callbacks: dict[str, Callable] = {}
        self.events: list[dict] = []
        self.elapsed = 0.0
        self.next_display = 0.0
        self.pending = dict.fromkeys(BUTTONS, 0.0)
        self.pressed = dict.fromkeys(BUTTONS, False)
        self.last_event = dict.fromkeys(BUTTONS, -10.0)
        self.indices = {
            name: self.articulation.get_dof_index(name + "_joint") for name in (*BUTTONS, "speed_dial", "mains_switch")
        }
        self.reload_config()
        self.reset()

    def reload_config(self, candidate: dict | None = None) -> None:
        """Validate the whole replacement before changing runtime state."""
        candidate = json.loads(self.config_path.read_text()) if candidate is None else candidate
        if not isinstance(candidate, dict) or not all(
            isinstance(candidate.get(key), dict)
            for key in ("buttons", "switch", "dial", "press", "displays", "initial")
        ):
            raise ValueError("Device configuration must be an object with all six required sections")
        if candidate.get("version") != 1 or set(candidate["buttons"]) != set(BUTTONS):
            raise ValueError("Configuration must contain version 1 and all six button IDs")
        for binding in candidate["buttons"].values():
            if not isinstance(binding, dict) or not isinstance(binding.get("action"), str):
                raise ValueError("Each button binding needs an action string")
            if binding["action"] not in ACTIONS | self.callbacks.keys():
                raise ValueError(f"Unknown button action: {binding['action']}")
            if "value" in binding and not math.isfinite(float(binding["value"])):
                raise ValueError("Button parameter must be finite")
        if candidate["switch"]["action"] not in {"mains_power", "motor_power", "heater_power", "rotation"}:
            raise ValueError("Switch action must be mains_power, motor_power, heater_power or rotation")
        dial = candidate["dial"]
        if not 0 <= dial["min_rpm"] < dial["max_rpm"] <= 120 or not 0 < dial["click_step_rpm"] <= 120:
            raise ValueError("Dial range must lie inside 0..120 rpm, with positive step")
        for key, default, limit in [("max_speed_deg_s", 60.0, 720), ("max_accel_deg_s2", 180.0, 3600)]:
            value = dial.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 < value <= limit
            ):
                raise ValueError(f"Invalid dial {key}")
        refresh = candidate.get("display_refresh_hz", 60)
        if (
            isinstance(refresh, bool)
            or not isinstance(refresh, int | float)
            or not math.isfinite(refresh)
            or not 1 <= refresh <= 120
        ):
            raise ValueError("Display refresh must be within 1..120 Hz")
        press = candidate["press"]
        if not 0 < press["release_m"] < press["threshold_m"] < 0.0022:
            raise ValueError("Button thresholds must lie inside the 2.2 mm travel")
        if not (0 < press["force_n"] <= 5 and 0.06 <= press["duration_s"] <= 2 and 0 <= press["debounce_s"] <= 2):
            raise ValueError("Invalid press force or timing")
        switch = candidate["switch"]
        if not -12 <= switch["off_angle_deg"] < -6 < 6 < switch["on_angle_deg"] <= 12:
            raise ValueError("Rocker endpoints must be within limits, beyond detection thresholds")
        if set(candidate["displays"]) != {"rpm", "bath"}:
            raise ValueError("Configuration must contain exactly the rpm and bath displays")
        for name in ("rpm", "bath"):
            display = candidate["displays"][name]
            if not isinstance(display, dict):
                raise ValueError("Each display must be a configuration object")
            if display["mode"] not in {"live", "image"} or display["source"] not in {
                "rpm_actual",
                "rpm_set",
                "temperature_set_c",
                "temperature_measured_c",
            }:
                raise ValueError("Unknown display mode/source")
            if not (self.config_path.parent / display["texture"]).is_file():
                raise ValueError(f"Missing texture: {display['texture']}")
            if len(display["color"]) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in display["color"]):
                raise ValueError("Display color must be three values in 0..1")
            if not math.isfinite(display["brightness"]) or not 0 <= display["brightness"] <= 100:
                raise ValueError("Invalid display brightness")
        initial = candidate["initial"]
        if not (0 <= initial["rpm"] <= 120 and 5 <= initial["temperature_set_c"] <= 95):
            raise ValueError("Invalid initial setpoints")
        self.config = candidate
        self._display_cache = {}
        self._lamp_cache = {}
        for name, settings in candidate["displays"].items():
            configure_display(self.stage, self.root, name, settings, self.config_path.parent)
        if hasattr(self, "rpm_set"):
            self.set_dial_rpm(self.rpm_target)
        self.next_display = 0
        logger.info("Loaded evaporator device config: {}", self.config_path)

    def reset(self) -> None:
        initial = self.config["initial"]
        self.mains_on = bool(initial["mains_on"])
        self.motor_enabled = bool(initial["motor_enabled"])
        self.rotating = self.heater_enabled = False
        self.temperature_set_c = float(initial["temperature_set_c"])
        self.temperature_edit_c = self.temperature_set_c
        self.temperature_measured_c = 22.0  # Observation input; no thermal simulation.
        self.dial_command = float(
            np.clip(self.articulation.get_joint_positions()[self.indices["speed_dial"]], 0, math.radians(300))
        )
        self.dial_velocity = 0.0
        self.dial_angle = self.dial_command
        self.dial_follow_until = self.elapsed + 0.2
        self.rpm_set = self._rpm_from_angle(self.dial_angle)
        self.rpm_target = float(initial["rpm"])
        self.pending = dict.fromkeys(BUTTONS, 0.0)
        self.pressed = dict.fromkeys(BUTTONS, False)
        self.last_event = dict.fromkeys(BUTTONS, -10.0)
        self.controls.stop()
        self.set_dial_rpm(self.rpm_target)
        self.switch_on = {
            "mains_power": self.mains_on,
            "motor_power": self.motor_enabled,
            "heater_power": self.heater_enabled,
            "rotation": self.rotating,
        }[self.config["switch"]["action"]]
        self._set_switch_target(self.switch_on)
        self.switch_settle_until = self.elapsed + 0.5

    def _target(self, name: str, value: float) -> None:
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.array([value], dtype=np.float32), joint_indices=np.array([self.indices[name]])
            )
        )

    def _set_switch_target(self, on: bool) -> None:
        self._target("mains_switch", math.radians(self.config["switch"]["on_angle_deg" if on else "off_angle_deg"]))

    def set_switch(self, on: bool) -> None:
        """Move the rocker with its drive; electronics react to measured angle."""
        self._set_switch_target(bool(on))
        self.switch_settle_until = self.elapsed + 0.12

    def _rpm_from_angle(self, angle: float) -> float:
        return dial_angle_to_rpm(angle)

    def set_dial_rpm(self, rpm: float) -> None:
        """Queue a destination; update() turns the physical dial through intermediate angles."""
        if not math.isfinite(rpm):
            raise ValueError("RPM must be finite")
        dial = self.config["dial"]
        self.rpm_target = float(np.clip(rpm, dial["min_rpm"], dial["max_rpm"]))

    def _update_dial(self, actual: float, dt: float) -> None:
        dial = self.config["dial"]
        goal = rpm_to_dial_angle(self.rpm_target)
        moving = abs(goal - self.dial_command) > 1e-7 or abs(self.dial_velocity) > 1e-7
        if moving:
            self.dial_follow_until = self.elapsed + 0.2
            max_speed = math.radians(dial.get("max_speed_deg_s", 60))
            acceleration = math.radians(dial.get("max_accel_deg_s2", 180))
            # Braking-distance profile. Substeps keep low frame rates stable too.
            remaining = dt
            while remaining > 1e-9:
                h = min(remaining, 1 / 120)
                error = goal - self.dial_command
                desired = math.copysign(min(max_speed, math.sqrt(2 * acceleration * abs(error))), error)
                velocity = self.dial_velocity + float(
                    np.clip(desired - self.dial_velocity, -acceleration * h, acceleration * h)
                )
                delta = (self.dial_velocity + velocity) * 0.5 * h
                if error * delta >= 0 and abs(delta) >= abs(error):
                    self.dial_command, self.dial_velocity = goal, 0.0
                else:
                    self.dial_command += delta
                    self.dial_velocity = velocity
                remaining -= h
        elif self.elapsed >= self.dial_follow_until and abs(actual - self.dial_command) > math.radians(0.5):
            # An idle knob turned by contact keeps its new position instead of springing back.
            self.dial_command = float(np.clip(actual, 0, math.radians(300)))
            self.dial_velocity = 0.0
            self.set_dial_rpm(self._rpm_from_angle(self.dial_command))
        self._target("speed_dial", self.dial_command)
        self.dial_angle = actual
        # Motor control and rpm_set displays follow measured motion, never the queued destination.
        self.rpm_set = self._rpm_from_angle(actual)

    def press(self, button: str, held: bool = False) -> None:
        if button not in BUTTONS:
            raise KeyError(button)
        self.pending[button] = float("inf") if held else self.config["press"]["duration_s"]

    def release(self, button: str) -> None:
        self.pending[button] = 0.0

    def activate_path(self, prim_path: str, reverse: bool = False) -> str | None:
        """Shared viewport/test picking route. Prim may be a mesh or collider."""
        if not prim_path.startswith(self.root + "/"):
            return None
        name = prim_path[len(self.root) + 1 :].split("/")[0]
        if name in BUTTONS:
            self.press(name)
        elif name == "mains_switch":
            self.set_switch(not self.switch_on)
        elif name == "speed_dial":
            self.set_dial_rpm(self.rpm_target + (-1 if reverse else 1) * self.config["dial"]["click_step_rpm"])
        else:
            return None
        return name

    def _dispatch(self, button: str) -> None:
        binding = self.config["buttons"][button]
        action, value = binding["action"], float(binding.get("value", 0))
        if not self.mains_on:
            self.events.append({"time": self.elapsed, "button": button, "action": action, "accepted": False})
            return
        c = self.controls
        if action == "toggle_motor_power":
            self.motor_enabled = not self.motor_enabled
            if not self.motor_enabled:
                self.rotating = False
        elif action in {"toggle_rotation", "start_rotation", "stop_rotation"}:
            desired = not self.rotating if action == "toggle_rotation" else action == "start_rotation"
            self.rotating = desired and self.motor_enabled
        elif action == "toggle_heater":
            self.heater_enabled = not self.heater_enabled
        elif action == "temperature_delta":
            self.temperature_edit_c = float(np.clip(self.temperature_edit_c + value, 5, 95))
        elif action == "confirm_temperature":
            self.temperature_set_c = self.temperature_edit_c
        elif action == "set_rpm":
            self.set_dial_rpm(value)
        elif action == "lift_delta":
            c.set_lift_mm(c.target_lift * 1000 + value)
        elif action == "tilt_delta":
            c.set_tilt_degrees(np.rad2deg(c.target_tilt) + value)
        elif action == "toggle_hinge":
            c.toggle_lock()
        elif action in {"stop", "home"}:
            self.rotating = False
            getattr(c, action)()
        elif action in self.callbacks:
            self.callbacks[action](self, binding)
        self.events.append({"time": self.elapsed, "button": button, "action": action, "accepted": True})
        self.events = self.events[-1000:]

    def _switch_changed(self, on: bool) -> None:
        self.switch_on = on
        action = self.config["switch"]["action"]
        if action == "mains_power":
            self.mains_on = on
            self.rotating = self.heater_enabled = False  # Never auto-restart after power returns.
        elif action == "motor_power":
            self.motor_enabled = on
            if not on:
                self.rotating = False
        elif action == "heater_power":
            self.heater_enabled = on and self.mains_on
        else:
            self.rotating = on and self.mains_on and self.motor_enabled
        self._set_switch_target(on)

    def update(self, dt: float) -> None:
        self.elapsed += dt
        q = self.articulation.get_joint_positions()
        press = self.config["press"]
        efforts = []
        for name in BUTTONS:
            position = float(q[self.indices[name]])
            if (
                position >= press["threshold_m"]
                and not self.pressed[name]
                and self.elapsed - self.last_event[name] >= press["debounce_s"]
            ):
                self.pressed[name] = True
                self.last_event[name] = self.elapsed
                self._dispatch(name)
            elif position < press["release_m"]:
                self.pressed[name] = False
            efforts.append(press["force_n"] if self.pending[name] > 0 else 0)
            self.pending[name] = max(0, self.pending[name] - dt)
        angle = float(q[self.indices["mains_switch"]])
        if self.elapsed > self.switch_settle_until:
            if angle > math.radians(6) and not self.switch_on:
                self._switch_changed(True)
            elif angle < math.radians(-6) and self.switch_on:
                self._switch_changed(False)
        self._update_dial(float(q[self.indices["speed_dial"]]), dt)
        if not self.mains_on or not self.motor_enabled or self.controls.free_hinge:
            self.rotating = False
        self.controls.set_rpm(self.rpm_set if self.rotating else 0)
        hinge_effort = self.controls.push_torque if self.controls.push_remaining > 0 else 0.0
        self.controls.update(dt)
        # Isaac Sim 5.1 apply_action zeroes non-addressed efforts; merge button
        # forces and hinge torque in one final command instead of overwriting.
        self.articulation.apply_action(
            ArticulationAction(
                joint_efforts=np.array([*efforts, hinge_effort], dtype=np.float32),
                joint_indices=np.array([*[self.indices[n] for n in BUTTONS], self.controls.indices["tilt_joint"]]),
            )
        )
        if self.elapsed >= self.next_display:
            self.refresh_displays()
            self.next_display = self.elapsed + 1 / self.config.get("display_refresh_hz", 60)

    def refresh_displays(self) -> None:
        # A capture may refresh after a physics step. Read that same pose instead
        # of displaying the preceding step's setting or the queued destination.
        self.dial_angle = float(self.articulation.get_joint_positions()[self.indices["speed_dial"]])
        self.rpm_set = self._rpm_from_angle(self.dial_angle)
        values = {
            "rpm_actual": self.controls.read_state()["rpm"],
            "rpm_set": self.rpm_set,
            "temperature_set_c": self.temperature_edit_c,
            "temperature_measured_c": self.temperature_measured_c,
        }
        for name, cfg in self.config["displays"].items():
            enabled = self.mains_on and (self.motor_enabled if name == "rpm" else True) and cfg["mode"] == "live"
            value = values[cfg["source"]]
            signature = (round(value * (10 if name == "bath" else 1)), enabled)
            if self._display_cache.get(name) != signature:
                set_value(self.stage, self.root, name, value, enabled)
                self._display_cache[name] = signature
            body = "lift" if name == "rpm" else "base"
            background = self.stage.GetPrimAtPath(f"{self.root}/{body}/Display_{name}/Background")
            powered = self.mains_on and (self.motor_enabled if name == "rpm" else True)
            visibility = "inherited" if powered else "invisible"
            attr = UsdGeom.Imageable(background).GetVisibilityAttr()
            if attr.Get() != visibility:
                attr.Set(visibility)
        # Authored static legends are left editable; status bulbs reflect actual electronics state.
        for suffix, on in [("", self.mains_on), ("_001", self.heater_enabled and self.mains_on)]:
            if self._lamp_cache.get(suffix) == on:
                continue
            prim = self.stage.GetPrimAtPath(self.root + "/base/Visuals/Bath___status_lamp" + suffix)
            if prim:
                material_name = "Cyan___membrane_buttons" if on else "Graphite___textured_polymer"
                mat = UsdShade.Material(self.stage.GetPrimAtPath(self.root + "/Materials/" + material_name))
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
                self._lamp_cache[suffix] = on

    def state(self) -> dict:
        q = self.articulation.get_joint_positions()
        return {
            "buttons": {
                name: {"pressed": self.pressed[name], "travel_m": float(q[self.indices[name]])} for name in BUTTONS
            },
            "rpm_target": self.rpm_target,
            "dial_command_deg": math.degrees(self.dial_command),
            "dial_command_speed_deg_s": math.degrees(self.dial_velocity),
            "dial_angle_deg": math.degrees(float(q[self.indices["speed_dial"]])),
            "switch_angle_deg": math.degrees(float(q[self.indices["mains_switch"]])),
            "temperature_measured_c": self.temperature_measured_c,
            "mains_on": self.mains_on,
            "motor_enabled": self.motor_enabled,
            "rotating": self.rotating,
            "heater_enabled": self.heater_enabled,
            "rpm_set": self.rpm_set,
            "temperature_set_c": self.temperature_set_c,
            "temperature_edit_c": self.temperature_edit_c,
            **self.controls.read_state(),
        }
