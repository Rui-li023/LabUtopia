"""On-model mouse gestures and editable device configuration in Isaac Sim."""

from __future__ import annotations

import json

import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport_window
from omni.physx import get_physx_scene_query_interface
from omni.ui import scene as sc
from pxr import Gf

from experiments.assets.rotary_evaporator_device import BUTTONS


class DeviceViewport:
    """Pick live collision geometry with a camera ray; never depend on selection changes."""

    def __init__(self, device):
        self.device = device
        self.active = True
        self.held = None
        self.start_mouse = (0, 0)
        self.start_lever_deg = 0
        self.start_rpm = 0
        self.start_time = 0
        self.window = get_active_viewport_window()
        self.viewport = self.window.viewport_api
        self.frame = self.window.get_frame("EvaporatorPhysicalControls")
        with self.frame:
            self.scene_view = sc.SceneView()
            self.viewport.add_scene_view(self.scene_view)
            with self.scene_view.scene:
                self.screen = sc.Screen(
                    gestures=[
                        sc.DragGesture(
                            mouse_button=0,
                            modifiers=0,
                            on_began_fn=self.on_begin,
                            on_changed_fn=self.on_change,
                            on_ended_fn=self.on_end,
                        ),
                        sc.ClickGesture(mouse_button=1, modifiers=0, on_ended_fn=self.on_reverse),
                    ]
                )

    def pick(self, mouse) -> str | None:
        inverse = (self.viewport.view * self.viewport.projection).GetInverse()
        start = inverse.Transform(Gf.Vec3d(mouse[0], mouse[1], -1))
        end = inverse.Transform(Gf.Vec3d(mouse[0], mouse[1], 1))
        direction = (end - start).GetNormalized()
        hit = get_physx_scene_query_interface().raycast_closest(tuple(start), tuple(direction), 100)
        return hit["collision"] if hit["hit"] else None

    def on_begin(self, gesture):
        if not self.active:
            return
        mouse = tuple(gesture.sender.gesture_payload.mouse)
        path = self.pick(mouse)
        if not path or not path.startswith(self.device.root + "/"):
            return
        name = path[len(self.device.root) + 1 :].split("/")[0]
        self.start_mouse, self.start_rpm = mouse, self.device.rpm_set
        self.start_time = self.device.elapsed
        if name in BUTTONS:
            self.held = name
            self.device.press(name, held=True)
        elif name == "speed_dial":
            self.held = name
        elif name == "tilt_lever":
            self.held = name
            self.start_lever_deg = self.device.controls.read_state()["tilt_deg"]
        elif name == "mains_switch":
            self.device.activate_path(path)

    def on_change(self, gesture):
        if self.active and self.held == "tilt_lever":
            delta = gesture.sender.gesture_payload.mouse[1] - self.start_mouse[1]
            self.device.controls.set_lever_degrees(self.start_lever_deg - delta * 80)
        if self.active and self.held == "speed_dial":
            mouse = gesture.sender.gesture_payload.mouse
            delta = mouse[1] - self.start_mouse[1]
            if abs(delta) > 0.003:
                self.device.set_dial_rpm(self.start_rpm + delta * 180)

    def on_end(self, gesture):
        if self.held in BUTTONS:
            # A quick click gets a minimum physical pulse; longer holds release immediately.
            self.device.pending[self.held] = max(
                0, self.device.config["press"]["duration_s"] - (self.device.elapsed - self.start_time)
            )
        elif self.active and self.held == "speed_dial":
            mouse = gesture.sender.gesture_payload.mouse
            if abs(mouse[1] - self.start_mouse[1]) < 0.003:
                self.device.set_dial_rpm(self.device.rpm_target + self.device.config["dial"]["click_step_rpm"])
        self.held = None

    def on_reverse(self, gesture):
        if self.active:
            path = self.pick(gesture.sender.gesture_payload.mouse)
            if path and "/speed_dial/" in path:
                self.device.activate_path(path, reverse=True)

    def cancel(self):
        if self.held in BUTTONS:
            self.device.release(self.held)
        self.held = None
        self.active = False

    def close(self):
        self.cancel()
        self.viewport.remove_scene_view(self.scene_view)
        self.scene_view.scene.clear()
        self.frame.clear()


class DeviceInspector:
    """One editable JSON document covers all button, switch and screen properties."""

    def __init__(self, device):
        self.device = device
        self.window = ui.Window("Evaporator | Device settings", width=390, height=640)
        self.window.position_x, self.window.position_y = 1035, 75
        with self.window.frame, ui.VStack(spacing=5):
            ui.Label("ON-MODEL CONTROLS", height=25, style={"font_size": 18})
            ui.Label(
                "Press the six buttons directly on the model.\nDial: drag up/down; left/right click = +/- step.\nClick the side rocker to switch power.\nBlack lower lever: drag vertically to tilt the head.",
                height=76,
                word_wrap=True,
            )
            self.status = ui.Label("", height=48, word_wrap=True)
            ui.Label("Edit button actions, switch, textures and display style:", height=22)
            with ui.ScrollingFrame(height=365):
                self.editor = ui.StringField(multiline=True).model
            self.editor.set_value(json.dumps(device.config, indent=2))
            with ui.HStack(height=28, spacing=5):
                ui.Button("Apply", clicked_fn=self.apply)
                ui.Button("Save config", clicked_fn=self.save)
                ui.Button("Reload file", clicked_fn=self.reload)
            self.message = ui.Label("Apply changes runtime; Save writes the config file.", height=36, word_wrap=True)

    def apply(self):
        try:
            self.device.reload_config(json.loads(self.editor.as_string))
            self.message.text = "Applied. Press a model button to try its new action."
            return True
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.message.text = str(error)
            return False

    def save(self):
        if self.apply():
            path = self.device.config_path
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.device.config, indent=2) + "\n")
            temporary.replace(path)
            self.message.text = "Saved " + path.name

    def reload(self):
        try:
            self.device.reload_config()
            self.editor.set_value(json.dumps(self.device.config, indent=2))
            self.message.text = "Reloaded " + self.device.config_path.name
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.message.text = str(error)

    def refresh(self):
        d = self.device
        self.status.text = f"Mains {'ON' if d.mains_on else 'OFF'} | Motor {'RUN' if d.rotating else 'STOP'} | Heat {'ON' if d.heater_enabled else 'OFF'}\nSpeed {d.rpm_set:.0f} rpm | Temp target {d.temperature_edit_c:.1f} C"

    def close(self):
        self.window.destroy()
