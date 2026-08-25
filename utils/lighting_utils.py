"""Lighting randomization utilities for domain randomization in LabUtopia.

Supports randomizing exposure, color temperature, light color, type, intensity,
and position to simulate real-world lighting variations (warm/cool light sources,
different light types, etc.).
"""

import os
import random
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom

# ---------------------------------------------------------------------------
# Color temperature presets (Kelvin ranges for real-world light sources)
# ---------------------------------------------------------------------------
COLOR_TEMP_PRESETS: Dict[str, Tuple[float, float]] = {
    "candle": (1800.0, 2200.0),
    "warm_white": (2700.0, 3200.0),  # Incandescent / warm LED
    "soft_white": (3000.0, 3500.0),  # Halogen
    "neutral": (3800.0, 4200.0),  # Fluorescent neutral
    "cool_white": (4500.0, 5000.0),  # Fluorescent cool
    "daylight": (5000.0, 5500.0),  # Direct sunlight
    "overcast": (6000.0, 7000.0),  # Cloudy sky
    "blue_sky": (8000.0, 12000.0),  # Clear blue sky
}

# ---------------------------------------------------------------------------
# Lab-realistic lighting scenarios
# ---------------------------------------------------------------------------
LAB_LIGHTING_SCENARIOS: Dict[str, Dict] = {
    "standard_lab": {
        "color_temp_range": (4500.0, 8500.0),
        "intensity_range": (500.0, 2000.0),
        "exposure_range": (0.0, 1.0),
        "light_types": ["RectLight", "SphereLight"],
    },
    "warm_lab": {
        "color_temp_range": (4000.0, 6000.0),
        "intensity_range": (300.0, 1500.0),
        "exposure_range": (-0.5, 0.5),
        "light_types": ["SphereLight", "RectLight"],
    },
    "cool_lab": {
        "color_temp_range": (5000.0, 8500.0),
        "intensity_range": (800.0, 1500.0),
        "exposure_range": (0.0, 1.0),
        "light_types": ["RectLight", "DistantLight"],
    },
    "natural_daylight": {
        "color_temp_range": (5000.0, 6500.0),
        "intensity_range": (1000.0, 1500.0),
        "exposure_range": (1.0, 1.0),
        "light_types": ["DistantLight", "DomeLight"],
    },
}

# Supported USD light type names
LIGHT_TYPE_NAMES = {"SphereLight", "RectLight", "DistantLight", "DomeLight", "DiskLight", "CylinderLight"}


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a mapping or an OmegaConf-like object."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    return getter(key, default) if getter is not None else getattr(cfg, key, default)


def resolve_lighting_options(cfg: Any, split: str | None = None) -> dict[str, Any]:
    """Merge scenario defaults with explicit (optionally split-specific) values."""
    scenario = _cfg_value(cfg, "scenario")
    preset: dict[str, Any] = {}
    if scenario:
        if scenario in LAB_LIGHTING_SCENARIOS:
            preset = LAB_LIGHTING_SCENARIOS[str(scenario)]
        else:
            logger.warning(f"Unknown lighting scenario '{scenario}', using generic defaults")

    def merged(key: str, fallback: Any) -> Any:
        if split:
            value = _cfg_value(cfg, f"{split}_{key}")
            if value is not None:
                return value
        value = _cfg_value(cfg, key)
        return value if value is not None else preset.get(key, fallback)

    randomize_intensity = bool(_cfg_value(cfg, "randomize_intensity", True))
    configured_intensity_range = tuple(merged("intensity_range", (500.0, 5000.0)))
    color_temp_range = merged("color_temp_range", (2700.0, 6500.0))
    if isinstance(color_temp_range, str):
        if color_temp_range not in COLOR_TEMP_PRESETS:
            raise ValueError(f"Unknown color-temperature preset: {color_temp_range}")
        color_temp_range = COLOR_TEMP_PRESETS[color_temp_range]

    def axes(key: str, defaults: dict[str, tuple[float, float]]) -> dict | None:
        axis_cfg = merged(key, None)
        if axis_cfg is None:
            return None
        return {axis: tuple(_cfg_value(axis_cfg, axis, default)) for axis, default in defaults.items()}

    default_parent_path = "/World/VisualRandomization" if split else "/World"
    default_light_types = ["RectLight", "SphereLight"] if split else list(LIGHT_TYPE_NAMES)
    default_shared_exposure = not randomize_intensity if split else False

    return {
        "enabled": bool(_cfg_value(cfg, "enabled", False)),
        "scenario": str(scenario) if scenario else None,
        "num_lights": int(_cfg_value(cfg, "num_lights", 3)),
        "parent_path": str(_cfg_value(cfg, "parent_path", default_parent_path)),
        "light_types": list(merged("light_types", default_light_types)),
        "intensity_range": configured_intensity_range if randomize_intensity else None,
        "created_intensity_range": configured_intensity_range,
        "exposure_range": tuple(merged("exposure_range", (-2.0, 4.0))),
        "color_temp_range": tuple(color_temp_range),
        "randomize_position": bool(_cfg_value(cfg, "randomize_position", False)),
        "position_range": axes("position_range", {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (1.5, 3.0)}),
        "randomize_rotation": bool(_cfg_value(cfg, "randomize_rotation", False)),
        "rotation_range": axes("rotation_range", {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (0.0, 360.0)}),
        "shared_exposure": bool(_cfg_value(cfg, "shared_exposure", default_shared_exposure)),
        "exposure_jitter_range": tuple(_cfg_value(cfg, "exposure_jitter_range", (0.0, 0.0))),
        "shared_color_temperature": bool(_cfg_value(cfg, "shared_color_temperature", False)),
        "geometry_ranges": _cfg_value(cfg, "geometry_ranges", {}) or {},
    }


# ---------------------------------------------------------------------------
# Color temperature to RGB conversion
# ---------------------------------------------------------------------------


def color_temperature_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Convert color temperature (Kelvin) to linear RGB values in [0, 1].

    Uses Tanner Helland's approximation of the Planckian locus.

    Args:
        kelvin: Color temperature in Kelvin (1000 -- 40000).

    Returns:
        ``(r, g, b)`` tuple with values in [0, 1].
    """
    kelvin = np.clip(kelvin, 1000.0, 40000.0)
    temp = kelvin / 100.0

    # Red channel
    if temp <= 66.0:
        r = 255.0
    else:
        r = np.clip(329.698727446 * ((temp - 60.0) ** -0.1332047592), 0.0, 255.0)

    # Green channel
    if temp <= 66.0:
        g = np.clip(99.4708025861 * np.log(temp) - 161.1195681661, 0.0, 255.0)
    else:
        g = np.clip(288.1221695283 * ((temp - 60.0) ** -0.0755148492), 0.0, 255.0)

    # Blue channel
    if temp >= 66.0:
        b = 255.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = np.clip(138.5177312231 * np.log(temp - 10.0) - 305.0447927307, 0.0, 255.0)

    return (float(r / 255.0), float(g / 255.0), float(b / 255.0))


# ---------------------------------------------------------------------------
# LightingRandomizer
# ---------------------------------------------------------------------------


class LightingRandomizer:
    """Randomize scene lighting parameters for visual domain randomization.

    Discovers existing lights in the USD stage and provides methods to
    randomize their intensity, color, color temperature, exposure, and
    position.  Also supports creating new lights with randomized properties.

    Typical usage -- call :meth:`randomize_scene_for_episode` inside the
    task's ``reset()`` to get a fresh lighting configuration each episode::

        randomizer = LightingRandomizer(stage)
        randomizer.randomize_scene_for_episode(scenario="standard_lab")

    Or create a fully randomized light rig::

        randomizer.create_random_light_setup(num_lights=3, scenario="warm_lab")
    """

    def __init__(self, stage: Usd.Stage) -> None:
        self.stage = stage
        self._created_light_paths: List[str] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_scene_lights(self) -> List[str]:
        """Find all light prim paths in the current USD stage.

        Returns:
            List of USD prim paths for all lights found.
        """
        light_paths: List[str] = []
        for prim in self.stage.Traverse():
            if prim.GetTypeName() in LIGHT_TYPE_NAMES:
                light_paths.append(prim.GetPath().pathString)
        return light_paths

    # ------------------------------------------------------------------
    # Per-attribute randomization
    # ------------------------------------------------------------------

    def randomize_intensity(
        self,
        light_path: str,
        intensity_range: Tuple[float, float] = (500.0, 5000.0),
    ) -> float:
        """Randomize light intensity.

        Args:
            light_path: USD prim path of the light.
            intensity_range: ``(min, max)`` intensity range.

        Returns:
            The new intensity value.
        """
        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return 0.0

        intensity = random.uniform(*intensity_range)
        self._set_float_attr(prim, "inputs:intensity", intensity)
        return intensity

    def randomize_exposure(
        self,
        light_path: str,
        exposure_range: Tuple[float, float] = (-2.0, 4.0),
    ) -> float:
        """Randomize light exposure.

        Exposure is a power-of-2 multiplier on intensity:
        ``effective_intensity = intensity * 2^exposure``.

        Args:
            light_path: USD prim path of the light.
            exposure_range: ``(min, max)`` exposure values.

        Returns:
            The new exposure value.
        """
        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return 0.0

        exposure = random.uniform(*exposure_range)
        self._set_float_attr(prim, "inputs:exposure", exposure)
        return exposure

    def randomize_color_temperature(
        self,
        light_path: str,
        temp_range: Union[Tuple[float, float], str] = (2700.0, 6500.0),
    ) -> float:
        """Randomize light color temperature.

        Accepts a numeric ``(min_K, max_K)`` range or a named preset from
        :data:`COLOR_TEMP_PRESETS` (e.g. ``"warm_white"``, ``"daylight"``).
        Sets both the USD ``colorTemperature`` attribute and the ``color``
        attribute (via Planckian locus conversion) for broad renderer support.

        Args:
            light_path: USD prim path of the light.
            temp_range: ``(min_kelvin, max_kelvin)`` or a preset name.

        Returns:
            The sampled color temperature in Kelvin.
        """
        if isinstance(temp_range, str):
            if temp_range not in COLOR_TEMP_PRESETS:
                logger.warning(f"Unknown color-temp preset '{temp_range}', using default range")
                temp_range = (2700.0, 6500.0)
            else:
                temp_range = COLOR_TEMP_PRESETS[temp_range]

        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return 0.0

        kelvin = random.uniform(*temp_range)

        self._apply_color_temperature(prim, kelvin)

        return kelvin

    def randomize_color(
        self,
        light_path: str,
        hue_range: Tuple[float, float] = (0.0, 1.0),
        saturation_range: Tuple[float, float] = (0.0, 0.3),
        value_range: Tuple[float, float] = (0.8, 1.0),
    ) -> Tuple[float, float, float]:
        """Randomize light color via HSV space for perceptual control.

        Low saturation produces near-white light; higher values create tinted
        light.  This disables color temperature mode.

        Args:
            light_path: USD prim path of the light.
            hue_range: ``(min, max)`` hue in ``[0, 1]``.
            saturation_range: ``(min, max)`` saturation in ``[0, 1]``.
            value_range: ``(min, max)`` value/brightness in ``[0, 1]``.

        Returns:
            The ``(r, g, b)`` color set on the light.
        """
        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return (1.0, 1.0, 1.0)

        h = random.uniform(*hue_range)
        s = random.uniform(*saturation_range)
        v = random.uniform(*value_range)
        rgb = self._hsv_to_rgb(h, s, v)

        # Disable color temperature when using direct color
        enable_attr = prim.GetAttribute("inputs:enableColorTemperature")
        if enable_attr and enable_attr.IsValid():
            enable_attr.Set(False)

        self._set_color_attr(prim, "inputs:color", rgb)
        return rgb

    def randomize_position(
        self,
        light_path: str,
        position_range: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        """Randomize light position in world space.

        Args:
            light_path: USD prim path of the light.
            position_range: Dict with ``x``, ``y``, ``z`` keys, each a
                           ``(min, max)`` tuple.

        Returns:
            The new position as a ``(3,)`` array.
        """
        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return np.zeros(3)

        position = np.array(
            [
                random.uniform(*position_range.get("x", (0.0, 0.0))),
                random.uniform(*position_range.get("y", (0.0, 0.0))),
                random.uniform(*position_range.get("z", (0.0, 0.0))),
            ]
        )

        self._set_translate(prim, position)
        return position

    def randomize_rotation(
        self,
        light_path: str,
        rotation_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Randomize light rotation (Euler XYZ angles in degrees).

        Useful for DistantLight and RectLight to change shadow direction.

        Args:
            light_path: USD prim path of the light.
            rotation_range: Dict with ``x``, ``y``, ``z`` keys for rotation
                           ranges in degrees.  Defaults to moderate tilt.

        Returns:
            The new rotation as a ``(3,)`` array of Euler angles in degrees.
        """
        if rotation_range is None:
            rotation_range = {
                "x": (-30.0, 30.0),
                "y": (-30.0, 30.0),
                "z": (0.0, 360.0),
            }

        prim = self.stage.GetPrimAtPath(light_path)
        if not prim.IsValid():
            logger.warning(f"Light prim not found: {light_path}")
            return np.zeros(3)

        rotation = np.array(
            [
                random.uniform(*rotation_range.get("x", (0.0, 0.0))),
                random.uniform(*rotation_range.get("y", (0.0, 0.0))),
                random.uniform(*rotation_range.get("z", (0.0, 0.0))),
            ]
        )

        self._set_rotation(prim, rotation)
        return rotation

    # ------------------------------------------------------------------
    # Light creation
    # ------------------------------------------------------------------

    def create_light(
        self,
        light_type: str,
        path: str,
        intensity: float = 1000.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        position: Tuple[float, float, float] = (0.0, 0.0, 2.0),
        exposure: float = 0.0,
        color_temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Create a new light prim in the scene.

        Args:
            light_type: One of ``"SphereLight"``, ``"RectLight"``,
                        ``"DistantLight"``, ``"DomeLight"``, ``"DiskLight"``.
            path: USD prim path for the new light.
            intensity: Light intensity.
            color: RGB color as ``(r, g, b)`` in ``[0, 1]``.
            position: World position as ``(x, y, z)``.
            exposure: Exposure value (power-of-2 multiplier).
            color_temperature: Optional color temperature in Kelvin.  If set,
                              overrides *color* with temperature-derived RGB.
            **kwargs: Type-specific attributes: ``radius`` (SphereLight,
                      DiskLight), ``width`` / ``height`` (RectLight),
                      ``angle`` (DistantLight).

        Returns:
            The USD prim path of the created light, or ``""`` on failure.
        """
        if light_type not in LIGHT_TYPE_NAMES:
            logger.error(f"Unknown light type '{light_type}'. Use one of {LIGHT_TYPE_NAMES}")
            return ""

        prim = self.stage.DefinePrim(path, light_type)
        if not prim.IsValid():
            logger.error(f"Failed to create light at {path}")
            return ""

        # Core attributes
        prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))
        prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(float(exposure))

        # Color / color temperature
        if color_temperature is not None:
            prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(float(color_temperature))
            rgb = color_temperature_to_rgb(color_temperature)
        else:
            rgb = color
        prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))

        # Position
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))

        # Type-specific attributes
        if light_type == "SphereLight":
            radius = kwargs.get("radius", 0.5)
            prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(float(radius))
        elif light_type == "RectLight":
            prim.CreateAttribute("inputs:width", Sdf.ValueTypeNames.Float).Set(float(kwargs.get("width", 1.0)))
            prim.CreateAttribute("inputs:height", Sdf.ValueTypeNames.Float).Set(float(kwargs.get("height", 1.0)))
        elif light_type == "DiskLight":
            prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(float(kwargs.get("radius", 0.5)))
        elif light_type == "DistantLight":
            prim.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(float(kwargs.get("angle", 0.53)))

        self._created_light_paths.append(path)
        logger.info(f"Created {light_type} at {path} (intensity={intensity:.0f}, exposure={exposure:.1f})")
        return path

    def create_random_light(
        self,
        parent_path: str = "/World",
        light_type: Optional[str] = None,
        intensity_range: Tuple[float, float] = (500.0, 5000.0),
        exposure_range: Tuple[float, float] = (-1.0, 3.0),
        color_temp_range: Tuple[float, float] = (2700.0, 6500.0),
        position_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> str:
        """Create a new light with fully randomized parameters.

        Args:
            parent_path: Parent USD path under which to create the light.
            light_type: Light type to create. If ``None``, randomly chosen.
            intensity_range: ``(min, max)`` for intensity.
            exposure_range: ``(min, max)`` for exposure.
            color_temp_range: ``(min, max)`` for color temperature in Kelvin.
            position_range: Position randomization ranges per axis.  Defaults
                           to lab-scale ranges above the workspace.

        Returns:
            The USD prim path of the created light.
        """
        if light_type is None:
            light_type = random.choice(list(LIGHT_TYPE_NAMES))

        if position_range is None:
            position_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (1.5, 3.0)}

        idx = len(self._created_light_paths)
        path = f"{parent_path}/RandomLight_{light_type}_{idx}"

        intensity = random.uniform(*intensity_range)
        exposure = random.uniform(*exposure_range)
        kelvin = random.uniform(*color_temp_range)
        position = (
            random.uniform(*position_range["x"]),
            random.uniform(*position_range["y"]),
            random.uniform(*position_range["z"]),
        )

        # Type-specific random geometry
        kwargs: Dict = {}
        if light_type == "SphereLight":
            kwargs["radius"] = random.uniform(0.1, 1.0)
        elif light_type == "RectLight":
            kwargs["width"] = random.uniform(0.3, 2.0)
            kwargs["height"] = random.uniform(0.3, 2.0)
        elif light_type == "DiskLight":
            kwargs["radius"] = random.uniform(0.2, 1.5)
        elif light_type == "DistantLight":
            kwargs["angle"] = random.uniform(0.2, 2.0)

        return self.create_light(
            light_type=light_type,
            path=path,
            intensity=intensity,
            exposure=exposure,
            color_temperature=kelvin,
            position=position,
            **kwargs,
        )

    def create_random_light_setup(
        self,
        num_lights: int = 3,
        parent_path: str = "/World",
        scenario: Optional[str] = None,
        position_range: Optional[Dict[str, Tuple[float, float]]] = None,
        intensity_range: Optional[Tuple[float, float]] = None,
        exposure_range: Optional[Tuple[float, float]] = None,
        color_temp_range: Optional[Tuple[float, float]] = None,
        light_types: Optional[List[str]] = None,
    ) -> List[str]:
        """Create a complete randomized lighting rig with multiple lights.

        Removes previously created lights first, then creates ``num_lights``
        new ones.  Optionally uses a named lab scenario for realistic ranges.

        Args:
            num_lights: Number of lights to create (1 -- 6).
            parent_path: Parent USD path.
            scenario: Optional scenario name from :data:`LAB_LIGHTING_SCENARIOS`.
            position_range: Override position randomization ranges.

        Returns:
            List of created light prim paths.
        """
        self.remove_created_lights()

        scenario_cfg = LAB_LIGHTING_SCENARIOS.get(scenario or "", {})
        intensity_range = intensity_range or scenario_cfg.get("intensity_range", (500.0, 5000.0))
        color_temp_range = color_temp_range or scenario_cfg.get("color_temp_range", (2700.0, 6500.0))
        allowed_types = light_types or scenario_cfg.get("light_types", list(LIGHT_TYPE_NAMES))
        exposure_range = exposure_range or scenario_cfg.get("exposure_range", (-1.0, 3.0))

        paths: List[str] = []
        for _ in range(num_lights):
            light_type = random.choice(allowed_types)
            path = self.create_random_light(
                parent_path=parent_path,
                light_type=light_type,
                intensity_range=intensity_range,
                exposure_range=exposure_range,
                color_temp_range=color_temp_range,
                position_range=position_range,
            )
            if path:
                paths.append(path)

        logger.info(
            f"Created lighting setup with {len(paths)} lights" + (f" (scenario: {scenario})" if scenario else "")
        )
        return paths

    # ------------------------------------------------------------------
    # Serializable layout application
    # ------------------------------------------------------------------

    def apply_background_layout(self, layout: Mapping[str, Any]) -> bool:
        """Create or update the DomeLight described by a visual layout."""
        path = str(layout["path"])
        prim = self.stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.GetTypeName() != "DomeLight":
            logger.error(f"Visual background path {path} is not a DomeLight")
            return False
        if not prim.IsValid():
            prim = self.stage.DefinePrim(path, "DomeLight")

        if layout.get("texture_file"):
            self._ensure_dome_texture_subframes()
        self._set_float_attr(prim, "inputs:intensity", float(layout["intensity"]))
        self._set_float_attr(prim, "inputs:exposure", float(layout["exposure"]))
        self._set_bool_attr(prim, "inputs:visibleInPrimaryRay", bool(layout.get("visible_in_primary_ray", True)))
        self._set_color_attr(prim, "inputs:color", tuple(layout.get("color", (1.0, 1.0, 1.0))))
        texture_attr = prim.GetAttribute("inputs:texture:file")
        if not texture_attr or not texture_attr.IsValid():
            texture_attr = prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
        texture_file = os.path.expanduser(str(layout.get("texture_file", "")))
        if texture_file and "://" not in texture_file and not os.path.isabs(texture_file):
            texture_file = os.path.abspath(texture_file)
        texture_attr.Set(Sdf.AssetPath(texture_file))
        format_attr = prim.GetAttribute("inputs:texture:format")
        if not format_attr or not format_attr.IsValid():
            format_attr = prim.CreateAttribute("inputs:texture:format", Sdf.ValueTypeNames.Token)
        format_attr.Set(str(layout.get("texture_format", "latlong")))
        self._set_rotation(prim, np.asarray(layout.get("rotation", (0.0, 0.0, 0.0))))
        return True

    @staticmethod
    def _ensure_dome_texture_subframes() -> None:
        """Avoid blank dynamically changed Dome textures in RaytracedLighting."""
        try:
            # Delayed deliberately: this utility is unit-tested without starting
            # Isaac Sim, where the Carbonite module is not importable.
            import carb
        except ImportError:
            return
        settings = carb.settings.get_settings()
        if settings.get("/rtx/rendermode") != "RaytracedLighting":
            return
        subframes = settings.get("/omni/replicator/RTSubframes")
        if subframes is None or subframes < 3:
            settings.set("/omni/replicator/RTSubframes", 3)
            logger.warning("Raised /omni/replicator/RTSubframes to 3 for Dome HDR randomization")

    def apply_light_layout(self, layout: Mapping[str, Any]) -> dict[str, dict]:
        """Apply an absolute, JSON-serializable light layout to the stage."""
        applied: dict[str, dict] = {}
        for light in layout.get("lights", []):
            path = str(light["path"])
            requested_type = light.get("type")
            prim = self.stage.GetPrimAtPath(path)
            if requested_type and prim.IsValid() and prim.GetTypeName() != requested_type:
                self.stage.RemovePrim(path)
                prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() and requested_type:
                prim = self.stage.DefinePrim(path, str(requested_type))
                self._created_light_paths.append(path)
            if not prim.IsValid():
                logger.warning(f"Light prim from visual layout not found: {path}")
                continue

            if "intensity" in light:
                self._set_float_attr(prim, "inputs:intensity", float(light["intensity"]))
            if "exposure" in light:
                self._set_float_attr(prim, "inputs:exposure", float(light["exposure"]))
            if "color_temperature" in light:
                self._apply_color_temperature(prim, float(light["color_temperature"]))
            if "position" in light:
                self._set_translate(prim, np.asarray(light["position"], dtype=float))
            if "rotation" in light:
                self._set_rotation(prim, np.asarray(light["rotation"], dtype=float))
            for name, value in (light.get("geometry") or {}).items():
                self._set_float_attr(prim, f"inputs:{name}", float(value))
            applied[path] = {key: value for key, value in light.items() if key != "path"}
        return applied

    # ------------------------------------------------------------------
    # Batch randomization of existing lights
    # ------------------------------------------------------------------

    def randomize_all(
        self,
        intensity_range: Optional[Tuple[float, float]] = (500.0, 5000.0),
        exposure_range: Tuple[float, float] = (-2.0, 4.0),
        color_temp_range: Union[Tuple[float, float], str] = (2700.0, 6500.0),
        position_range: Optional[Dict[str, Tuple[float, float]]] = None,
        randomize_position_flag: bool = False,
        rotation_range: Optional[Dict[str, Tuple[float, float]]] = None,
        randomize_rotation_flag: bool = False,
        shared_exposure: bool = False,
        exposure_jitter_range: Tuple[float, float] = (0.0, 0.0),
        shared_color_temperature: bool = False,
    ) -> Dict[str, Dict]:
        """Randomize all properties of every light currently in the scene.

        Args:
            intensity_range: ``(min, max)`` for intensity.
            exposure_range: ``(min, max)`` for exposure.
            color_temp_range: ``(min_K, max_K)`` or a preset name.
            position_range: Per-axis position ranges (used only when
                           *randomize_position_flag* is ``True``).
            randomize_position_flag: Whether to also randomize positions.

        Returns:
            Dict mapping light path to a dict of new parameter values.
        """
        light_paths = self.find_scene_lights()
        if not light_paths:
            logger.warning("No lights found in scene to randomize")
            return {}

        if isinstance(color_temp_range, str):
            color_temp_range = COLOR_TEMP_PRESETS.get(color_temp_range, (2700.0, 6500.0))
        common_exposure = random.uniform(*exposure_range) if shared_exposure else None
        common_temperature = random.uniform(*color_temp_range) if shared_color_temperature else None
        lights: list[dict[str, Any]] = []
        for path in sorted(light_paths):
            result: dict[str, Any] = {"path": path}
            # None preserves authored intensities. A shared exposure then scales
            # a calibrated rig without changing relative light contributions.
            if intensity_range is not None:
                result["intensity"] = random.uniform(*intensity_range)
            exposure = common_exposure if common_exposure is not None else random.uniform(*exposure_range)
            result["exposure"] = exposure + random.uniform(*exposure_jitter_range)
            result["color_temperature"] = (
                common_temperature if common_temperature is not None else random.uniform(*color_temp_range)
            )
            if randomize_position_flag and position_range:
                result["position"] = [random.uniform(*position_range.get(axis, (0.0, 0.0))) for axis in ("x", "y", "z")]
            if randomize_rotation_flag:
                ranges = rotation_range or {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (0.0, 360.0)}
                result["rotation"] = [random.uniform(*ranges.get(axis, (0.0, 0.0))) for axis in ("x", "y", "z")]
            lights.append(result)

        results = self.apply_light_layout({"lights": lights})

        logger.info(f"Randomized {len(results)} scene lights")
        return results

    def randomize_scene_for_episode(
        self,
        scenario: Optional[str] = None,
        position_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict]:
        """Per-episode lighting randomization (call from ``task.reset()``).

        Randomizes existing scene lights using a named lab scenario or
        default ranges.

        Args:
            scenario: Lab scenario name (e.g. ``"standard_lab"``,
                      ``"warm_lab"``).  See :data:`LAB_LIGHTING_SCENARIOS`.
            position_range: Optional position randomization ranges.

        Returns:
            Dict mapping light path to new parameter values.
        """
        if scenario and scenario in LAB_LIGHTING_SCENARIOS:
            cfg = LAB_LIGHTING_SCENARIOS[scenario]
            return self.randomize_all(
                intensity_range=cfg["intensity_range"],
                exposure_range=cfg["exposure_range"],
                color_temp_range=cfg["color_temp_range"],
                position_range=position_range,
                randomize_position_flag=position_range is not None,
            )
        return self.randomize_all(
            position_range=position_range,
            randomize_position_flag=position_range is not None,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_created_lights(self) -> None:
        """Remove all lights that were created by this randomizer instance."""
        for path in self._created_light_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                self.stage.RemovePrim(path)
        removed = len(self._created_light_paths)
        self._created_light_paths.clear()
        if removed > 0:
            logger.info(f"Removed {removed} created lights")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_float_attr(self, prim: Usd.Prim, name: str, value: float) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))

    def _set_bool_attr(self, prim: Usd.Prim, name: str, value: bool) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool)
        attr.Set(bool(value))

    def _set_color_attr(self, prim: Usd.Prim, name: str, rgb: Tuple[float, float, float]) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Color3f)
        attr.Set(Gf.Vec3f(*rgb))

    def _apply_color_temperature(self, prim: Usd.Prim, kelvin: float) -> None:
        """Set one sampled temperature without applying its tint twice."""
        self._set_bool_attr(prim, "inputs:enableColorTemperature", True)
        self._set_float_attr(prim, "inputs:colorTemperature", kelvin)
        # RTX already evaluates colorTemperature. A Planckian inputs:color
        # would apply the same tint twice and darken both endpoints.
        self._set_color_attr(prim, "inputs:color", (1.0, 1.0, 1.0))

    def _set_translate(self, prim: Usd.Prim, position: np.ndarray) -> None:
        """Set or update the translate xform op on *prim*."""
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*[float(v) for v in position]))
                return
        xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))

    def _set_rotation(self, prim: Usd.Prim, euler_xyz_deg: np.ndarray) -> None:
        """Set or update the rotateXYZ xform op on *prim*."""
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                op.Set(Gf.Vec3f(*[float(v) for v in euler_xyz_deg]))
                return
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in euler_xyz_deg]))

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB (all values in ``[0, 1]``)."""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)
