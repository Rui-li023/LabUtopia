"""Pure sampling helpers for reproducible visual-domain layouts.

The functions in this module intentionally do not import Isaac Sim or USD.  A
layout can therefore be sampled and tested without starting the simulator, then
applied by :class:`utils.lighting_utils.LightingRandomizer` and ``BaseTask``.
"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any


def derive_layout_seed(base_seed: int, episode_index: int, split: str) -> int:
    """Derive a stable per-episode seed independent of process RNG state."""
    payload = f"{int(base_seed)}:{split}:{int(episode_index)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _range(value: Any, default: Sequence[float]) -> tuple[float, float]:
    values = list(default if value is None else value)
    if len(values) != 2:
        raise ValueError(f"Expected a two-value range, got {values!r}")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise ValueError(f"Range minimum {low} exceeds maximum {high}")
    return low, high


def _axis_ranges(value: Any, default: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, float]]:
    value = value or {}
    return {
        axis: _range(value.get(axis) if isinstance(value, Mapping) else None, axis_default)
        for axis, axis_default in default.items()
    }


def _candidates(section: Mapping[str, Any], stem: str, split: str) -> list[Any]:
    split_values = section.get(f"{split}_{stem}")
    if split_values is not None:
        return list(split_values)
    values = section.get(stem)
    if values is not None:
        return list(values)
    return []


def _sample_background(section: Mapping[str, Any], split: str, rng: random.Random) -> dict[str, Any] | None:
    if not bool(section.get("enabled", False)):
        return None

    hdrs = _candidates(section, "hdrs", split)
    colors = _candidates(section, "colors", split)
    rotation_cfg = section.get("rotation_range", [0.0, 360.0])
    if isinstance(rotation_cfg, Mapping):
        rotation_ranges = _axis_ranges(
            rotation_cfg,
            {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 360.0)},
        )
    else:
        rotation_ranges = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": _range(rotation_cfg, (0.0, 360.0)),
        }

    background: dict[str, Any] = {
        "path": str(section.get("dome_light_path", "/World/VisualRandomization/Background")),
        "texture_file": str(rng.choice(hdrs)) if hdrs else "",
        "texture_format": str(section.get("texture_format", "latlong")),
        "visible_in_primary_ray": bool(section.get("visible_in_primary_ray", True)),
        "intensity": rng.uniform(*_range(section.get("intensity_range"), (500.0, 1000.0))),
        "exposure": rng.uniform(*_range(section.get("exposure_range"), (-0.5, 0.5))),
        "rotation": [rng.uniform(*rotation_ranges[axis]) for axis in ("x", "y", "z")],
        "color": [1.0, 1.0, 1.0],
    }
    if colors:
        color = list(rng.choice(colors))
        if len(color) != 3:
            raise ValueError(f"Background color must contain three values, got {color!r}")
        background["color"] = [float(channel) for channel in color]
    return background


def _sample_surfaces(sections: Sequence[Mapping[str, Any]], split: str, rng: random.Random) -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    for section in sections:
        if not bool(section.get("enabled", True)):
            continue
        materials = _candidates(section, "materials", split)
        if not materials:
            continue
        paths_value = section.get("paths")
        if paths_value is not None:
            paths = [str(path) for path in paths_value]
        elif section.get("path") is not None:
            paths = [str(section["path"])]
        else:
            paths = []
        if paths:
            layouts.append({"paths": paths, "material": str(rng.choice(materials))})
    return layouts


def _sample_geometry(light_type: str, options: Mapping[str, Any], rng: random.Random) -> dict[str, float]:
    ranges = options.get("geometry_ranges") or {}
    if light_type == "RectLight":
        return {
            "width": rng.uniform(*_range(ranges.get("width"), (0.3, 2.0))),
            "height": rng.uniform(*_range(ranges.get("height"), (0.3, 2.0))),
        }
    if light_type == "CylinderLight":
        return {
            "radius": rng.uniform(*_range(ranges.get("radius"), (0.1, 1.0))),
            "length": rng.uniform(*_range(ranges.get("length"), (0.3, 2.0))),
        }
    if light_type in {"SphereLight", "DiskLight"}:
        defaults = (0.1, 1.0) if light_type != "DiskLight" else (0.2, 1.5)
        return {"radius": rng.uniform(*_range(ranges.get("radius"), defaults))}
    if light_type == "DistantLight":
        return {"angle": rng.uniform(*_range(ranges.get("angle"), (0.2, 2.0)))}
    return {}


def _sample_lighting(
    light_paths: Sequence[str], options: Mapping[str, Any], rng: random.Random
) -> dict[str, Any] | None:
    if not bool(options.get("enabled", False)):
        return None

    intensity_range = options.get("intensity_range")
    exposure_range = _range(options.get("exposure_range"), (-2.0, 4.0))
    temperature_range = _range(options.get("color_temp_range"), (2700.0, 6500.0))
    exposure_jitter_range = _range(options.get("exposure_jitter_range"), (0.0, 0.0))
    shared_exposure = rng.uniform(*exposure_range) if options.get("shared_exposure", False) else None
    shared_temperature = rng.uniform(*temperature_range) if options.get("shared_color_temperature", False) else None

    paths = sorted(str(path) for path in light_paths)
    create_lights = not paths
    if create_lights:
        parent = str(options.get("parent_path", "/World/VisualRandomization"))
        paths = [f"{parent}/Light_{index}" for index in range(int(options.get("num_lights", 3)))]

    allowed_types = list(options.get("light_types") or ["RectLight", "SphereLight"])
    position_ranges = _axis_ranges(
        options.get("position_range"),
        {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (1.5, 3.0)},
    )
    rotation_ranges = _axis_ranges(
        options.get("rotation_range"),
        {"x": (-30.0, 30.0), "y": (-30.0, 30.0), "z": (0.0, 360.0)},
    )

    lights: list[dict[str, Any]] = []
    for path in paths:
        exposure = shared_exposure
        if exposure is None:
            exposure = rng.uniform(*exposure_range)
        exposure += rng.uniform(*exposure_jitter_range)
        temperature = shared_temperature
        if temperature is None:
            temperature = rng.uniform(*temperature_range)

        item: dict[str, Any] = {
            "path": path,
            "exposure": exposure,
            "color_temperature": temperature,
        }
        if intensity_range is not None:
            item["intensity"] = rng.uniform(*_range(intensity_range, (500.0, 5000.0)))
        if options.get("randomize_position", False) or create_lights:
            item["position"] = [rng.uniform(*position_ranges[axis]) for axis in ("x", "y", "z")]
        if options.get("randomize_rotation", False) or create_lights:
            item["rotation"] = [rng.uniform(*rotation_ranges[axis]) for axis in ("x", "y", "z")]
        if create_lights:
            item["type"] = str(rng.choice(allowed_types))
            item["intensity"] = item.get(
                "intensity",
                rng.uniform(*_range(options.get("created_intensity_range"), (500.0, 5000.0))),
            )
            item["geometry"] = _sample_geometry(item["type"], options, rng)
        lights.append(item)

    return {
        "shared_exposure": shared_exposure,
        "shared_color_temperature": shared_temperature,
        "lights": lights,
    }


def sample_visual_layout(
    config: Mapping[str, Any],
    episode_index: int,
    light_paths: Sequence[str],
    lighting_options: Mapping[str, Any],
    default_seed: int = 0,
) -> dict[str, Any]:
    """Sample one JSON-serializable background/surface/light layout."""
    split = str(config.get("split", "train"))
    if split not in {"train", "test"}:
        raise ValueError(f"visual_randomization.split must be 'train' or 'test', got {split!r}")
    base_seed = int(config.get("seed", default_seed))
    layout_seed = derive_layout_seed(base_seed, episode_index, split)
    rng = random.Random(layout_seed)

    background = _sample_background(config.get("background") or {}, split, rng)
    surfaces = _sample_surfaces(config.get("surfaces") or [], split, rng)
    lighting = _sample_lighting(light_paths, lighting_options, rng)
    return {
        "version": 1,
        "layout_id": f"{split}-{episode_index:06d}-{layout_seed:016x}",
        "base_seed": base_seed,
        "layout_seed": layout_seed,
        "episode_index": int(episode_index),
        "split": split,
        "background": background,
        "surfaces": surfaces,
        "lighting": lighting,
    }
