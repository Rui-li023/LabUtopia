import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pxr import Sdf, Usd

from data_collectors.data_collector import _flatten_init_state
from utils.lighting_utils import LightingRandomizer, resolve_lighting_options
from utils.visual_randomization import sample_visual_layout

VISUAL_CONFIG = {
    "seed": 17,
    "split": "test",
    "background": {
        "enabled": True,
        "train_hdrs": ["train_a.hdr"],
        "test_hdrs": ["test_a.hdr"],
        "train_colors": [[1.0, 1.0, 1.0]],
        "test_colors": [[0.7, 0.8, 1.0]],
        "intensity_range": [500.0, 1000.0],
    },
    "surfaces": [
        {
            "path": "/World/Table",
            "train_materials": ["/World/Looks/Train"],
            "test_materials": ["/World/Looks/Test"],
        }
    ],
}


def test_layout_is_deterministic_and_json_serializable():
    lighting = {
        "enabled": True,
        "intensity_range": None,
        "exposure_range": [-1.0, 1.0],
        "color_temp_range": [4000.0, 8000.0],
        "shared_exposure": True,
        "shared_color_temperature": True,
    }
    first = sample_visual_layout(
        VISUAL_CONFIG,
        episode_index=3,
        light_paths=["/World/Lights/B", "/World/Lights/A"],
        lighting_options=lighting,
    )
    repeated = sample_visual_layout(
        VISUAL_CONFIG,
        episode_index=3,
        light_paths=["/World/Lights/A", "/World/Lights/B"],
        lighting_options=lighting,
    )

    assert first == repeated
    assert first["background"]["texture_file"] == "test_a.hdr"
    assert first["surfaces"] == [{"paths": ["/World/Table"], "material": "/World/Looks/Test"}]
    assert len({light["exposure"] for light in first["lighting"]["lights"]}) == 1
    assert len({light["color_temperature"] for light in first["lighting"]["lights"]}) == 1
    json.dumps(first)


def test_episode_and_split_change_layout_identity():
    lighting = {"enabled": False}
    episode_zero = sample_visual_layout(VISUAL_CONFIG, 0, [], lighting)
    episode_one = sample_visual_layout(VISUAL_CONFIG, 1, [], lighting)
    train_config = {**VISUAL_CONFIG, "split": "train"}
    train = sample_visual_layout(train_config, 0, [], lighting)

    assert episode_zero["layout_id"] != episode_one["layout_id"]
    assert episode_zero["layout_id"] != train["layout_id"]
    assert train["background"]["texture_file"] == "train_a.hdr"
    assert train["surfaces"][0]["material"] == "/World/Looks/Train"


def test_level3_profile_uses_disjoint_existing_backgrounds():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo_root / "config/level3_pick_visual.yaml")
    background = cfg.visual_randomization.background
    train_hdrs = set(background.train_hdrs)
    test_hdrs = set(background.test_hdrs)

    assert train_hdrs
    assert test_hdrs
    assert train_hdrs.isdisjoint(test_hdrs)
    for relative_path in train_hdrs | test_hdrs:
        assert (repo_root / relative_path).is_file()


def test_layout_uses_existing_init_state_json_channel():
    layout = sample_visual_layout(VISUAL_CONFIG, 0, [], {"enabled": False})
    flattened = _flatten_init_state({"object_poses": {}, "object_materials": {}, "extra": {"visual_layout": layout}})

    restored = json.loads(flattened["init_extra_json"])
    assert restored["visual_layout"] == layout


def test_scenario_defaults_are_overridden_and_split_aware():
    options = resolve_lighting_options(
        {
            "enabled": True,
            "scenario": "standard_lab",
            "randomize_intensity": False,
            "exposure_range": [-0.5, 0.5],
            "train_color_temp_range": [4500.0, 7500.0],
            "test_color_temp_range": [3000.0, 4000.0],
        },
        split="test",
    )

    assert options["intensity_range"] is None
    assert options["created_intensity_range"] == (500.0, 2000.0)
    assert options["exposure_range"] == (-0.5, 0.5)
    assert options["color_temp_range"] == (3000.0, 4000.0)
    assert options["shared_exposure"] is True


def test_empty_scene_layout_creates_replayable_typed_lights():
    lighting = {
        "enabled": True,
        "num_lights": 2,
        "parent_path": "/World/Generated",
        "light_types": ["CylinderLight"],
        "intensity_range": None,
        "created_intensity_range": [800.0, 800.0],
        "exposure_range": [0.0, 0.0],
        "color_temp_range": [5000.0, 5000.0],
    }
    layout = sample_visual_layout(VISUAL_CONFIG, 0, [], lighting)
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")

    LightingRandomizer(stage).apply_light_layout(layout["lighting"])

    for index in range(2):
        prim = stage.GetPrimAtPath(f"/World/Generated/Light_{index}")
        assert prim.GetTypeName() == "CylinderLight"
        assert prim.GetAttribute("inputs:intensity").Get() == pytest.approx(800.0)
        assert prim.GetAttribute("inputs:length").HasAuthoredValueOpinion()


def test_sampled_layout_can_be_applied_and_restored_on_usd_stage():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Key", "RectLight")
    randomizer = LightingRandomizer(stage)
    background = {
        "path": "/World/Background",
        "texture_file": "studio.hdr",
        "texture_format": "latlong",
        "intensity": 750.0,
        "exposure": 0.25,
        "rotation": [0.0, 0.0, 135.0],
        "color": [0.8, 0.9, 1.0],
    }
    lighting = {
        "lights": [
            {
                "path": "/World/Key",
                "intensity": 1234.0,
                "exposure": -0.25,
                "color_temperature": 5200.0,
                "rotation": [10.0, 20.0, 30.0],
                "geometry": {"width": 1.5, "height": 0.5},
            }
        ]
    }

    assert randomizer.apply_background_layout(background)
    randomizer.apply_light_layout(lighting)
    dome = stage.GetPrimAtPath("/World/Background")
    key = stage.GetPrimAtPath("/World/Key")
    assert dome.GetAttribute("inputs:texture:file").Get() == Sdf.AssetPath(os.path.abspath("studio.hdr"))
    assert dome.GetAttribute("inputs:visibleInPrimaryRay").Get() is True
    assert dome.GetAttribute("inputs:intensity").Get() == pytest.approx(750.0)
    assert key.GetAttribute("inputs:intensity").Get() == pytest.approx(1234.0)
    assert key.GetAttribute("inputs:exposure").Get() == pytest.approx(-0.25)
    assert key.GetAttribute("inputs:colorTemperature").Get() == pytest.approx(5200.0)
    assert key.GetAttribute("inputs:width").Get() == pytest.approx(1.5)
