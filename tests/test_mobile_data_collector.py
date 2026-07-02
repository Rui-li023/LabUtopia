import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_collectors.mobile_data_collector import MobileDataCollector


def _make_collector(tmp_path):
    return MobileDataCollector(
        camera_configs=[{"name": "front_camera", "image_type": "rgb"}],
        save_dir=str(tmp_path),
        max_episodes=2,
        max_workers=1,
    )


def _state(finger=0.03):
    s = np.arange(11, dtype=np.float32) / 100.0
    s[10] = finger
    return s


def test_phase_and_11dim_layout(tmp_path):
    c = _make_collector(tmp_path)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for phase in (0, 0, 1):
        c.cache_step(
            {"front_camera": img},
            _state(),
            action=np.arange(11, dtype=np.float32) / 100.0,
            language_instruction="pick up the beaker",
            phase=phase,
        )
    c.write_cached_data()
    c.close()
    h5_path = os.path.join(str(tmp_path), "dataset", "episode_0000", "episode_0000.h5")
    with h5py.File(h5_path, "r") as f:
        assert f["actions"].shape == (3, 11)
        assert f["agent_pose"].shape == (3, 11)
        assert list(f["phase"][:]) == [0, 0, 1]
        # gripper width doubling happens at index 10, NOT index 7
        assert f["agent_pose"][0][10] == pytest.approx(0.06)
        assert f["agent_pose"][0][7] == pytest.approx(0.07)


def test_write_requires_action_every_step(tmp_path):
    c = _make_collector(tmp_path)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    c.cache_step({"front_camera": img}, _state(), action=None, phase=0)
    with pytest.raises(ValueError):
        c.write_cached_data()
    c.close()


def test_clear_cache_resets_phases(tmp_path):
    c = _make_collector(tmp_path)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    c.cache_step({"front_camera": img}, _state(),
                 action=np.zeros(11, dtype=np.float32), phase=3)
    c.clear_cache()
    assert c.temp_phases == []
    c.close()


def test_default_collector_unchanged(tmp_path):
    """Regression: the base DataCollector still writes 8-dim episodes without phase."""
    from data_collectors.data_collector import DataCollector
    c = DataCollector(
        camera_configs=[{"name": "front_camera", "image_type": "rgb"}],
        save_dir=str(tmp_path), max_episodes=2, max_workers=1,
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(2):
        c.cache_step({"front_camera": img},
                     np.arange(8, dtype=np.float32) / 100.0,
                     action=np.arange(8, dtype=np.float32) / 100.0,
                     language_instruction="x")
    c.write_cached_data()
    c.close()
    h5_path = os.path.join(str(tmp_path), "dataset", "episode_0000", "episode_0000.h5")
    with h5py.File(h5_path, "r") as f:
        assert f["actions"].shape == (2, 8)
        assert "phase" not in f
        assert f["agent_pose"][0][7] == pytest.approx(0.14)  # legacy index-7 doubling
