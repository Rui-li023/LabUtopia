"""Physical candidates compose without changing the baseline task contract."""

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.gripper_contact import PipetteContactParameters


class PipettePhysicsConfigTests(unittest.TestCase):
    def test_single_actuator_candidate_changes_only_force_and_explicit_semantics(self):
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            for name in ("level2_pipette_to_rack", "level4_four_pipettes_to_rack"):
                with self.subTest(name=name):
                    baseline = compose(
                        config_name=name,
                        overrides=["+physics=pipette_insertion_tracking_candidate", "++physics_substeps=4"],
                    )
                    candidate = compose(config_name=name, overrides=["+physics=pipette_single_actuator_50n_candidate"])
                    parameters = PipetteContactParameters.from_config(candidate)
                    self.assertEqual(parameters.closing_force_drive_cap, 50)
                    self.assertEqual(parameters.force_semantics, "single_mimic_drive")
                    direct = compose(config_name="physics/" + name + "_50n")
                    self.assertEqual(OmegaConf.to_container(direct), OmegaConf.to_container(candidate))
                    candidate.robot.gripper.closing_force = baseline.robot.gripper.closing_force
                    del candidate.robot.gripper.force_semantics
                    self.assertEqual(OmegaConf.to_container(candidate), OmegaConf.to_container(baseline))

    def test_insertion_tracking_candidate_preserves_all_physics(self):
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            for name in ("level2_pipette_to_rack", "level4_four_pipettes_to_rack"):
                with self.subTest(name=name):
                    baseline = compose(config_name=name)
                    candidate = compose(config_name=name, overrides=["+physics=pipette_insertion_tracking_candidate"])
                    self.assertTrue(candidate.task.pipette_track_insertion_axis)
                    del candidate.task.pipette_track_insertion_axis
                    self.assertEqual(OmegaConf.to_container(candidate), OmegaConf.to_container(baseline))

    def test_compliance_candidate_changes_only_pad_spring_and_damper(self):
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            for name in ("level2_pipette_to_rack", "level4_four_pipettes_to_rack"):
                with self.subTest(name=name):
                    baseline = compose(config_name=name)
                    candidate = compose(config_name=name, overrides=["+physics=pipette_grip_compliant_candidate"])
                    parameters = PipetteContactParameters.from_config(candidate)
                    self.assertEqual(parameters.contact_stiffness, 100000)
                    self.assertEqual(parameters.contact_damping, 100)
                    for key in ("gripper_pad_contact_stiffness", "gripper_pad_contact_damping"):
                        candidate.task[key] = baseline.task[key]
                    self.assertEqual(OmegaConf.to_container(candidate), OmegaConf.to_container(baseline))

    def test_open80_candidate_changes_only_opening_width(self):
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            for name in ("level2_pipette_to_rack", "level4_four_pipettes_to_rack"):
                with self.subTest(name=name):
                    baseline = compose(config_name=name)
                    candidate = compose(config_name=name, overrides=["+physics=pipette_grip_open80_candidate"])
                    self.assertEqual(baseline.task.gripper_opening_width, 0.040)
                    self.assertEqual(candidate.task.gripper_opening_width, 0.080)
                    candidate.task.gripper_opening_width = baseline.task.gripper_opening_width
                    self.assertEqual(OmegaConf.to_container(candidate), OmegaConf.to_container(baseline))

    def test_candidate_is_runnable_and_preserves_scene_and_success_gates(self):
        root = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            for name in ("level2_pipette_to_rack", "level4_four_pipettes_to_rack"):
                with self.subTest(name=name):
                    baseline = compose(config_name=name)
                    candidate = compose(config_name="physics/" + name)
                    grouped = compose(config_name=name, overrides=["+physics=pipette_grip_candidate"])
                    self.assertNotIn("physics", candidate)
                    for key in ("task_type", "controller_type", "usd_path", "physics_substeps"):
                        self.assertEqual(candidate[key], baseline[key])
                    for key in (
                        "obj_paths",
                        "max_grasp_rotation_deg",
                        "placement_x_tolerance",
                        "placement_y_tolerance",
                        "placement_z_tolerance",
                        "settle_timeout_steps",
                    ):
                        self.assertEqual(candidate.task[key], baseline.task[key])
                    for key in ("task", "robot"):
                        self.assertEqual(OmegaConf.to_container(candidate[key]), OmegaConf.to_container(grouped[key]))
                        self.assertEqual(OmegaConf.to_container(candidate[key]), OmegaConf.to_container(baseline[key]))
                    self.assertEqual(candidate.task.rack_contact.static_friction, 0.3)
                    self.assertEqual(candidate.task.rack_contact.dynamic_friction, 0.3)
                    self.assertNotIn("friction", candidate.task.obj_paths[1])
                    self.assertTrue(candidate.task.pipette_object_aligned_grasp)
                    self.assertFalse(candidate.task.pipette_require_pick_orientation)
                    self.assertTrue(candidate.task.pipette_enforce_runtime_joint_margin)
                    parameters = PipetteContactParameters.from_config(candidate)
                    self.assertEqual(parameters.closing_force_per_finger, 35)
                    self.assertEqual(parameters.closing_speed, 0.02)
                    self.assertEqual(parameters.opening_speed, 0.05)
                    if name.startswith("level4"):
                        self.assertEqual(candidate.task.slot_indices, [0, 1, 2, 3])
                        self.assertEqual(candidate.task.pipette_paths, baseline.task.pipette_paths)
