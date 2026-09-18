"""Read-only rejection tests; a real-USD integration probe is also required."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from utils.franka_mimic_force import validate_franka_drive_force, validate_single_mimic_finger_drive


class FrankaMimicForceTests(unittest.TestCase):
    def test_larger_force_requires_explicit_single_actuator_semantics(self):
        validate_franka_drive_force(35, "legacy")
        validate_franka_drive_force(50, "single_mimic_drive")
        validate_franka_drive_force(70, "single_mimic_drive")
        for force, semantics in (
            (35.1, "legacy"),
            (70.1, "single_mimic_drive"),
            (140, "single_mimic_drive"),
            (float("nan"), "single_mimic_drive"),
            (float("inf"), "legacy"),
            (0, "single_mimic_drive"),
            (-1, "legacy"),
            (35, "per_finger"),
        ):
            with self.subTest(force=force, semantics=semantics), self.assertRaises(ValueError):
                validate_franka_drive_force(force, semantics)

    def fixture(self):
        root = "/World/Franka/panda_hand/panda_finger_joint"
        attributes = [
            {"physics:jointEnabled": True, "physics:excludeFromArticulation": False, "physics:axis": "X"},
            {"physics:jointEnabled": True, "physics:excludeFromArticulation": False, "physics:axis": "X"},
        ]
        attributes[0]["drive:linear:physics:type"] = "force"
        attributes[1].update(
            {
                "physxMimicJoint:rotX:gearing": -1.0,
                "physxMimicJoint:rotX:offset": 0.0,
                "physxMimicJoint:rotX:referenceJointAxis": "rotX",
            }
        )
        schemas = [["PhysicsDriveAPI:linear"], ["PhysxMimicJointAPI:rotX"]]
        joints = []
        for index in range(2):
            prim = Mock()
            prim.IsValid.return_value = True
            prim.GetTypeName.return_value = "PhysicsPrismaticJoint"
            prim.GetAppliedSchemas.return_value = schemas[index]
            prim.GetAttribute.side_effect = lambda key, i=index: SimpleNamespace(Get=lambda: attributes[i].get(key))
            prim.GetRelationship.return_value.GetTargets.return_value = [root + "1"]
            joints.append(prim)
        stage = Mock()
        stage.GetPrimAtPath.side_effect = {root + str(i + 1): prim for i, prim in enumerate(joints)}.get
        return stage, joints, attributes, schemas

    def test_accepts_exact_one_drive_topology_without_writes(self):
        stage, joints, _, _ = self.fixture()
        result = validate_single_mimic_finger_drive(stage, "/World/Franka")
        self.assertEqual(result["continuous_total_force_cap_N"], 70)
        for joint in joints:
            self.assertTrue(all(call[0].startswith(("Get", "Is")) for call in joint.mock_calls))

    def test_rejects_independently_driven_follower(self):
        stage, _, _, schemas = self.fixture()
        schemas[1].append("PhysicsDriveAPI:linear")
        with self.assertRaisesRegex(ValueError, "undriven"):
            validate_single_mimic_finger_drive(stage, "/World/Franka")

    def test_original_zero_gain_follower_requires_explicit_opt_in(self):
        stage, _, attributes, schemas = self.fixture()
        schemas[1].append("PhysicsDriveAPI:linear")
        attributes[1].update({"drive:linear:physics:stiffness": 0.0, "drive:linear:physics:damping": 0.0})
        with self.assertRaises(ValueError):
            validate_single_mimic_finger_drive(stage, "/World/Franka")
        validate_single_mimic_finger_drive(stage, "/World/Franka", allow_zero_gain_follower=True)
        for name in ("stiffness", "damping"):
            key = "drive:linear:physics:" + name
            attributes[1][key] = 1.0
            with self.assertRaises(ValueError):
                validate_single_mimic_finger_drive(stage, "/World/Franka", allow_zero_gain_follower=True)
            attributes[1][key] = 0.0

    def test_rejects_different_mimic_reference(self):
        stage, joints, _, _ = self.fixture()
        joints[1].GetRelationship.return_value.GetTargets.return_value = ["/some_other_joint"]
        with self.assertRaisesRegex(ValueError, "gearing"):
            validate_single_mimic_finger_drive(stage, "/World/Franka")

    def test_rejects_missing_joint(self):
        stage, joints, _, _ = self.fixture()
        joints[0].IsValid.return_value = False
        with self.assertRaisesRegex(ValueError, "prismatic"):
            validate_single_mimic_finger_drive(stage, "/World/Franka")

    def test_rejects_unverified_attributes(self):
        for index, key, value in (
            (0, "physics:jointEnabled", False),
            (1, "physics:excludeFromArticulation", True),
            (1, "physics:axis", "Y"),
            (0, "drive:linear:physics:type", "acceleration"),
            (1, "physxMimicJoint:rotX:gearing", -2.0),
            (1, "physxMimicJoint:rotX:gearing", float("nan")),
            (1, "physxMimicJoint:rotX:offset", 0.001),
            (1, "physxMimicJoint:rotX:offset", None),
            (1, "physxMimicJoint:rotX:referenceJointAxis", "rotY"),
        ):
            with self.subTest(key=key, value=value):
                stage, _, attributes, _ = self.fixture()
                attributes[index][key] = value
                with self.assertRaises(ValueError):
                    validate_single_mimic_finger_drive(stage, "/World/Franka")


if __name__ == "__main__":
    unittest.main()
