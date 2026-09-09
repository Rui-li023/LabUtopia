"""Isolated finger-drive diagnostic or explicit free-coupon force calibration.

The optional calibration uses a test coupon, never a task pipette, and
does not count as grasp/placement acceptance or modify any saved robot asset.
"""

import argparse
import hashlib
import json
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--servo-drive", choices=("velocity", "position"), default="velocity")
parser.add_argument("--opening-width", type=float, default=0.08)
parser.add_argument("--physics-hz", type=int, choices=(120, 240, 480), default=120)
parser.add_argument("--calibration-report", type=Path)
parser.add_argument("--force-caps", nargs="+", type=float, default=[30, 35, 50, 70])
options = parser.parse_args()
if options.calibration_report is not None:
    if (
        options.calibration_report.exists()
        or not options.force_caps
        or any(not 0 < force <= 70 for force in options.force_caps)
    ):
        parser.error("Calibration needs a new report and finite drive caps in (0, 70] N")
    if options.opening_width < 0.04 or options.servo_drive != "velocity":
        parser.error("Coupon calibration requires velocity drive and at least 40 mm opening")

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from loguru import logger  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from robots.franka.franka import Franka  # noqa: E402
from utils.franka_mimic_force import validate_single_mimic_finger_drive  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / options.physics_hz)
robot = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
robot.set_gripper_control_mode("force", closing_force=30)
world.reset()
robot.initialize()
robot.post_reset()
cache = UsdGeom.XformCache()
hand_inverse = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath("/World/Franka/panda_hand")).GetInverse()
tcp = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(robot.gripper_center_prim_path)) * hand_inverse
logger.info("GEOMETRY TCP in hand={}", tcp.ExtractTranslation())
for name in ("panda_leftfinger", "panda_rightfinger"):
    root = world.stage.GetPrimAtPath("/World/Franka/" + name)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            mat, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            logger.info(
                "FINGER PHYSICS {} material={} values={}",
                prim.GetPath(),
                mat.GetPath() if mat else None,
                {a.GetName(): str(a.Get()) for a in mat.GetPrim().GetAttributes()} if mat else {},
            )
            transform = cache.GetLocalToWorldTransform(prim) * hand_inverse
            points = np.asarray([transform.Transform(point) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
            edge = points[:, 1].min() if name == "panda_leftfinger" else points[:, 1].max()
            inner = points[np.abs(points[:, 1] - edge) < 0.001]
            logger.info(
                "GEOMETRY {} bounds={}..{}, jaw-facing strip={}..{}",
                prim.GetPath(),
                points.min(0),
                points.max(0),
                inner.min(0),
                inner.max(0),
            )
for prim in world.stage.Traverse():
    if "finger_joint" in prim.GetName():
        logger.info("JOINT {} {}", prim.GetPath(), {a.GetName(): str(a.Get()) for a in prim.GetAttributes()})


def calibrate_force():
    """Measure balanced jaw normal forces on a free 30 mm calibration coupon.

    Gravity is disabled only on this isolated test coupon. Unlike a fixed
    coupon, it cannot carry an external reaction that obscures jaw-force
    sharing through the mimic joint. This is not a task-grasp experiment.
    """
    stage = world.stage
    topology = validate_single_mimic_finger_drive(stage, robot.prim_path_str)
    # Articulation reset sets tensor joint positions; the USD link transforms
    # still describe the asset's authored pose until physics has advanced.
    # Position the test coupon only after the open hand has actually settled.
    robot.enable_gripper_force_servo(
        speed=0.02,
        opening_speed=0.05,
        position_iterations=128,
        velocity_iterations=32,
        drive_mode="velocity",
        opening_width=options.opening_width,
    )
    world.reset()
    robot.initialize()
    robot.post_reset()
    robot.open_gripper()
    arm_target = robot.get_joint_positions()[:7].copy()
    for _ in range(options.physics_hz):
        robot.apply_action(ArticulationAction(joint_positions=arm_target, joint_indices=np.arange(7)))
        robot.apply_gripper_effort()
        world.step(render=False)
    tcp_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot.gripper_center_prim_path))
    hand_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Franka/panda_hand"))
    coupon = UsdGeom.Cube.Define(stage, "/World/ForceCalibrationCoupon")
    coupon.CreateSizeAttr(1.0)
    coupon.AddTranslateOp().Set(tcp_matrix.ExtractTranslation())
    coupon.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(hand_matrix.ExtractRotationQuat())
    coupon.AddScaleOp().Set(Gf.Vec3f(0.020, 0.030, 0.012))
    UsdPhysics.CollisionAPI.Apply(coupon.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(coupon.GetPrim())
    UsdPhysics.MassAPI.Apply(coupon.GetPrim()).CreateMassAttr(0.09)
    coupon_physics = PhysxSchema.PhysxRigidBodyAPI.Apply(coupon.GetPrim())
    coupon_physics.CreateDisableGravityAttr(True)
    coupon_physics.CreateSolverPositionIterationCountAttr(128)
    coupon_physics.CreateSolverVelocityIterationCountAttr(32)
    coupon_body = world.scene.add(
        SingleRigidPrim(str(coupon.GetPath()), name="force_calibration_coupon", reset_xform_properties=False)
    )
    for name in ("panda_leftfinger", "panda_rightfinger"):
        body = stage.GetPrimAtPath("/World/Franka/" + name)
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
    impulses = np.zeros(2)
    contact_counts = np.zeros(2, dtype=int)

    def contact_report(headers, contacts):
        for header in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.collider0, header.collider1)]
            if str(coupon.GetPath()) not in paths:
                continue
            for index, name in enumerate(("panda_leftfinger", "panda_rightfinger")):
                if not any(path.startswith("/World/Franka/" + name + "/") for path in paths):
                    continue
                for point in contacts[
                    header.contact_data_offset : header.contact_data_offset + header.num_contact_data
                ]:
                    impulse = np.array([point.impulse.x, point.impulse.y, point.impulse.z])
                    normal = np.array([point.normal.x, point.normal.y, point.normal.z])
                    impulses[index] += abs(float(np.dot(impulse, normal)))
                    contact_counts[index] += 1

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
    if subscription is None:
        raise RuntimeError("Could not subscribe to physical coupon contacts")
    records = []
    for force in options.force_caps:
        robot.set_gripper_control_mode("force", closing_force=force, closing_speed=0.02)
        robot.enable_gripper_force_servo(
            speed=0.02,
            opening_speed=0.05,
            position_iterations=128,
            velocity_iterations=32,
            drive_mode="velocity",
            opening_width=options.opening_width,
        )
        world.reset()
        robot.initialize()
        robot.post_reset()
        arm_target = robot.get_joint_positions()[:7].copy()
        history = []
        open_tcp = None
        for opening, seconds in ((True, 1.0), (False, 3.0)):
            robot.open_gripper() if opening else robot.close_gripper()
            for frame in range(round(seconds * options.physics_hz)):
                impulses[:] = 0
                contact_counts[:] = 0
                robot.apply_action(ArticulationAction(joint_positions=arm_target, joint_indices=np.arange(7)))
                robot.apply_gripper_effort()
                world.step(render=False)
                if not opening:
                    history.append(
                        {
                            "frame": frame,
                            "normal_force_N": (impulses * options.physics_hz).tolist(),
                            "contacts": contact_counts.tolist(),
                            "finger_positions_m": robot.get_joint_positions()[-2:].tolist(),
                            "finger_velocities_m_s": robot.get_joint_velocities()[-2:].tolist(),
                        }
                    )
            if opening:
                open_tcp = UsdGeom.XformCache().GetLocalToWorldTransform(
                    stage.GetPrimAtPath(robot.gripper_center_prim_path)
                )
        tail = history[-options.physics_hz :]
        normals = np.asarray([item["normal_force_N"] for item in tail])
        record = {
            "drive_cap_N": force,
            "final_second_mean_normal_force_N": normals.mean(0).tolist(),
            "final_second_min_normal_force_N": normals.min(0).tolist(),
            "final_second_max_normal_force_N": normals.max(0).tolist(),
            "final_second_mean_total_normal_force_N": float(normals.sum(1).mean()),
            "every_tail_step_has_both_finger_contacts": bool(np.all(np.asarray([v["contacts"] for v in tail]) > 0)),
            "runtime_max_efforts_N": robot._articulation_view.get_max_efforts()[0, -2:].tolist(),
            "open_tcp_position_world": list(open_tcp.ExtractTranslation()),
            "coupon_to_open_tcp_distance_m": float(
                (tcp_matrix.ExtractTranslation() - open_tcp.ExtractTranslation()).GetLength()
            ),
            "final_coupon_position_world": coupon_body.get_world_pose()[0].tolist(),
            "every_tail_step_has_balanced_positive_normal_forces": bool(
                np.all(normals > 1.0)
                and np.max(np.abs(normals[:, 0] - normals[:, 1]) / np.maximum(normals.sum(1), 1e-9)) < 0.1
            ),
            "samples": history,
        }
        records.append(record)
        logger.info("COUPON {}", {k: v for k, v in record.items() if k != "samples"})
    joints = []
    for prim in stage.Traverse():
        if "finger_joint" not in prim.GetName():
            continue
        joints.append(
            {
                "path": str(prim.GetPath()),
                "schemas": list(prim.GetAppliedSchemas()),
                "attributes": {
                    a.GetName(): str(a.Get())
                    for a in prim.GetAttributes()
                    if "drive:" in a.GetName() or "Mimic" in a.GetName()
                },
                "relationships": {
                    r.GetName(): [str(p) for p in r.GetTargets()]
                    for r in prim.GetRelationships()
                    if "Mimic" in r.GetName()
                },
            }
        )
    report = {
        "calibration_only": True,
        "task_acceptance_credit": False,
        "physics_hz": options.physics_hz,
        "coupon_static": False,
        "coupon_gravity_disabled": True,
        "coupon_mass_kg": 0.09,
        "coupon_dimensions_m": [0.020, 0.030, 0.012],
        "coupon_position_world": list(tcp_matrix.ExtractTranslation()),
        "robot_source_layers": [layer.identifier for layer in stage.GetUsedLayers()],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("robots/franka/franka.py"), Path("utils/franka_mimic_force.py"))
        },
        "joints": joints,
        "validated_single_actuator_topology": topology,
        "runs": records,
        "calibration_valid": all(
            item["every_tail_step_has_both_finger_contacts"]
            and item["every_tail_step_has_balanced_positive_normal_forces"]
            for item in records
        ),
    }
    options.calibration_report.parent.mkdir(parents=True, exist_ok=True)
    with options.calibration_report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    subscription = None
    if not report["calibration_valid"]:
        raise RuntimeError("Invalid force calibration: steady bilateral coupon contacts were not measured")


if options.calibration_report is not None:
    calibrate_force()
    app.close()
    raise SystemExit(0)

for label, efforts in (
    ("both_1N", [-1, -1]),
    ("both_5N", [-5, -5]),
    ("both_15N", [-15, -15]),
    ("lead_30N", [-30, 0]),
    ("both_30N", [-30, -30]),
):
    robot.set_joint_positions(np.array([0.03, 0.03]), joint_indices=np.array([7, 8]))
    for frame in range(120):
        robot.apply_action(ArticulationAction(joint_efforts=np.array(efforts), joint_indices=np.array([7, 8])))
        world.step(render=False)
        if frame in (0, 30, 119):
            logger.info(
                "DRIVE {} frame={} joints={} velocities={} effort={}",
                label,
                frame,
                robot.get_joint_positions()[-2:],
                robot.get_joint_velocities()[-2:],
                robot.get_articulation_controller().get_applied_action(),
            )
robot.enable_gripper_force_servo(speed=0.05, drive_mode=options.servo_drive, opening_width=options.opening_width)
for reset_index in range(2):
    world.reset()
    robot.initialize()
    robot.post_reset()
    for opening in (False, True):
        robot.open_gripper() if opening else robot.close_gripper()
        for _frame in range(120):
            robot.apply_gripper_effort()
            world.step(render=False)
        positions = robot.get_joint_positions()[-2:]
        logger.info("SERVO reset={} opening={} positions={}", reset_index, opening, positions)
        if opening:
            np.testing.assert_allclose(positions, options.opening_width / 2, atol=0.001)
        else:
            assert np.all(positions < 0.001)
app.close()
