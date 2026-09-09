"""Headless Franka contact bench for the rotary evaporator.

The robot moves its own joints. Instrument actions must originate from contact;
no button press, dial target or switch command is sent to the instrument here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=ROOT / "outputs/rotary_evaporator/franka_contact")
parser.add_argument(
    "--buttons",
    nargs="+",
    default=["motor_run", "motor_run", "bath_up", "bath_down", "bath_set", "bath_mode", "motor_power", "motor_power"],
)
parser.add_argument("--scan", action="store_true")
parser.add_argument("--dial", action="store_true", help="Physically grip and turn the extended dial")
parser.add_argument("--snapshot", action="store_true")
parser.add_argument("--record", action="store_true", help="Record the robot and instrument from an offscreen camera")
parser.add_argument("--renderer", choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting")
parser.add_argument("--render-samples", type=int, default=32)
parser.add_argument(
    "--usd", type=Path, default=ROOT / "assets/chemistry_lab/rotary_evaporator/rotary_evaporator_franka.usda"
)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
app = SimulationApp(
    {
        "headless": True,
        "width": 640,
        "height": 640,
        "renderer": args.renderer,
        "samples_per_pixel_per_frame": args.render_samples,
        "max_bounces": 12,
        "max_specular_transmission_bounces": 12,
    }
)

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader  # noqa: E402
from isaacsim.robot_motion.motion_generation.lula.path_planners import RRT  # noqa: E402
from loguru import logger  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from experiments.assets.rotary_evaporator_controls import EvaporatorControls  # noqa: E402
from experiments.assets.rotary_evaporator_device import EvaporatorDevice  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO")

DT = 1 / 240
GROUND_Z = -0.10
BASE_POSITION = np.array([-0.30, -0.50, 0.0])
BASE_QUAT = np.array([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
HOME = np.array([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398, 0, 0])
FRONT = Rotation.from_euler("x", -90, degrees=True)
FINGER_TIP = np.array([0, 0.010, 0.1123])


def wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def setup_scene():
    world = World(stage_units_in_meters=1, physics_dt=DT, rendering_dt=1 / 24)
    world.scene.add(GroundPlane("/World/Ground", z_position=GROUND_Z, size=10))
    add_reference_to_stage(str(args.usd.resolve()), "/World/Evaporator")
    instrument = world.scene.add(SingleArticulation("/World/Evaporator", name="evaporator"))
    add_reference_to_stage(str(ROOT / "assets/robots/Franka.usd"), "/World/Franka")
    robot = world.scene.add(
        SingleArticulation("/World/Franka", name="franka", position=BASE_POSITION, orientation=BASE_QUAT)
    )
    robot.set_joints_default_state(positions=HOME)
    stage = world.stage
    # All robot changes are local scene overrides; the shared Franka asset stays intact.
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Franka") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        if path.startswith(("/World/Franka", "/World/Evaporator")) and prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.0003)
            collision.CreateRestOffsetAttr(0)
    material = UsdShade.Material.Define(stage, "/World/FingertipContact")
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(0.9)
    friction.CreateDynamicFrictionAttr(0.7)
    friction.CreateRestitutionAttr(0)
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        source = stage.GetPrimAtPath(f"/World/Franka/{finger}/geometry/{finger}")
        source.RemoveAPI(UsdPhysics.CollisionAPI)
        source.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        source.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        # Dedicated contact geometry has one physical material, avoiding the
        # visual mesh's material subsets and PhysX's default-friction fallback.
        contact = UsdGeom.Mesh.Define(stage, f"/World/Franka/{finger}/ContactShape")
        for attr in ("points", "faceVertexCounts", "faceVertexIndices", "extent"):
            contact.GetPrim().GetAttribute(attr).Set(source.GetAttribute(attr).Get())
        relative = (
            UsdGeom.XformCache().GetLocalToWorldTransform(source)
            * UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/Franka/{finger}")).GetInverse()
        )
        contact.AddTransformOp().Set(relative)
        contact.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(contact.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(contact.GetPrim()).CreateApproximationAttr("convexHull")
        collision = PhysxSchema.PhysxCollisionAPI.Apply(contact.GetPrim())
        collision.CreateContactOffsetAttr(0.0003)
        collision.CreateRestOffsetAttr(0)
        UsdShade.MaterialBindingAPI.Apply(contact.GetPrim()).Bind(material, materialPurpose="physics")
    pedestal = UsdGeom.Cube.Define(stage, "/World/RobotPedestal")
    pedestal.CreateSizeAttr(1)
    pedestal.AddTranslateOp().Set(Gf.Vec3d(BASE_POSITION[0], BASE_POSITION[1], (BASE_POSITION[2] + GROUND_Z) / 2))
    pedestal.AddScaleOp().Set(Gf.Vec3f(0.20, 0.20, BASE_POSITION[2] - GROUND_Z))
    pedestal.CreateDisplayColorAttr([(0.16, 0.18, 0.20)])
    UsdPhysics.CollisionAPI.Apply(pedestal.GetPrim())
    # Both mounts end at z=0. The open space in front of the instrument gives
    # the wrist room below the low bath panel without contacting the floor.
    stand = UsdGeom.Cube.Define(stage, "/World/InstrumentPedestal")
    stand.CreateSizeAttr(1)
    stand.AddTranslateOp().Set(Gf.Vec3d(0, -0.005, GROUND_Z / 2))
    stand.AddScaleOp().Set(Gf.Vec3f(0.55, 0.30, -GROUND_Z))
    stand.CreateDisplayColorAttr([(0.16, 0.18, 0.20)])
    UsdPhysics.CollisionAPI.Apply(stand.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(800)
    light = UsdLux.RectLight.Define(stage, "/World/Key")
    light.CreateIntensityAttr(1800)
    light.CreateWidthAttr(1)
    light.CreateHeightAttr(1)
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1, -2, 2), Gf.Vec3d(0, 0, 0.5), Gf.Vec3d(0, 0, 1)).GetInverse()
    light.AddTransformOp().Set(transform)
    world.reset()
    controller = robot.get_articulation_controller()
    kp = np.array([12000, 12000, 10000, 10000, 5000, 5000, 4000, 10000, 0], dtype=float)
    kd = np.array([300, 300, 250, 250, 150, 150, 120, 200, 0], dtype=float)
    controller.set_gains(kps=kp, kds=kd)
    robot._articulation_view.set_max_efforts(np.array([[87, 87, 87, 87, 12, 12, 12, 7.2, 0]], dtype=float))
    device = EvaporatorDevice(EvaporatorControls(instrument), stage)
    solver = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("Franka"))
    solver.set_robot_base_pose(BASE_POSITION, BASE_QUAT)
    return world, robot, device, solver


class ContactBench:
    def __init__(self, world, robot, device, solver):
        self.world, self.robot, self.device, self.solver = world, robot, device, solver
        self.command = HOME.copy()
        self.frame = 0
        self.phase = "initialize"
        self.target = None
        self.contacts = []
        self.results = []
        self.recorder = None
        self.max_travels = dict.fromkeys(self.device.indices, 0.0)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)

    def on_contact(self, headers, data):
        for h in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1)]
            if not any(p.startswith("/World/Franka/") for p in paths):
                continue
            impulse = sum(
                np.linalg.norm([p.impulse.x, p.impulse.y, p.impulse.z])
                for p in data[h.contact_data_offset : h.contact_data_offset + h.num_contact_data]
            )
            if impulse < 1e-8:
                continue
            if any("/Franka/panda_link0/" in p for p in paths) and "/World/RobotPedestal" in paths:
                continue  # Intended fixed base mounting surface.
            allowed = (
                self.target
                and any(f"/Evaporator/{self.target}/" in p for p in paths)
                and any("finger/" in p for p in paths)
            )
            self.contacts.append(
                {
                    "frame": self.frame,
                    "phase": self.phase,
                    "paths": paths,
                    "normal_force_estimate_n": float(impulse / DT),
                    "expected": bool(allowed),
                }
            )

    def tick(self, count=1):
        for _ in range(count):
            first_contact = len(self.contacts)
            self.device.update(DT)
            self.robot.apply_action(ArticulationAction(joint_positions=self.command, joint_indices=np.arange(9)))
            self.world.step(render=False)
            self.frame += 1
            assert np.isfinite(self.robot.get_joint_positions()).all()
            device_q = self.device.articulation.get_joint_positions()
            for button, index in self.device.indices.items():
                self.max_travels[button] = max(self.max_travels[button], float(device_q[index]))
            unexpected = [c for c in self.contacts[first_contact:] if not c["expected"]]
            if unexpected:
                raise RuntimeError(f"Unexpected contact in {self.phase}: {unexpected[0]}")
            if self.recorder and self.frame % 10 == 0:
                self.device.refresh_displays()
                self.recorder.frame(self.phase, self.device.state())

    def pose(self):
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.world.stage.GetPrimAtPath("/World/Franka/panda_hand")
        )
        quat = matrix.ExtractRotationQuat()
        return np.array(matrix.ExtractTranslation()), Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()])

    def center(self, name):
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.world.stage.GetPrimAtPath(self.device.root + "/" + name)
        )
        return np.array(matrix.ExtractTranslation())

    def ik(self, position, orientation, warm=None, continuous=False):
        if continuous:
            # Local damped least squares stays on the current joint branch.
            # Lula's multi-seed endpoint IK is reserved for planned transits.
            q = self.command[:7].astype(float).copy()
            bounds = self.robot._articulation_view.get_dof_limits()[0, :7]
            for _ in range(24):
                p, r = self.solver.compute_forward_kinematics("panda_hand", q)
                error = np.r_[position - p, Rotation.from_matrix(orientation.as_matrix() @ r.T).as_rotvec() * 0.15]
                if np.linalg.norm(error[:3]) < 0.00008 and np.linalg.norm(error[3:]) < 0.0003:
                    return q
                jac = np.empty((6, 7))
                for j in range(7):
                    probe = q.copy()
                    probe[j] += 0.0001
                    p1, r1 = self.solver.compute_forward_kinematics("panda_hand", probe)
                    jac[:, j] = np.r_[(p1 - p), Rotation.from_matrix(r1 @ r.T).as_rotvec() * 0.15] / 0.0001
                delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-6, error)
                q = np.clip(q + np.clip(delta, -0.035, 0.035), bounds[:, 0] + 0.0001, bounds[:, 1] - 0.0001)
            raise RuntimeError(f"Local IK did not converge in {self.phase}: {error}")
        q, success = self.solver.compute_inverse_kinematics(
            "panda_hand",
            np.asarray(position),
            wxyz(orientation),
            self.command[:7] if warm is None else warm,
            position_tolerance=0.0001,
            orientation_tolerance=0.002,
        )
        if not success:
            raise RuntimeError(f"IK failed in {self.phase}: {position}")
        return q

    def move(self, position, orientation, seconds=2.0):
        start, rotation = self.pose()
        rotations = Slerp([0, 1], Rotation.from_quat([rotation.as_quat(), orientation.as_quat()]))
        steps = max(1, round(seconds / DT))
        for i in range(1, steps + 1):
            t = i / steps
            smooth = t * t * (3 - 2 * t)
            if i % 2 == 0 or i == 1:
                target = self.ik(start + (position - start) * smooth, rotations(smooth), continuous=True)
                jump = float(np.max(np.abs(target - self.command[:7])))
                if jump > 0.12:
                    raise RuntimeError(
                        f"Discontinuous IK in {self.phase}: jump={jump:.3f}, hand={start + (position - start) * smooth}, previous={self.command[:7]}, next={target}"
                    )
                self.command[:7] = target
            self.tick()

    def transit(self, position, orientation):
        """Plan in joint space around actual instrument collider bounds."""

        class Box:
            def __init__(self, center, size):
                self.center, self.size = center, size

            def get_size(self):
                return 1.0

            def get_local_scale(self):
                return self.size

            def get_world_pose(self):
                return self.center, np.array([1.0, 0, 0, 0])

        config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        config["end_effector_frame_name"] = "panda_hand"
        planner = RRT(**config)
        planner.set_robot_base_pose(BASE_POSITION, BASE_QUAT)
        planner.set_max_iterations(6000)
        planner.set_param("step_size", 0.04)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], False, True)
        for prim in self.world.stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and (
                str(prim.GetPath()).startswith(self.device.root) or str(prim.GetPath()) == "/World/InstrumentPedestal"
            ):
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                center = np.array(bounds.GetMidpoint())
                size = np.array(bounds.GetSize()) + 0.016
                planner.add_cuboid(Box(center, size), static=True)
        planner.update_world()
        goal = self.ik(position, orientation)
        planner.set_cspace_target(goal)
        path = None
        for seed in (42, 43, 44):
            planner.set_random_seed(seed)
            path = planner.compute_path(self.robot.get_joint_positions()[:7], np.empty(0))
            if path is not None:
                break
        if path is None:
            raise RuntimeError(f"No collision-free transit in {self.phase}")
        logger.info("RRT {}: {} waypoints", self.phase, len(path))
        for target in path[1:]:
            start = self.command[:7].copy()
            duration = max(0.08, float(np.max(np.abs(target - start))) / 0.65)
            count = math.ceil(duration / DT)
            for i in range(1, count + 1):
                t = i / count
                self.command[:7] = start + (target - start) * (t * t * (3 - 2 * t))
                self.tick()
        self.tick(120)
        tracking = np.max(np.abs(self.robot.get_joint_positions()[:7] - self.command[:7]))
        logger.info("TRANSIT tracking error {:.5f} rad in {}", tracking, self.phase)
        if tracking > 0.05:
            raise RuntimeError(f"Robot did not reach transit endpoint in {self.phase}: {tracking:.5f} rad")

    def press_button(self, name):
        self.target = name
        self.phase = "approach_" + name
        contact_start = len(self.contacts)
        self.max_travels[name] = 0.0
        center = self.center(name)
        # One closed fingertip lands on the cap; the other clears it vertically.
        hand = center - FRONT.apply(FINGER_TIP) + np.array([0, -0.002, 0])
        pre = hand + np.array([0, -0.08, 0])
        self.transit(pre, FRONT)
        self.tick(120)
        previous = len(self.device.events)
        maximum = 0.0
        self.phase = "press_" + name
        # Slow approach and bounded 2.5 mm overtravel, stopping on a real event.
        for offset in np.linspace(-0.08, 0.0025, 250):
            self.command[:7] = self.ik(hand + np.array([0, offset, 0]), FRONT, continuous=True)
            self.tick(4)
            maximum = max(maximum, float(self.device.articulation.get_joint_positions()[self.device.indices[name]]))
            if len(self.device.events) > previous:
                break
        self.tick(60)
        events = self.device.events[previous:]
        self.phase = "retract_" + name
        self.move(pre, FRONT, 1.5)
        self.tick(120)
        contacts = self.contacts[contact_start:]
        maximum = max(maximum, self.max_travels[name])
        result = {
            "button": name,
            "events": events,
            "max_travel_m": maximum,
            "contact_samples": len(contacts),
            "unexpected_contacts": [c for c in contacts if not c["expected"]],
            "state": self.device.state(),
        }
        self.results.append(result)
        logger.info(
            "BUTTON {} events={} travel={:.5f} contacts={} unexpected={}",
            name,
            len(events),
            maximum,
            len(contacts),
            len(result["unexpected_contacts"]),
        )
        assert len(events) == 1 and events[0]["button"] == name, result
        assert maximum > self.device.config["press"]["threshold_m"] and any(c["expected"] for c in contacts), result
        assert not result["unexpected_contacts"], result["unexpected_contacts"][:3]
        self.target = None

    def turn_dial(self):
        self.target = "speed_dial"
        self.phase = "approach_dial"
        begin = len(self.contacts)
        center = self.center("speed_dial") + np.array([0, -0.023, 0])
        hand = center - FRONT.apply(np.array([0, 0, 0.095]))
        pre = hand + np.array([0, -0.12, 0])
        self.transit(pre, FRONT)
        self.command[7:] = 0.035
        self.tick(240)
        self.move(hand, FRONT, 2.5)
        # Correct the small gravity/load deflection before closing symmetric
        # mimic fingers around a fixed-axis knob. A one-sided contact otherwise
        # stops both jaws with a gap remaining at the other finger.
        compensated = hand.copy()
        for _ in range(4):
            self.tick(60)
            measured, _ = self.pose()
            compensated += hand - measured
            self.move(compensated, FRONT, 0.25)
        self.phase = "grasp_dial"
        self.command[7:] = 0.012
        self.tick(300)
        grasped = self.contacts[begin:]
        for finger in ("panda_leftfinger", "panda_rightfinger"):
            assert any(c["expected"] and any(finger in p for p in c["paths"]) for c in grasped), (
                f"No grip contact on {finger}"
            )
        initial = self.device.state()["dial_angle_deg"]
        for rpm in (80, 60):
            # Read the actual grasp angle: closing can nudge a free knob. Keep
            # the final physical pointer on the printed 80/60 rpm graduations.
            destination = rpm * 2.5
            angle = destination - initial
            self.phase = "turn_dial_" + str(rpm) + "rpm"
            self.move(compensated, Rotation.from_euler("y", angle, degrees=True) * FRONT, 3.0)
            self.tick(180)
            for _ in range(3):
                error = destination - self.device.state()["dial_angle_deg"]
                if abs(error) < 0.3:
                    break
                angle += error
                self.move(compensated, Rotation.from_euler("y", angle, degrees=True) * FRONT, 0.5)
                self.tick(120)
            state = self.device.state()
            measured = state["dial_angle_deg"] - initial
            logger.info("DIAL requested_delta={} measured_delta={} rpm_set={}", angle, measured, state["rpm_set"])
            result = {
                "operation": "dial",
                "requested_rpm": rpm,
                "requested_absolute_degrees": destination,
                "requested_delta_degrees": angle,
                "measured_delta_degrees": measured,
                "state": state,
            }
            self.results.append(result)
            assert abs(state["dial_angle_deg"] - destination) < 0.6 and round(state["rpm_set"]) == rpm, result
        self.phase = "release_dial"
        self.command[7:] = 0.035
        self.tick(240)
        self.move(pre, FRONT, 2.0)
        self.command[7:] = 0
        self.tick(240)
        assert not any(not c["expected"] for c in self.contacts[begin:]), "Unexpected contact during dial grasp"
        self.target = None


def main():
    world, robot, device, solver = setup_scene()
    bench = ContactBench(world, robot, device, solver)
    if args.snapshot or args.record:
        from experiments.assets.franka_evaporator_recording import BenchRecording

        bench.recorder = BenchRecording(world, args.out, args.record)
    success, error = False, ""
    try:
        bench.tick(900)
        if bench.recorder:
            bench.recorder.snapshot("franka_proportions")
        position, rotation = bench.pose()
        fk, _ = solver.compute_forward_kinematics("panda_hand", robot.get_joint_positions()[:7])
        logger.info("POSE hand={} FK={} fingers={}", position, fk, robot.get_joint_positions()[7:])
        assert np.linalg.norm(position - fk) < 0.001
        for name in dict.fromkeys(args.buttons):
            hand = bench.center(name) - FRONT.apply(FINGER_TIP) + np.array([0, -0.082, 0])
            solution = bench.ik(hand, FRONT)
            logger.info("REACHABLE {} {} q={}", name, hand, solution)
        if not args.scan:
            for name in args.buttons:
                bench.press_button(name)
            if args.dial:
                bench.turn_dial()
        assert not any(not contact["expected"] for contact in bench.contacts), "Unexpected robot/instrument contact"
        success = True
    except Exception as exc:
        error = str(exc)
        logger.exception("Franka contact validation failed")
    finally:
        report = {
            "passed": success,
            "mode": "reachability_only" if args.scan else "physical_contact",
            "error": error,
            "physics_steps": bench.frame,
            "results": bench.results,
            "contacts": bench.contacts,
            "unexpected_contact_count": sum(not c["expected"] for c in bench.contacts),
            "device_state": device.state(),
            "robot_q": robot.get_joint_positions().tolist(),
            "robot_command": bench.command.tolist(),
            "robot_base_position_m": BASE_POSITION.tolist(),
            "instrument_usd": str(args.usd.resolve()),
        }
        if bench.recorder:
            bench.recorder.snapshot()
            report["recording"] = bench.recorder.close()
        world.stage.GetRootLayer().Export(str(args.out / "scene.usda"))
        (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        logger.info("Report {}", args.out / "report.json")
        world.stop()
        app.close()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
