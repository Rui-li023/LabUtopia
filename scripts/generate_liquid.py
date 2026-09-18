"""Generate/preview independent liquid volumes from YAML, without a task.

Example: python -m scripts.generate_liquid --config experiments/config/liquid/particle_positions.yaml
         --output outputs/liquid_positions_demo --headless
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

from utils.liquid_metrics import BeakerInterior
from utils.liquid_particles import ParticleFluidParameters, sample_container, sample_volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New output directory; existing results are never overwritten")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(cfg, dict) or not cfg.get("volumes"):
        parser.error("Configuration requires a nonempty volumes list")
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a fresh directory")
    output.mkdir(parents=True)

    # Isaac modules must load after SimulationApp, including --help usage.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "fast_shutdown": True,
            "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
        }
    )
    try:
        import cv2
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom, UsdLux
        from scipy.spatial.transform import Rotation

        from utils.particle_fluid import ParticleFluid, configure_particle_world
        from utils.robot_rendering import prepare_robot_visuals

        world = World(stage_units_in_meters=1, physics_prim_path="/physicsScene", backend="numpy")
        stage = omni.usd.get_context().get_stage()
        if cfg.get("scene_usd"):
            add_reference_to_stage(str(Path(cfg["scene_usd"]).resolve()), "/World")
        configure_particle_world(world, stage)
        for robot_path in cfg.get("robot_roots", []):
            prepare_robot_visuals(stage, world, robot_path)
        if cfg.get("add_ground_plane", not cfg.get("scene_usd")):
            world.scene.add_default_ground_plane()
        fluids = {}
        for volume in cfg["volumes"]:
            parameters = ParticleFluidParameters(**{**cfg.get("parameters", {}), **volume.get("parameters", {})})
            matrix = volume.get("frame_matrix")
            if volume.get("frame_path"):
                if matrix is not None:
                    raise ValueError("Specify frame_path OR frame_matrix, not both")
                prim = stage.GetPrimAtPath(volume["frame_path"])
                if not prim.IsValid():
                    raise ValueError(f"Unknown seed frame: {volume['frame_path']}")
                matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            if volume.get("shape") == "container":
                # Explicit measured profile, never an inferred solid AABB.
                profile = BeakerInterior(**volume["interior"])
                points = sample_container(
                    profile, matrix, spacing=parameters.spacing, fill_height=volume["fill_height"]
                )
            else:
                geometry = {
                    key: volume[key] for key in ("shape", "center", "size", "radius", "height") if key in volume
                }
                points = sample_volume(**geometry, matrix=matrix, spacing=parameters.spacing)
            root = volume["root_path"]
            fluids[root] = ParticleFluid(
                stage,
                scene_path=world.get_physics_context().prim_path,
                root_path=root,
                positions=points,
                velocities=volume.get("velocity"),
                parameters=parameters,
            )
        light = UsdLux.DomeLight.Define(stage, "/LiquidPreviewLight")
        light.CreateIntensityAttr(800)
        preview = cfg.get("preview", {})
        eye = np.asarray(preview.get("eye", [0.55, -0.65, 0.55]), dtype=float)
        target = np.asarray(preview.get("target", [0, 0, 0.08]), dtype=float)
        backward = eye - target
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        quaternion = Rotation.from_matrix(np.column_stack([right, up, backward])).as_quat()[[3, 0, 1, 2]]
        camera = Camera(prim_path="/LiquidPreviewCamera", resolution=(512, 512))
        world.reset()
        camera.initialize()
        camera.set_world_pose(position=eye, orientation=quaternion, camera_axes="usd")
        camera.set_clipping_range(0.01, 100.0)
        # Isaac's setters take stage metres, unlike raw USD camera attributes.
        camera.set_horizontal_aperture(float(preview.get("sensor_width_mm", 36)) / 1000)
        camera.set_focal_length(float(preview.get("focal_length_mm", 35)) / 1000)
        snapshots = {root: fluid.snapshot() for root, fluid in fluids.items()}
        stage.GetRootLayer().Export(str(output / "scene_initial.usda"))
        (output / "initial_state.json").write_text(json.dumps(snapshots))
        # Render products/annotators may take several updates to become ready.
        # Warm only the renderer so the exported liquid state stays unchanged.
        initial_rgb = None
        for _ in range(60):
            world.render()
            initial_rgb = camera.get_rgb()
            if initial_rgb is not None and np.asarray(initial_rgb).size:
                break
        if initial_rgb is None or np.asarray(initial_rgb).size == 0:
            raise RuntimeError("Liquid preview camera produced no initial image")
        for _ in range(10):
            world.render()
        initial_rgb = camera.get_rgb()
        cv2.imwrite(str(output / "initial.png"), cv2.cvtColor(np.asarray(initial_rgb)[..., :3], cv2.COLOR_RGB2BGR))
        for _ in range(args.steps):
            world.step(render=True)
        rgb = camera.get_rgb()
        if rgb is None or np.asarray(rgb).size == 0:
            raise RuntimeError("Liquid preview camera produced no image")
        cv2.imwrite(str(output / "preview.png"), cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR))
        report = {"test": "generic_particle_volumes", "task_success_evaluation": False, "volumes": {}}
        for root, fluid in fluids.items():
            final = fluid.snapshot()
            initial_points = np.asarray(snapshots[root]["positions"])
            final_points = np.asarray(final["positions"])
            report["volumes"][root] = {
                "particles": len(final_points),
                "initial_center": initial_points.mean(0).tolist(),
                "final_center": final_points.mean(0).tolist(),
                "centroid_shift_m": float(np.linalg.norm(final_points.mean(0) - initial_points.mean(0))),
            }
            fluid.restore(json.loads(json.dumps(snapshots[root])))
            np.testing.assert_array_equal(fluid.particles.GetPositionsAttr().Get(), snapshots[root]["positions"])
            np.testing.assert_array_equal(fluid.particles.GetVelocitiesAttr().Get(), snapshots[root]["velocities"])
            report["volumes"][root]["snapshot_roundtrip"] = True
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    except BaseException:
        # Kit fast shutdown can exit before a pending exception is printed.
        # Persist the cause first; a missing result.json is never a pass.
        error = traceback.format_exc()
        (output / "error.txt").write_text(error)
        print(error, file=sys.stderr, flush=True)
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
