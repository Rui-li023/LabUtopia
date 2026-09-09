"""Extended runner for one-off rendering and physics diagnostics.

Use the repository-root ``main.py`` for normal collection, inference, and replay.
"""

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy',
                       choices=['numpy', 'gpu'],
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true',
                       help='Disable video display and saving')
    parser.add_argument('--physics-only', action='store_true',
                       help='Collect-mode diagnostic only: no cameras, mock collector, no acceptance dataset')
    parser.add_argument('--config-name', type=str, default='level3_heat_liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', type=str, default='config',
                       help='Configuration directory path (default: config)')
    parser.add_argument('--max-episodes', type=int,
                       help='Override the configured episode count for this run')
    parser.add_argument('--max-attempts', type=int,
                       help='Stop after this many attempts, including failed episodes')
    parser.add_argument('--renderer', choices=['RayTracedLighting', 'PathTracing'],
                       help='Optional renderer override for publication renders')
    parser.add_argument('--render-width', type=int,
                       help='Override every configured camera width (pixels)')
    parser.add_argument('--render-height', type=int,
                       help='Override every configured camera height (pixels)')
    parser.add_argument('--single-camera', type=int,
                       help='Keep only the zero-based camera at this index')
    parser.add_argument('--camera-look-at', nargs=6, type=float, metavar=('EX', 'EY', 'EZ', 'TX', 'TY', 'TZ'),
                       help='Override the first camera with an eye position and look-at target')
    parser.add_argument('--camera-focal-length', type=float,
                       help='Override the first camera focal length, in STAGE UNITS (Isaac Sim'
                            " convention, matching the configs' focal_length). With a metres"
                            ' stage this is mm/10, so 2.71 here is a ~27 mm lens, not 2.71 mm.')
    parser.add_argument('--save-png-dir', type=str,
                       help='Save a clean PNG from the selected camera when an episode finishes')
    parser.add_argument('--save-png-on-failure', action='store_true',
                       help='Also save the final camera frame when an episode fails')
    parser.add_argument('--png-camera', type=int, default=0,
                       help='Zero-based camera index to save with --save-png-dir (default: 0)')
    parser.add_argument('--paper-lighting', action='store_true',
                       help='Use a fixed HDR + neutral studio light rig for publication renders')
    parser.add_argument('--paper-no-hdr', action='store_true',
                       help='Disable the HDR dome for a geometry-only publication background')
    parser.add_argument('--paper-background-usd', type=str,
                       help='Optional USD asset to reference under /World/PaperBackdrop for publication renders')
    parser.add_argument('--paper-background-translation', nargs=3, type=float,
                       metavar=('X', 'Y', 'Z'), default=[0.0, 5.0, 0.0],
                       help='World translation for --paper-background-usd (default: 0 5 0)')
    parser.add_argument('--paper-background-rotation', nargs=3, type=float,
                       metavar=('RX', 'RY', 'RZ'), default=[0.0, 0.0, 180.0],
                       help='XYZ Euler rotation in degrees for --paper-background-usd (default: 0 0 180)')
    parser.add_argument('--paper-background-scale', type=float, default=1.0,
                       help='Uniform scale for --paper-background-usd (default: 1.0)')
    parser.add_argument('--paper-background-room-scale', type=float, default=1.0,
                       help='Additional scale for a lab_015 room layer inside the background (default: 1.0)')
    parser.add_argument('--paper-hide-image-planes', action='store_true',
                       help='Hide image-backed backdrop meshes (for geometry-only lab renders)')
    parser.add_argument('--paper-hide-prims', nargs='+', default=[], metavar='PRIM_PATH',
                       help='Hide these prims in the render only; their colliders stay active so '
                            'the physics scene (and every tuned grasp) is unchanged')
    parser.add_argument('--paper-camera', action='store_true',
                       help='Replace every configured camera with a single static third-person '
                            'camera authored at --paper-camera-prim. Required for scenes whose '
                            'cameras are all robot-mounted (level 5), where overriding camera 0 '
                            'would only move a camera that is parented to the moving base.')
    parser.add_argument('--paper-camera-prim', type=str, default='/World/PaperCamera',
                       help='Prim path for the --paper-camera third-person camera')
    parser.add_argument('--paper-ood-materials', action='store_true',
                       help="Bind each material_paths entry's test_materials (the OOD set used at "
                            'level-3 inference) instead of its training materials, so an ID/OOD '
                            'figure pair can be rendered from the same collect config')
    parser.add_argument('--paper-fixed-phase-order', action='store_true',
                       help='Set task.fixed_phase_sequence so a controller that shuffles its '
                            'sub-task groups for data diversity (level-4 open-transport-pour) '
                            'runs them in the canonical storyline order instead')
    parser.add_argument('--seed', type=int,
                       help='Override cfg.seed so two runs (e.g. ID vs OOD) draw the same object '
                            'placements, material cycle and lighting samples')
    parser.add_argument('--keyframe-dir', type=str,
                       help='Save one clean PNG per controller sub-stage of the first SUCCESSFUL '
                            'episode into <dir>/frames/ (plus a frames.tsv manifest)')
    parser.add_argument('--keyframe-granularity', choices=['phase', 'event'], default='event',
                       help="Sub-stage granularity: 'phase' = one frame per task-level phase "
                            "(level 2/4/5 composed tasks); 'event' = additionally one per atomic "
                            'state-machine transition (level-1 single-action tasks)')
    parser.add_argument('--keyframe-min-steps', type=int, default=0,
                       help='Skip a sub-stage boundary that lands within this many simulation '
                            'steps of the previous saved frame')
    parser.add_argument('--keyframe-keep-attempts', action='store_true',
                       help='Keep the frames of failed attempts for inspection')
    parser.add_argument('--paper-replay-full', action='store_true',
                       help='In replay mode, play the whole recording instead of stopping at the '
                            'first sustained success. A figure strip needs the motion that '
                            'continues past the gate (level-5 shake/pour keep going after the lift)')
    parser.add_argument('--keyframe-every', type=int, default=0,
                       help='Also capture a frame every N simulation steps. Needed in replay mode, '
                            'where a scripted controller never advances its phases, so there are no '
                            'stage boundaries to key on (0 = off)')
    parser.add_argument('--keyframe-warmup', type=int, default=30,
                       help='Simulation steps to let RTX converge before the first keyframe is '
                            'read back (default: 30). A frame grabbed at step 1 is half-denoised.')
    parser.add_argument('--camera-probe', action='append', default=None,
                       metavar='EX,EY,EZ,TX,TY,TZ,FOCAL',
                       help='Framing scout: render the scene at rest from each of these '
                            'eye/target/focal specs, write them to --keyframe-dir/probe/ and '
                            'exit. Repeatable. One simulator launch tries every candidate, '
                            'instead of one launch (and one whole episode) per candidate.')
    parser.add_argument('--probe-prims', nargs='+', default=[], metavar='PRIM_PATH',
                       help='Log these prims\' world positions during --camera-probe, so the next '
                            'candidate can be aimed at measured coordinates')
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up simulation app based on arguments
simulation_config = {
    # Respect the CLI switch and allow remote/headless runners to opt in via
    # an environment variable without needing a virtual X server.
    "headless": args.headless or os.environ.get("LABUTOPIA_HEADLESS", "").lower()
    in {"1", "true", "yes"},
    "extra_args": ["--/rtx/raytracing/fractionalCutoutOpacity=true"],
}
if args.renderer:
    simulation_config["renderer"] = args.renderer
if args.render_width:
    simulation_config["width"] = args.render_width
if args.render_height:
    simulation_config["height"] = args.render_height

simulation_app = SimulationApp(simulation_config)

import hydra
import subprocess
import threading
from omegaconf import OmegaConf
import cv2
import numpy as np

import omni
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from isaacsim.core.utils import extensions
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
import carb

extensions.enable_extension("omni.physx.bundle")
extensions.enable_extension("omni.usdphysics.ui")

if args.renderer:
    # SimulationApp selects the renderer at startup; this setting makes the RTX
    # mode explicit as well, which avoids inheriting a stale interactive setting.
    carb.settings.get_settings().set(
        "/rtx/rendermode",
        "PathTracing" if args.renderer == "PathTracing" else "RaytracedLighting",
    )

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller
from robots.franka.rmpflow_controller import RMPFlowController
from controllers.atomic_actions.atomic_base_controller import AtomicBaseController
from experiments.paper.keyframes import KeyframeRecorder, frame_to_bgr
from experiments.robots.registry import register_experimental_robots
from experiments.support.physics_scene_steps import synchronize_physics_scene_steps

register_experimental_robots()


def _convert_to_h264(src_path: str, is_success: bool):
    """Re-encode a video to H264 in-place using ffmpeg (runs in background thread)."""
    if is_success:
        tmp_path = src_path + ".success.h264.tmp.mp4"
        dest_path = src_path.replace(".mp4", "_success.mp4")
    else:
        tmp_path = src_path + ".failure.h264.tmp.mp4"
        dest_path = src_path.replace(".mp4", "_failure.mp4")
    logger.info(f"Converting video to H264: {src_path}")
    ret = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", src_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            tmp_path,
        ],
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if ret.returncode == 0:
        os.replace(tmp_path, dest_path)
        os.remove(src_path)
        logger.success(f"Video saved: {dest_path}")
    else:
        stderr_msg = ret.stderr.decode(errors="replace") if ret.stderr else ""
        logger.error(f"H264 conversion failed for: {src_path}\n{stderr_msg}")
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


_convert_threads: list[threading.Thread] = []


def release_and_convert(writer: cv2.VideoWriter, output_path: str, is_success: bool):
    """Release cv2 writer and kick off H264 conversion in a background thread."""
    writer.release()
    t = threading.Thread(target=_convert_to_h264, args=(output_path, is_success), daemon=False)
    t.start()
    _convert_threads.append(t)


def _round_values(values):
    try:
        return [round(float(value), 4) for value in values]
    except (TypeError, ValueError):
        return values


def _object_quaternion(state, task):
    object_path = state.get("object_path")
    if not object_path:
        return None
    pose = task.object_utils.get_world_pose(object_path)
    if not pose:
        return None
    return [round(float(value), 3) for value in pose["orientation"]]


def _look_at_quaternion(eye: list[float], target: list[float]) -> list[float]:
    """Return the scalar-first quaternion expected by Isaac Sim's Camera API."""
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye),
        Gf.Vec3d(*target),
        Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()
    quat = matrix.ExtractRotation().GetQuat()
    imaginary = quat.GetImaginary()
    return [float(quat.GetReal())] + [float(imaginary[i]) for i in range(3)]


def _add_paper_background(
    usd_path: str,
    translation: list[float],
    rotation: list[float],
    scale: float,
    room_scale: float,
    hide_image_planes: bool = False,
    disable_dome_lights: bool = False,
):
    """Reference a publication-only backdrop and apply a stable world transform."""
    usd_path = os.path.abspath(usd_path)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"Paper background USD does not exist: {usd_path}")
    if scale <= 0 or room_scale <= 0:
        raise ValueError("--paper-background-scale must be positive")

    prim_path = "/World/PaperBackdrop"
    prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xformable = UsdGeom.Xformable(prim)
    # Keep the backdrop transform independent from any root ops authored in the asset.
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in rotation]))
    xformable.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    # lab_003's lab_015 room is authored in millimetres while its stage is
    # declared in metres. Apply the optional correction to that child only so
    # its walls/cabinets sit at a physically meaningful distance from the pick
    # table; the table and vessels retain their original scale.
    stage = omni.usd.get_context().get_stage()
    disabled_dome_count = 0
    if disable_dome_lights:
        # lab_001 has an authored city panorama DomeLight. Zero all authored
        # domes for a geometry-only paper frame; the neutral rect/sphere lights
        # remain active and illuminate the real lab meshes.
        for stage_child in Usd.PrimRange(stage.GetPseudoRoot()):
            if stage_child.IsA(UsdLux.DomeLight):
                UsdLux.DomeLight(stage_child).CreateIntensityAttr(0.0)
                disabled_dome_count += 1
    room_prim = stage.GetPrimAtPath(f"{prim_path}/lab_015")
    if room_prim.IsValid() and room_scale != 1.0:
        room_xform = UsdGeom.Xformable(room_prim)
        # The room may already author xformOp:scale; use a suffixed op so the
        # correction composes cleanly with the asset's existing transform stack.
        room_xform.AddScaleOp(opSuffix="room_fix").Set(
            Gf.Vec3f(float(room_scale), float(room_scale), float(room_scale))
        )
    # A backdrop is visual-only. Disable any colliders/rigid bodies authored by
    # a reusable lab prop so it cannot perturb the pick trajectory.
    collision_count = 0
    hidden_image_plane_count = 0
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
            collision_count += 1
        if hide_image_planes and child.IsA(UsdGeom.Mesh):
            material = UsdShade.MaterialBindingAPI(child).GetDirectBinding().GetMaterial()
            # lab_003's material_109 is the four-mesh city panorama (image_0029.jpg).
            if material and material.GetPath().name == "material_109":
                UsdGeom.Imageable(child).MakeInvisible()
                hidden_image_plane_count += 1
    logger.info(
        "Paper background referenced: {} -> {} (translation={}, rotation={}, scale={}, room_scale={}, disabled_colliders={}, hidden_image_planes={}, disabled_domes={})",
        usd_path,
        prim_path,
        [round(float(v), 3) for v in translation],
        [round(float(v), 3) for v in rotation],
        round(float(scale), 4),
        round(float(room_scale), 6),
        collision_count,
        hidden_image_plane_count,
        disabled_dome_count,
    )
    return prim


_USD_LIGHT_TYPE_NAMES = {
    "SphereLight",
    "RectLight",
    "DistantLight",
    "DomeLight",
    "DiskLight",
    "CylinderLight",
}


def _hide_prims_for_render(prim_paths: list[str]) -> None:
    """Make prims invisible without touching physics.

    Visibility is a render-only property, so colliders authored on (or under) the
    hidden prim keep supporting the scene. That is what lets a publication frame
    swap lab_001's work surface for the lab_003 bench while every arm still rests
    on — and every beaker still lands on — the exact collider the grasp was tuned
    against.
    """
    if not prim_paths:
        return
    stage = omni.usd.get_context().get_stage()
    hidden = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logger.warning(f"--paper-hide-prims: {prim_path} is not on the stage; skipping")
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            logger.warning(f"--paper-hide-prims: {prim_path} is not imageable; skipping")
            continue
        imageable.MakeInvisible()
        # A hidden light must also stop emitting. Visibility alone is honoured by
        # RTX, but the lighting randomizer re-authors intensity every episode, so
        # zeroing it here keeps a duplicated backdrop key light from doubling the
        # exposure of the real scene lights.
        if prim.HasAPI(UsdLux.LightAPI) or prim.GetTypeName() in _USD_LIGHT_TYPE_NAMES:
            UsdLux.LightAPI(prim).CreateIntensityAttr(0.0)
        hidden.append(prim_path)
    logger.info("Hidden for render only (colliders kept): {}", hidden)


def _save_clean_png(state: dict, cfg, output_dir: str, camera_index: int, episode_num: int, success: bool) -> str | None:
    """Write one camera RGB frame without the diagnostic video labels."""
    frames = list((state.get("camera_display") or {}).values())
    if camera_index < 0 or camera_index >= len(frames):
        logger.warning(f"PNG camera index {camera_index} is out of range ({len(frames)} camera frame(s))")
        return None
    image = np.asarray(frames[camera_index])
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[2] < 3:
        logger.warning(f"PNG frame has unsupported shape {image.shape}")
        return None
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    status = "success" if success else "failure"
    filename = f"episode_{int(episode_num):04d}_{status}.png"
    output_path = os.path.abspath(os.path.join(output_dir, filename))
    if not cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        logger.error(f"Failed to write PNG: {output_path}")
        return None
    logger.info(f"PNG saved: {output_path} ({image.shape[1]}x{image.shape[0]})")
    return output_path

def _run_camera_probe(specs: list[str], world, task, output_dir: str | None, camera_index: int) -> None:
    """Render the scene at rest from each candidate camera and write the frames.

    Framing a publication figure is iterative, and each candidate would otherwise
    cost a full simulator launch plus a full episode. The scene at reset is enough
    to judge composition, so try every candidate in one launch.
    """
    if not output_dir:
        raise ValueError("--camera-probe requires --keyframe-dir")
    if not task.cameras:
        raise ValueError("--camera-probe requires at least one camera")
    probe_dir = os.path.abspath(os.path.join(output_dir, "probe"))
    os.makedirs(probe_dir, exist_ok=True)
    # World positions of the prims the figure has to show, so the next candidate
    # can be aimed at real coordinates instead of guessed ones.
    for prim_path in (args.probe_prims or []):
        center = task.object_utils.get_geometry_center(object_path=prim_path)
        if center is None:
            center = task.object_utils.get_object_xform_position(object_path=prim_path)
        logger.info(
            f"[camera-probe] {prim_path} -> "
            f"{None if center is None else np.round(np.asarray(center, dtype=float), 3).tolist()}"
        )
    camera = task.cameras[0]
    for index, spec in enumerate(specs):
        values = [float(v) for v in spec.replace(" ", "").split(",")]
        if len(values) != 7:
            raise ValueError(f"--camera-probe expects EX,EY,EZ,TX,TY,TZ,FOCAL; got {spec!r}")
        eye, target, focal = values[:3], values[3:6], values[6]
        camera.set_world_pose(
            position=np.asarray(eye, dtype=float),
            orientation=np.asarray(_look_at_quaternion(eye, target), dtype=float),
            camera_axes="usd",
        )
        camera.set_focal_length(focal)
        # A few render ticks so RTX resolves the new view before it is read back.
        state = None
        for _ in range(12):
            world.step(render=True)
            state = task.step() or state
        image = frame_to_bgr(state, camera_index) if state is not None else None
        if image is None:
            logger.warning(f"[camera-probe] candidate {index} produced no frame")
            continue
        path = os.path.join(probe_dir, f"probe_{index:02d}.png")
        cv2.imwrite(path, image)
        logger.success(f"[camera-probe] {path} eye={eye} target={target} focal={focal}")


def main():
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    hydra.initialize_config_dir(config_dir=str(config_dir), job_name=args.config_name)
    compose_overrides = []
    production_config_dir = REPO_ROOT / "config"
    if config_dir.resolve() != production_config_dir.resolve():
        compose_overrides.append(f"hydra.searchpath=[file://{production_config_dir}]")
    cfg = hydra.compose(config_name=args.config_name, overrides=compose_overrides)
    if (args.paper_lighting or args.paper_camera or args.seed is not None
            or args.paper_fixed_phase_order):
        # Some legacy level-1 configs are structured DictConfigs without a
        # lighting/environment/seed key. Publication-only overrides may add those
        # keys in memory; the source YAML remains untouched.
        OmegaConf.set_struct(cfg, False)
    if args.paper_camera:
        # A dedicated static camera, authored fresh under /World, replaces every
        # configured camera. Overriding cfg.cameras[0] is not enough for level-5
        # scenes: all three of their cameras are parented to the Ridgebase, so a
        # world pose written onto one is undone the moment the base drives off.
        if args.camera_look_at is None:
            raise ValueError("--paper-camera requires --camera-look-at")
        if not cfg.cameras:
            raise ValueError("--paper-camera requires at least one configured camera to name")
        base_camera = cfg.cameras[0]
        cfg.cameras = OmegaConf.create(
            [
                {
                    "prim_path": args.paper_camera_prim,
                    # Keep the first camera's name so anything that keys observations
                    # by camera name still resolves.
                    "name": str(base_camera.name),
                    "resolution": list(base_camera.resolution),
                    "image_type": "rgb",
                    "focal_length": float(getattr(base_camera, "focal_length", 2.71)),
                    "translation": [0.0, 0.0, 0.0],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "clipping_range": [0.05, 200.0],
                }
            ]
        )
        logger.info(f"Paper camera: single static camera at {args.paper_camera_prim}")
    if args.paper_ood_materials:
        # Level-3 binds `materials` during collection and `test_materials` only in
        # infer mode (base_task.setup_materials). Swapping the lists here renders
        # the OOD half of an ID/OOD figure pair from the same collect config, so
        # both halves share one scripted rollout and one camera.
        material_paths = getattr(getattr(cfg, "task", None), "material_paths", None)
        if not material_paths:
            raise ValueError(
                f"--paper-ood-materials: {args.config_name} has no task.material_paths"
            )
        swapped = 0
        for mat_cfg in material_paths:
            test_materials = getattr(mat_cfg, "test_materials", None)
            if test_materials:
                mat_cfg.materials = list(test_materials)
                swapped += 1
        if swapped == 0:
            raise ValueError(
                f"--paper-ood-materials: no test_materials declared in {args.config_name}"
            )
        logger.info(f"OOD materials bound for {swapped} material_paths entry/entries")
    if args.paper_fixed_phase_order:
        OmegaConf.set_struct(cfg, False)
        cfg.task.fixed_phase_sequence = True
        logger.info("Canonical (unshuffled) sub-task order requested for this render")
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.single_camera is not None:
        configured_cameras = list(cfg.cameras)
        if args.single_camera < 0 or args.single_camera >= len(configured_cameras):
            raise ValueError(
                f"--single-camera index {args.single_camera} is out of range "
                f"for {len(configured_cameras)} configured camera(s)"
            )
        cfg.cameras = OmegaConf.create([configured_cameras[args.single_camera]])
    if args.render_width is not None or args.render_height is not None:
        if args.render_width is not None and args.render_width <= 0:
            raise ValueError("--render-width must be positive")
        if args.render_height is not None and args.render_height <= 0:
            raise ValueError("--render-height must be positive")
        for camera_cfg in cfg.cameras:
            current_width, current_height = list(camera_cfg.resolution)
            camera_cfg.resolution = [
                args.render_width if args.render_width is not None else current_width,
                args.render_height if args.render_height is not None else current_height,
            ]
    if args.camera_look_at is not None:
        eye = list(args.camera_look_at[:3])
        target = list(args.camera_look_at[3:])
        if not cfg.cameras:
            raise ValueError("--camera-look-at requires at least one configured camera")
        cfg.cameras[0].translation = eye
        cfg.cameras[0].orientation = _look_at_quaternion(eye, target)
        # A publication camera must not move. Several level-3 configs attach a
        # per-episode `randomize` block (±5 cm / ±0.02 rad / ±0.5 focal) to their
        # cameras; leaving it on would jitter the framing between the ID and OOD
        # halves of a comparison figure.
        if getattr(cfg.cameras[0], "randomize", None) is not None:
            OmegaConf.set_struct(cfg, False)
            cfg.cameras[0].randomize = None
            logger.info("[paper-camera] per-episode camera randomization disabled")
    if args.camera_focal_length is not None:
        if args.camera_focal_length <= 0:
            raise ValueError("--camera-focal-length must be positive")
        if not cfg.cameras:
            raise ValueError("--camera-focal-length requires at least one configured camera")
        cfg.cameras[0].focal_length = args.camera_focal_length
    if args.save_png_dir and (args.png_camera < 0 or args.png_camera >= len(cfg.cameras)):
        raise ValueError(
            f"--png-camera index {args.png_camera} is out of range for {len(cfg.cameras)} configured camera(s)"
        )
    if args.paper_lighting:
        # Keep the authored lab lights, but give them one consistent neutral
        # exposure/temperature and add a soft HDR fill for a legible background.
        cfg.lighting = OmegaConf.create(
            {
                "enabled": True,
                "scenario": "standard_lab",
                # Pin the authored key/fill lights to a restrained studio rig.
                # The USD scene contains high-intensity lights intended for the
                # interactive viewport; those clip the light arm materials in a
                # publication frame unless their intensity and exposure are reset.
                "randomize_intensity": True,
                "intensity_range": [1150.0, 1150.0],
                "shared_exposure": True,
                "exposure_range": [0.45, 0.45],
                "shared_color_temperature": True,
                "color_temp_range": [5400.0, 5400.0],
                "randomize_position": False,
                "randomize_rotation": False,
            }
        )
        if args.paper_no_hdr:
            cfg.environment = None
            logger.info("Publication lighting enabled: fixed 5400 K key lights (HDR dome disabled)")
        else:
            cfg.environment = OmegaConf.create(
                {
                    "hdr_path": "assets/hdr/lab_003/environment.exr",
                    "intensity": 550.0,
                    "exposure": -0.2,
                    "randomize_each_episode": False,
                }
            )
            logger.info("Publication lighting enabled: fixed 5400 K key lights + lab_003 HDR fill")
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be at least 1")
        cfg.max_episodes = args.max_episodes
    if args.max_attempts is not None and args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.physics_only:
        if cfg.mode != "collect":
            raise ValueError("--physics-only is a collect-mode diagnostic, not dataset replay")
        if args.save_png_dir or args.keyframe_dir or args.camera_probe:
            raise ValueError("--physics-only cannot capture images")
        cfg.cameras = []
        cfg.collector.type = "mock"
        logger.warning("PHYSICS-ONLY DIAGNOSTIC: no images or acceptance dataset are produced")
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")
    logger.info(f"Config loaded: {args.config_name}, run dir: {cfg.multi_run.run_dir}")

    # Optional global RNG seed. Off unless the config sets `seed`, so existing
    # runs are unchanged. With it, two runs of the same config draw the same
    # object placements / material cycle / lighting samples — which is what makes
    # an A/B (e.g. baseline vs unseen-lighting) a controlled comparison instead
    # of a comparison of two different random scene sequences.
    _seed = getattr(cfg, "seed", None)
    if _seed is not None:
        random.seed(int(_seed))
        np.random.seed(int(_seed))
        logger.info(f"[seed] global RNG seeded with {int(_seed)}")

    # Opt-in contact substeps keep the observation/action clock unchanged.
    # Thin grasp contacts need finer integration than a 60 Hz control loop.
    time_steps = {}
    physics_substeps = int(getattr(cfg, "physics_substeps", 1))
    if physics_substeps < 1 or physics_substeps > 8:
        raise ValueError("physics_substeps must be between 1 and 8")
    if physics_substeps > 1:
        time_steps = {"physics_dt": 1 / (60 * physics_substeps), "rendering_dt": 1 / 60}
        logger.info("Physics uses {} substeps per 60 Hz observation/action", physics_substeps)

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu", **time_steps)
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy", **time_steps)

    # Override configuration based on command line arguments
    if args.no_video or args.physics_only:
        save_video = False
        show_video = False
    else:
        save_video = True
        # A headless run can still render and save camera frames, but OpenCV has no
        # display server for imshow(). Keep remote diagnostics recording enabled
        # without trying to create a GUI window.
        show_video = not simulation_config["headless"]

    # Scene BEFORE robot. A scene USD may already contain the arm we want to
    # drive (the sim2real bench has both of its Frankas baked in); the robot
    # constructor only binds to an existing prim if that prim is already on the
    # stage, otherwise it references a fresh one and clobbers the authored base
    # transform. No shipped scene contains a robot, so this reorder is a no-op
    # for every pre-existing config.
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    if args.paper_background_usd:
        _add_paper_background(
            args.paper_background_usd,
            args.paper_background_translation,
            args.paper_background_rotation,
            args.paper_background_scale,
            args.paper_background_room_scale,
            args.paper_hide_image_planes,
            args.paper_no_hdr,
        )
    # After the backdrop so its own prims (e.g. a duplicate GroundPlane that would
    # z-fight with the scene's) can be targeted too.
    _hide_prims_for_render(list(args.paper_hide_prims))

    if physics_substeps > 1:
        logger.info(
            "Synchronized referenced physics scenes: {}",
            synchronize_physics_scene_steps(stage, 60 * physics_substeps),
        )

    robot_kwargs = {"position": np.array(cfg.robot.position)}
    if hasattr(cfg.robot, "default_joint_positions"):
        robot_kwargs["default_joint_positions"] = np.array(cfg.robot.default_joint_positions)
    if hasattr(cfg.robot, "usd_path"):
        robot_kwargs["usd_path"] = str(cfg.robot.usd_path)
    if hasattr(cfg.robot, "prim_path"):
        # Bind to an arm that already exists in the scene instead of spawning one.
        # Orientation is deliberately NOT passed: leaving it unset preserves the
        # base yaw authored in the scene, which RMPFlowController then reads via
        # get_world_pose() and feeds to set_robot_base_pose().
        robot_kwargs["prim_path"] = str(cfg.robot.prim_path)
    robot = create_robot(cfg.robot.type, **robot_kwargs)
    logger.info(f"Robot created: {cfg.robot.type} at {getattr(robot, 'prim_path', cfg.robot.get('prim_path', '/World/Franka'))}")

    # Configure gripper control mode if specified
    gripper_cfg = getattr(cfg.robot, "gripper", None)
    if gripper_cfg:
        mode = str(getattr(gripper_cfg, "control_mode", "position"))
        # backward compat: force_mode: true → control_mode: "force"
        if getattr(gripper_cfg, "force_mode", False) and mode == "position":
            mode = "force"
        if mode != "position":
            robot.set_gripper_control_mode(
                mode=mode,
                closing_force=float(getattr(gripper_cfg, "closing_force", 20.0)),
                closing_speed=float(getattr(gripper_cfg, "closing_speed", 0.2)),
            )

    ObjectUtils.get_instance(stage)

    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    # Some lab scenes author Camera1/Camera2 prims directly.  BaseTask preserves
    # those prims, so publication overrides must be applied after construction as
    # well as written into cfg (the latter covers cameras created from scratch).
    if args.camera_look_at is not None and task.cameras:
        task.cameras[0].set_world_pose(
            position=np.asarray(cfg.cameras[0].translation, dtype=float),
            orientation=np.asarray(cfg.cameras[0].orientation, dtype=float),
            camera_axes="usd",
        )
    if args.camera_focal_length is not None and task.cameras:
        task.cameras[0].set_focal_length(float(cfg.cameras[0].focal_length))

    # Position-only collection: make RMPFlow roll out an internal virtual robot
    # (ignore measured joint state) so its position targets advance at planned
    # speed while the real arm tracks them with pure position PD — the identical
    # control law replay and inference use. Must be set BEFORE the controller is
    # constructed (atomic controllers build their RMPFlowController in __init__).
    # Applies to collect AND to replay of a position-only dataset (the replay
    # config inherits the flag): controllers with scripted phases in replay
    # (e.g. pour's scripted pick) must drive the arm with the exact same
    # control law the data was collected with, or the hand-off joint state
    # diverges from what the recorded actions assume.
    if bool(getattr(cfg, "collect_position_only", False)):
        RMPFlowController.ignore_robot_state_updates = True
        AtomicBaseController.record_commanded_gripper = True
        logger.info("[collect_position_only] RMPFlow virtual-robot rollout ON, "
                    "velocity feed-forward stripped, gripper channel = commanded")

    # Long-horizon tasks (level4) chain many atomic actions, so per-episode
    # action noise compounds across the sequence; disable it via config.
    if not bool(getattr(cfg, "action_randomization", True)):
        AtomicBaseController.randomization_enabled = False
        logger.info("[action_randomization=false] per-episode action noise "
                    "neutralized (deterministic mid-range parameters)")

    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )

    keyframe_recorder = None
    if args.keyframe_dir:
        keyframe_recorder = KeyframeRecorder(
            output_dir=args.keyframe_dir,
            camera_index=args.png_camera,
            granularity=args.keyframe_granularity,
            min_steps=args.keyframe_min_steps,
            keep_attempts=args.keyframe_keep_attempts,
            warmup_steps=args.keyframe_warmup,
            every_steps=args.keyframe_every,
        )

    if args.paper_replay_full:
        if cfg.mode != "replay":
            raise ValueError("--paper-replay-full only applies to replay mode")
        task_controller.replay_run_to_end = True
        logger.info("Replay will play every recorded action, not just up to the success gate")

    video_writer = None
    video_output_path = None
    _replay_count_logged = False
    # An episode can end two ways: the controller returns done (handled in the
    # `if done:` branch below) or the task hits max_steps and only flips
    # reset_needed. The latter used to skip the whole end-of-episode block, so
    # timed-out episodes printed no failure reason and never cleared the
    # per-episode robot diagnostics (they leaked into every later episode).
    _episode_finalized = False
    _steps_since_reset = 0
    arm_diagnostics_enabled = bool(os.environ.get("LABUTOPIA_ARM_DIAGNOSTICS"))

    def _reset_episode_diagnostics(rb) -> None:
        if not arm_diagnostics_enabled:
            return
        _am = getattr(rb, "_argmax_diag", (None, None, None))
        logger.info(f"[EP-ARM-ERR] mode={cfg.mode} ep={task_controller.episode_num} "
                    f"max_arm_track_err={round(float(getattr(rb, '_ep_max_arm_err', 0.0)), 4)} "
                    f"at_frame={_am[0]}/{getattr(rb, '_frame_diag', 0)} joint={_am[1]} per_joint_err={_am[2]}")
        rb._ep_max_arm_err = 0.0
        rb._frame_diag = 0
        rb._argmax_diag = (None, None, None)
        rb._closed_once = False

    _arm_stiffness_scale = float(getattr(cfg, "arm_stiffness_scale", 1.0))

    def _apply_arm_gains():
        # Scale the arm position-drive gains so pure joint-position control (no
        # velocity feed-forward) tracks briskly instead of crawling. Re-applied
        # after each reset because re-init restores USD defaults.
        if _arm_stiffness_scale == 1.0:
            return
        try:
            ac = robot.get_articulation_controller()
            kps, kds = ac.get_gains()
            if kps is None:
                return
            kps = list(kps); kds = list(kds)
            for j in range(min(7, len(kps))):
                kps[j] = kps[j] * _arm_stiffness_scale
                kds[j] = kds[j] * (_arm_stiffness_scale ** 0.5)
            ac.set_gains(kps=kps, kds=kds)
            logger.info(f"[gains] arm kp x{_arm_stiffness_scale} -> kp[:7]={[round(float(k),1) for k in kps[:7]]}")
        except Exception as e:
            logger.warning(f"[gains] failed to scale arm gains: {e}")

    def _log_reset_object_poses(label: str) -> None:
        if not os.environ.get("LABUTOPIA_RESET_DEBUG"):
            return
        root_path = getattr(task, "current_obj_path", None)
        if not root_path:
            return
        poses = {}
        for path in (root_path, f"{root_path}/mesh"):
            pose = task.object_utils.get_world_pose(path)
            if pose is not None:
                poses[path] = np.round(pose["position"], 4).tolist()
        logger.info(f"[reset-debug] {label} poses={poses}")

    task.reset()
    # World.reset() can restore authored camera transforms, so apply the
    # publication pose once more after the initial task reset.  This is needed
    # for scenes that ship their own Camera1 prim.
    if args.camera_look_at is not None and task.cameras:
        task.cameras[0].set_world_pose(
            position=np.asarray(cfg.cameras[0].translation, dtype=float),
            orientation=np.asarray(cfg.cameras[0].orientation, dtype=float),
            camera_axes="usd",
        )
        logger.info(
            "[paper-camera] world pose position={} orientation={} target={}",
            [round(float(v), 4) for v in cfg.cameras[0].translation],
            [round(float(v), 4) for v in cfg.cameras[0].orientation],
            [round(float(v), 4) for v in args.camera_look_at[3:]],
        )
    if args.camera_focal_length is not None and task.cameras:
        task.cameras[0].set_focal_length(float(cfg.cameras[0].focal_length))
    _apply_arm_gains()
    if keyframe_recorder is not None:
        # The first episode never passes through the reset branch below.
        keyframe_recorder.begin_episode()
    if args.camera_probe:
        _run_camera_probe(args.camera_probe, world, task, args.keyframe_dir, args.png_camera)
        simulation_app.close()
        return
    if os.environ.get("LABUTOPIA_PICK_DEBUG"):
        _debug_kp, _debug_kd = robot.get_articulation_controller().get_gains()
        logger.info(
            f"[ROBOT-DOF] names={list(robot.dof_names or [])} "
            f"gripper_indices={list(robot.get_gripper_joint_indices())} "
            f"kp={[round(float(v), 3) for v in _debug_kp]} "
            f"kd={[round(float(v), 3) for v in _debug_kd]}"
        )

    while simulation_app.is_running():
        if args.physics_only:
            for _ in range(physics_substeps):
                world.step(render=False)
        else:
            world.step(render=True)

        if world.is_stopped():
            task_controller.reset_needed = True

        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    release_and_convert(
                        video_writer,
                        video_output_path,
                        task_controller.is_success(),
                    )
                    video_writer = None
                    video_output_path = None

                # Determine how many episodes to run depending on mode.
                # Replay defaults to the whole dataset — the top-level
                # `max_episodes` is a collect/infer knob and is NOT a cap here.
                # Set `replay.max_episodes` to cap a replay run (or
                # `replay.episode_indices` to pick a subset); the effective
                # count is logged once so a run never silently differs from
                # what the config appears to ask for.
                if cfg.mode == "replay":
                    n_dataset = len(task_controller._replay_loader)
                    _replay_cap = getattr(getattr(cfg, "replay", None), "max_episodes", None)
                    max_episodes = min(n_dataset, int(_replay_cap)) if _replay_cap else n_dataset
                    if not _replay_count_logged:
                        source = ("replay.max_episodes" if _replay_cap
                                  else f"whole dataset (top-level max_episodes={cfg.max_episodes} "
                                       "does not apply to replay)")
                        logger.info(f"[Replay] dataset has {n_dataset} episodes; running "
                                    f"{max_episodes} — capped by {source}")
                        _replay_count_logged = True
                else:
                    max_episodes = cfg.max_episodes

                # An episode that ended on max_steps never reached the `if done:`
                # branch, so finalize it here: report why it failed and clear the
                # per-episode diagnostics before they leak into the next episode.
                # `_steps_since_reset` guards the very first pass (scene setup, and
                # replay's initial reset), where no episode has run yet — without it
                # every run logged a spurious "Episode 0 failed" and every parsed
                # stat came out one failure too high.
                if not _episode_finalized and _steps_since_reset > 0:
                    if keyframe_recorder is not None:
                        keyframe_recorder.end_episode(None, False)
                    # Log the standard "failed" line too: a timed-out episode is a
                    # failed one, and the campaign log parsers count these lines.
                    logger.warning(f"Episode {task_controller.episode_num} failed. "
                                   f"(ended on max_steps="
                                   f"{getattr(getattr(cfg, 'task', None), 'max_steps', '?')})")
                    _reset_episode_diagnostics(robot)
                    task_controller.print_failure_reason()
                episode_just_finished = _steps_since_reset > 0
                _episode_finalized = False
                _steps_since_reset = 0

                # Check if we've completed all episodes BEFORE setting up the next one.
                # In infer/replay `episode_num` is only incremented by reset(), which
                # has not run yet for the episode that just ended — count it here, or
                # the run does max_episodes+1 episodes and drops the last one from the
                # stats. (In collect, `episode_num` is the data collector's count of
                # WRITTEN episodes, which is already final.)
                # Real collection deliberately caps SUCCESSFUL datasets, so failed
                # attempts do not count. Smoke tests use MockCollector and need a
                # finite attempt cap; its current failed attempt is not reflected in
                # _episode_num until reset(), hence the explicit +1 here.
                count_attempts = bool(
                    getattr(getattr(task_controller, "data_collector", None), "counts_attempts", False)
                )
                current_episode = int(episode_just_finished)
                episodes_done = (
                    task_controller.episode_num + current_episode
                    if cfg.mode != "collect" or count_attempts
                    else task_controller.episode_num
                )
                attempts_done = task_controller._episode_num + current_episode
                attempts_exhausted = (
                    args.max_attempts is not None and attempts_done >= args.max_attempts
                )
                if (
                    episodes_done >= max_episodes
                    or attempts_exhausted
                    or getattr(task_controller, "_replay_done", False)
                ):
                    if cfg.mode == "collect" and episode_just_finished:
                        # A real collector reaches max_episodes as soon as the final
                        # successful sample is written. Shutdown therefore happens
                        # before reset(), which normally updates and prints the
                        # controller's attempt statistics. Emit that missing final
                        # line without resetting task/controller state during teardown.
                        final_attempts = task_controller._episode_num + 1
                        final_successes = task_controller.success_count + int(
                            task_controller.is_success()
                        )
                        final_rate = final_successes / max(final_attempts, 1) * 100.0
                        logger.info(
                            "Episode Stats: Success Rate = "
                            f"{final_successes}/{final_attempts} ({final_rate:.2f}%)"
                        )
                    if cfg.mode != "collect" and not getattr(task_controller, "_is_initial_replay_reset", False):
                        # reset() is what counts the episode and prints "Episode Stats",
                        # so the final episode is otherwise missing from the tally.
                        task_controller.reset()
                    if attempts_exhausted and episodes_done < max_episodes:
                        logger.info(
                            f"Attempt limit reached ({attempts_done}/{args.max_attempts}). "
                            "Shutting down."
                        )
                    else:
                        logger.info(f"All {max_episodes} episodes completed. Shutting down.")
                    if keyframe_recorder is not None and not keyframe_recorder.completed:
                        logger.error(
                            "[keyframes] no successful episode was captured; "
                            f"nothing written to {args.keyframe_dir}"
                        )
                    task_controller.close()
                    # Join the ffmpeg transcode threads BEFORE tearing down the
                    # app: closing first killed the last episode's conversion and
                    # left a half-written *.tmp.mp4 instead of its video.
                    for t in _convert_threads:
                        t.join()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break

                # In replay mode: restore scene FIRST, then reset controller
                # This ensures correct episode numbering and environment setup
                if cfg.mode == "replay":
                    _log_reset_object_poses("before task reset")
                    init_state = task_controller.get_current_init_state()
                    task.reset_with_init_state(init_state)
                    if args.paper_ood_materials:
                        # reset_with_init_state restores the material the episode
                        # was COLLECTED with, which would undo the OOD swap. Re-bind
                        # afterwards so an ID/OOD figure pair can be rendered from
                        # the same recorded rollout — identical motion, only the
                        # appearance differs, which is exactly what the pair claims.
                        task.apply_materials()
                    _log_reset_object_poses("after task reset")
                    task_controller.reset()
                else:
                    _log_reset_object_poses("before task reset")
                    task_controller.reset()
                    task.reset()
                    _log_reset_object_poses("after task reset")
                _apply_arm_gains()
                if keyframe_recorder is not None:
                    keyframe_recorder.begin_episode()

                continue

            state = task.step()
            if state is None:
                continue

            action, done, is_success = task_controller.step(state)
            _steps_since_reset += 1
            if keyframe_recorder is not None and not done:
                # After the controller step, so the sub-stage counters already
                # reflect this frame's transition; `state` still holds the render
                # taken BEFORE the new sub-action moved anything, i.e. the pose the
                # sub-stage that just finished achieved.
                keyframe_recorder.observe(task_controller, state)
            if action is not None:
                # collect_position_only: apply the scripted action as a PURE joint
                # POSITION command (drop RMPFlow's velocity/effort feed-forward), so
                # collection uses the exact same control law as replay & inference
                # (which only have joint positions). Makes the recorded joints a
                # complete control signal -> collect == replay == policy execution.
                if bool(getattr(cfg, "collect_position_only", False)) \
                        and getattr(action, "joint_positions", None) is not None:
                    # Position-only control law in collect AND in the scripted
                    # phases of replay (recorded replay actions carry no
                    # velocities, so this is a no-op for them). Velocity-ONLY
                    # actions (joint_positions=None, e.g. the pour controller's
                    # wrist rotation on a velocity-switched DOF) are the task's
                    # defining actuation — leave them intact.
                    action.joint_velocities = None
                    action.joint_efforts = None
                # Sync BEFORE potentially stripping the finger channels below —
                # velocity/force modes read the finger target to decide open/close.
                robot.sync_gripper_from_action(action)
                # Replay parity for velocity/force gripper modes: during collect
                # the scripted (RMPFlow) action position-commands only the 7 arm
                # DOFs — fingers are driven solely by apply_gripper_effort(). The
                # replayed 9-DOF action would ALSO position-slam the fingers
                # (drive spring to 0 crushes/ejects what collect's gentle
                # velocity close held), so strip the finger channels and let
                # apply_gripper_effort() reproduce collect's exact actuation.
                if (cfg.mode == "replay"
                        and getattr(robot, "_gripper_control_mode", "position") != "position"
                        and getattr(action, "joint_positions", None) is not None
                        and len(action.joint_positions) > 7):
                    # None = "leave this DOF uncommanded" (dims must stay = DOF count)
                    action.joint_positions = list(action.joint_positions[:7]) + \
                        [None] * (len(action.joint_positions) - 7)
                    if action.joint_velocities is not None and len(action.joint_velocities) > 7:
                        action.joint_velocities = list(action.joint_velocities[:7]) + \
                            [None] * (len(action.joint_velocities) - 7)
                robot.get_articulation_controller().apply_action(action)
            robot.apply_gripper_effort()
            # Optional collect-vs-replay diagnostics. Disabled by default because
            # the per-frame queries and traces are useful for debugging, not normal
            # collection or evaluation.
            if arm_diagnostics_enabled:
                try:
                    # Per-frame arm-tracking error (arm-motion consistency). Track
                    # the frame and joint where the maximum occurs.
                    robot._frame_diag = getattr(robot, "_frame_diag", 0) + 1
                    if action is not None and getattr(action, "joint_positions", None) is not None:
                        _action_positions = list(action.joint_positions)
                        _action_indices = getattr(action, "joint_indices", None)
                        if _action_indices is None:
                            _action_indices = range(len(_action_positions))
                        _command_by_dof = {
                            int(index): float(value)
                            for index, value in zip(_action_indices, _action_positions)
                            if value is not None
                        }
                        _arm_indices = robot.get_arm_joint_indices()
                        _actual_positions = robot.get_joint_positions()
                        _perr = [
                            abs(float(_actual_positions[index]) - _command_by_dof[index])
                            for index in _arm_indices
                            if index in _command_by_dof
                        ]
                    else:
                        _perr = []
                    if _perr:
                        _err = max(_perr)
                        robot._last_err = _err
                        if _err > getattr(robot, "_ep_max_arm_err", 0.0):
                            robot._ep_max_arm_err = _err
                            robot._argmax_diag = (
                                robot._frame_diag,
                                _perr.index(_err),
                                [round(error, 3) for error in _perr],
                            )
                        if not getattr(robot, "_closed_once", False) and robot._frame_diag % 25 == 0:
                            logger.info(
                                f"[APPROACH] mode={cfg.mode} ep={task_controller.episode_num} "
                                f"frame={robot._frame_diag} arm_err={round(_err, 4)} "
                                f"max_joint={_perr.index(_err)} "
                                f"per_joint={[round(error, 3) for error in _perr]}"
                            )

                    # Track grasp/release transitions in both position and effort
                    # gripper modes.
                    _gs = 0
                    try:
                        if int(robot.get_gripper_state()) == 1:
                            _gs = 1
                    except (TypeError, ValueError):
                        pass
                    if (
                        action is not None
                        and getattr(action, "joint_positions", None) is not None
                        and len(action.joint_positions) >= 8
                    ):
                        _legacy_gripper_target = action.joint_positions[7]
                        if _legacy_gripper_target is not None and float(_legacy_gripper_target) <= 0.02:
                            _gs = 1
                    _prev = getattr(robot, "_prev_gs_diag", 0)
                    _ft = getattr(robot, "_finger_trace_left", 0)
                    if _gs == 1 and _prev == 0:
                        robot._finger_trace_left = 60
                    elif _ft > 0:
                        robot._finger_trace_left = _ft - 1
                        if _ft % 5 == 0:
                            _joint_positions = robot.get_joint_positions()
                            _gripper_indices = robot.get_gripper_joint_indices()
                            _fp = {
                                str(robot.dof_names[index]): round(float(_joint_positions[index]), 4)
                                for index in _gripper_indices
                            }
                            logger.info(
                                f"[FINGER] mode={cfg.mode} ep={task_controller.episode_num} "
                                f"t-close={60 - _ft} fingers={_fp}"
                            )

                    if _gs == 1 and _prev == 0:  # grasp instant
                        robot._closed_once = True
                        logger.info(
                            f"[GRASP-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"obj_pos={_round_values(state.get('object_position'))} "
                            f"obj_quat={_object_quaternion(state, task)} "
                            f"arm_err_now={round(float(getattr(robot, '_last_err', 0.0)), 4)} "
                            f"arm_err_peak_approach={round(float(robot._ep_max_arm_err), 4)}"
                        )
                    elif _gs == 0 and _prev == 1:  # release instant
                        logger.info(
                            f"[RELEASE-DIAG] mode={cfg.mode} ep={task_controller.episode_num} "
                            f"obj_pos={_round_values(state.get('object_position'))} "
                            f"obj_quat={_object_quaternion(state, task)} "
                            f"arm_err_max={round(float(getattr(robot, '_ep_max_arm_err', 0.0)), 4)}"
                        )
                    robot._prev_gs_diag = _gs
                except Exception:
                    logger.exception("Arm diagnostics failed")
            if done:
                if is_success:
                    logger.success(f"Episode {task_controller.episode_num} succeeded.")
                else:
                    logger.warning(f"Episode {task_controller.episode_num} failed.")
                if keyframe_recorder is not None:
                    keyframe_recorder.end_episode(state, bool(is_success))
                if args.save_png_dir and (is_success or args.save_png_on_failure):
                    _save_clean_png(
                        state,
                        cfg,
                        args.save_png_dir,
                        args.png_camera,
                        task_controller.episode_num,
                        is_success,
                    )
                _reset_episode_diagnostics(robot)
                task_controller.print_failure_reason()
                task.on_task_complete(is_success)
                _episode_finalized = True
                continue

            if save_video or show_video:
                camera_images = []
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)

                if camera_images:
                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    if show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                    if save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            video_output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(video_output_path, fourcc, 60.0, (width, height))
                            if not video_writer.isOpened():
                                logger.error(f"Failed to open VideoWriter for {video_output_path}")
                                video_writer = None
                        if video_writer is not None:
                            video_writer.write(combined_img)


if __name__ == "__main__":
    main()
