"""
HDR panorama capture utility for Isaac Sim 4.x / 5.x.

Uses 6 standard 90-degree perspective cameras (cube map) and stitches
them into an equirectangular panorama.  Does NOT rely on any panoramic
projection attribute, so it works on every render mode and version.
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_S2 = 0.7071067811865476  # √2/2

# ---------------------------------------------------------------------------
# Cube face definitions  (Z-up world, matching Isaac Sim stage up-axis = Z)
# Default camera: forward = -Z, up = +Y.
# Each quaternion rotates that camera so it looks in the stated direction
# with the stated up vector.
# ---------------------------------------------------------------------------
_CUBE_FACES = [
    ("nz", (1.0,   0.0,   0.0,   0.0)),   # forward=-Z (down),  up=+Y
    ("pz", (0.0,   0.0,   1.0,   0.0)),   # forward=+Z (up),    up=+Y  (180° Y)
    ("px", (0.5,   0.5,  -0.5,  -0.5)),   # forward=+X,         up=+Z
    ("nx", (0.5,   0.5,   0.5,   0.5)),   # forward=-X,         up=+Z
    ("py", (_S2,   _S2,   0.0,   0.0)),   # forward=+Y,         up=+Z  (+90° X)
    ("ny", (0.0,   0.0,  -_S2,  -_S2)),   # forward=-Y,         up=+Z
]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _define_cube_camera(stage, prim_path, position, quat_wxyz):
    from pxr import UsdGeom, Gf
    cam = UsdGeom.Camera.Define(stage, prim_path)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    w, x, y, z = quat_wxyz
    xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    # 90° FOV: aperture = 2 × focal_length
    cam.CreateFocalLengthAttr(10.0)
    cam.CreateHorizontalApertureAttr(20.0)
    cam.CreateVerticalApertureAttr(20.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1e6))


def _remove_prim(stage, path):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        try:
            stage.RemovePrim(path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

_ANN_PRIORITY = ("HdrColor", "hdr", "LdrColor", "rgb")


def _attach_annotator(rep, rp):
    for name in _ANN_PRIORITY:
        try:
            ann = rep.annotators.get(name)
            ann.attach(rp)
            logger.info(f"Annotator '{name}' attached")
            return ann
        except Exception:
            continue
    raise RuntimeError(f"No annotator found. Tried: {_ANN_PRIORITY}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _step(rep, app, n=1, rt_subframes: int = 0):
    """
    Advance the replicator orchestrator by *n* frames.

    rt_subframes > 0 forces RTX to render that many sub-frames per step,
    which is required for HdrColor data to be non-zero on the first capture.
    """
    orch = rep.orchestrator
    for _ in range(n):
        # Prefer step() with rt_subframes (Isaac Sim 4.x RTX requirement)
        if hasattr(orch, "step"):
            kwargs = {}
            if rt_subframes > 0:
                try:
                    orch.step(rt_subframes=rt_subframes, pause_timeline=False)
                except TypeError:
                    orch.step(pause_timeline=False)
            else:
                orch.step(pause_timeline=False)
        elif hasattr(orch, "run_until_complete"):
            orch.run_until_complete(num_frames=1)
        else:
            raise RuntimeError("No compatible orchestrator API")
        app.update()


# ---------------------------------------------------------------------------
# Cube-map → equirectangular stitching
# ---------------------------------------------------------------------------

def _stitch(faces, out_w, out_h):
    """
    Stitch 6 float32 cube-face arrays into an equirectangular panorama.

    Uses **Z-up** convention, matching Isaac Sim / USD stage up-axis = Z.
    Direction from pixel (lon, lat):
      dx = cos(lat)*sin(lon)
      dy = cos(lat)*cos(lon)   <- +Y = scene "forward" at lon=0
      dz = sin(lat)            <- Z = elevation (world up)

    Sampling formulae per face (sc,tc in [-1,+1]; +sc=right, +tc=up):
      px (forward=+X, right=-Y, up=+Z): sc = -dy/dx,   tc = dz/|dx|
      nx (forward=-X, right=+Y, up=+Z): sc =  dy/|dx|, tc = dz/|dx|
      py (forward=+Y, right=+X, up=+Z): sc =  dx/dy,   tc = dz/|dy|
      ny (forward=-Y, right=-X, up=+Z): sc = -dx/|dy|, tc = dz/|dy|
      pz (forward=+Z, right=-X, up=+Y): sc = -dx/dz,   tc = dy/|dz|
      nz (forward=-Z, right=+X, up=+Y): sc =  dx/|dz|, tc = dy/|dz|
    """
    S = next(iter(faces.values())).shape[0]
    C = next(iter(faces.values())).shape[2]
    out = np.zeros((out_h, out_w, C), dtype=np.float32)

    lon = (np.linspace(0, 1, out_w, endpoint=False) + 0.5 / out_w) * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.linspace(0, 1, out_h, endpoint=False) + 0.5 / out_h) * np.pi
    LON, LAT = np.meshgrid(lon, lat)

    # Z-up world direction vectors
    dx = np.cos(LAT) * np.sin(LON)
    dy = np.cos(LAT) * np.cos(LON)
    dz = np.sin(LAT)                   # Z = up
    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)

    # Avoid division by zero
    _eps = 1e-9
    rules = [
        ("px", (dx > 0) & (ax >= ay) & (ax >= az), -dy / (dx  + _eps),  dz / (ax + _eps)),
        ("nx", (dx < 0) & (ax >= ay) & (ax >= az),  dy / (ax  + _eps),  dz / (ax + _eps)),
        ("py", (dy > 0) & (ay > ax)  & (ay > az),   dx / (dy  + _eps),  dz / (ay + _eps)),
        ("ny", (dy < 0) & (ay > ax)  & (ay > az),  -dx / (ay  + _eps),  dz / (ay + _eps)),
        ("pz", (dz > 0) & (az > ax)  & (az > ay),  -dx / (dz  + _eps),  dy / (az + _eps)),
        ("nz", (dz < 0) & (az > ax)  & (az > ay),   dx / (az  + _eps),  dy / (az + _eps)),
    ]

    for name, mask, sc, tc in rules:
        if name not in faces:
            continue
        pu = np.clip(((sc + 1.0) * 0.5 * (S - 1)).astype(np.int32), 0, S - 1)
        pv = np.clip(((1.0 - tc) * 0.5 * (S - 1)).astype(np.int32), 0, S - 1)
        out[mask] = faces[name][pv[mask], pu[mask]]

    return out


# ---------------------------------------------------------------------------
# EXR writer
# ---------------------------------------------------------------------------

def _write_exr(data, output_dir, filename="environment.exr"):
    rgb = data[:, :, :3].astype(np.float32)
    if data.dtype == np.uint8:
        rgb /= 255.0
    if np.all(rgb == 0):
        logger.warning("All-black image — scene may have no light sources")
    path = os.path.join(output_dir, filename)
    try:
        import imageio
        imageio.v3.imwrite(path, rgb)
    except AttributeError:
        import imageio as iio
        iio.imwrite(path, rgb, format="exr")
    logger.info(f"EXR written: {path}  max={rgb.max():.4f}")
    return path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def capture_hdr_panorama(
    output_dir: str,
    position: Tuple[float, float, float] = (0.0, 0.0, 1.5),
    face_size: int = 1024,
    camera_prim_base: str = "/World/HDRCaptureCam",
    cleanup_cameras: bool = True,
    render_warmup_frames: int = 16,
) -> Optional[str]:
    """
    Capture an equirectangular HDR panorama (.exr) via a 6-face cube map.

    Args:
        output_dir:           Output directory for the .exr file.
        position:             World-space capture point (x, y, z).
        face_size:            Resolution of each cube face in pixels (NxN).
                              Output EXR will be (4N x 2N).
        camera_prim_base:     Base USD path for temporary cameras.
        cleanup_cameras:      Remove camera prims after capture.
        render_warmup_frames: Warm-up frames per face before capture.

    Returns:
        Absolute path to the saved .exr, or None on failure.
    """
    import omni.usd
    import omni.kit.app
    import omni.replicator.core as rep

    os.makedirs(output_dir, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    app = omni.kit.app.get_app()

    cube_faces = {}
    cam_paths = []
    result_path = None

    try:
        for face_name, quat in _CUBE_FACES:
            prim_path = f"{camera_prim_base}_{face_name}"
            cam_paths.append(prim_path)

            _define_cube_camera(stage, prim_path, position, quat)
            for _ in range(3):
                app.update()

            rp = rep.create.render_product(prim_path, (face_size, face_size))
            for _ in range(16):          # more updates so RTX initialises the RP
                app.update()

            ann = _attach_annotator(rep, rp)
            for _ in range(8):           # let annotator bind before first step
                app.update()

            # ── Adaptive warm-up ────────────────────────────────────────────
            # Step in small batches; stop as soon as the annotator returns
            # non-zero data.  Falls back to render_warmup_frames as the cap.
            _BATCH       = 8    # frames per probe
            _MAX_RETRIES = max(1, render_warmup_frames // _BATCH)
            rgb = None
            for attempt in range(_MAX_RETRIES):
                _step(rep, app, _BATCH, rt_subframes=32)
                data = ann.get_data()
                if data is None:
                    continue
                candidate = data[:, :, :3]
                if candidate.dtype == np.uint8:
                    candidate = candidate.astype(np.float32) / 255.0
                else:
                    candidate = candidate.astype(np.float32)
                if not np.all(candidate == 0):
                    rgb = candidate
                    logger.info(
                        f"Face '{face_name}' converged after "
                        f"{(attempt+1)*_BATCH} frames  max={rgb.max():.4f}"
                    )
                    break
            # ────────────────────────────────────────────────────────────────

            rp.destroy()

            if rgb is None:
                logger.warning(
                    f"Face '{face_name}': still all-zero after "
                    f"{_MAX_RETRIES*_BATCH} frames, skipping"
                )
                continue

            cube_faces[face_name] = rgb
            logger.info(f"Face '{face_name}' OK  max={rgb.max():.4f}")

        if not cube_faces:
            logger.error("No cube faces captured")
            return None

        logger.info(f"Stitching {len(cube_faces)}/6 faces → {face_size*4}x{face_size*2}")
        equirect = _stitch(cube_faces, face_size * 4, face_size * 2)
        result_path = _write_exr(equirect, output_dir)

    except Exception:
        logger.exception("capture_hdr_panorama failed")

    finally:
        if cleanup_cameras:
            for p in cam_paths:
                _remove_prim(stage, p)

    return result_path
