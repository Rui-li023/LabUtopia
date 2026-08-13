"""Calibrated camera intrinsics for the sim2real bench, and the USD conversion.

Single source of truth: the builder authors cameras from this table and the
verifier checks the rendered result against it. Pure math, no pxr import, so it
can be used outside Isaac Sim.

Sign conventions below were established by measuring
``Gf.Camera.frustum.ComputeProjectionMatrix()``, not derived from the docs:

    horizontalApertureOffset = -(cx - W/2) * horizontalAperture / W     # NEGATIVE
    verticalApertureOffset   = +(cy - H/2) * verticalAperture   / H     # POSITIVE

A positive horizontal offset slides the frustum toward +X, which pushes scene
content left in the image and therefore *decreases* the pixel the optical axis
lands on — hence the sign flip. The vertical axis flips twice (image rows count
down, USD +Y is up) and comes out positive.
"""

import math
from dataclasses import dataclass, replace

RES_W, RES_H = 640, 480  # every intrinsic below is DEFINED at this resolution
APERTURE = 36.0  # horizontal film-back gauge; vertical is derived per camera from fy


@dataclass(frozen=True)
class Intrinsics:
    """OpenCV pinhole intrinsics at (RES_W, RES_H) plus the vendor distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"  # vendor label: rational | inverse_brown_conrady | pinhole
    coeffs: tuple[float, ...] = ()  # OpenCV order k1 k2 p1 p2 k3 k4 k5 k6
    serial: str = ""
    borrowed_from: str = ""  # non-empty => not this camera's own calibration


# MEASURED 2026-08-07 off the real rig, colour streams at 640x480.
INTRINSICS: dict[str, Intrinsics] = {
    "third_view_orbbec335L": Intrinsics(
        366.978,
        366.935,
        321.018,
        241.658,
        model="rational",
        coeffs=(-0.033379, 0.036421, 0.000303, 0.000059, -0.012734, 0.0, 0.0, 0.0),
        serial="CP2R5530009H",
    ),
    "left_wrist_orbbec335": Intrinsics(
        460.927,
        461.357,
        322.002,
        236.508,
        model="rational",
        coeffs=(0.007475, -0.047748, -0.000990, 0.000189, 0.032395, 0.0, 0.0, 0.0),
        serial="CP02653000Z2",
    ),
    "second_third_realsense435i": Intrinsics(
        616.023,
        616.174,
        331.030,
        240.514,
        model="inverse_brown_conrady",
        coeffs=(0.0,) * 8,
        serial="040322071795",
    ),
}


def borrow(source: str, note: str = "") -> Intrinsics:
    """Reuse another camera's calibration, recorded explicitly rather than hidden."""
    return replace(INTRINSICS[source], serial=note, borrowed_from=source)


# The right arm's wrist camera was not part of the calibration dump. Same model
# as the left one (Gemini 335), so borrow it — and say so in the asset.
INTRINSICS["right_wrist_orbbec335"] = borrow("left_wrist_orbbec335")

# OpenCV coefficient order the opencvPinhole schema expects. k1..k3 alone is
# plumb_bob / Brown-Conrady; adding k4..k6 makes it the rational polynomial
# model. There is no separate "rational" model in Isaac Sim 5.1.
OPENCV_PINHOLE_COEFFS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")


def usd_projection(
    fx: float, fy: float, cx: float, cy: float, w: int = RES_W, h: int = RES_H, aperture: float = APERTURE
) -> dict[str, float]:
    """OpenCV intrinsics -> the five USD camera projection attributes."""
    focal = fx * aperture / w
    v_aperture = focal * h / fy  # exact fy; NOT the square-pixel aperture*h/w
    return {
        "focalLength": focal,
        "horizontalAperture": aperture,
        "verticalAperture": v_aperture,
        "horizontalApertureOffset": -(cx - w / 2.0) * aperture / w,
        "verticalApertureOffset": (cy - h / 2.0) * v_aperture / h,
    }


def usd_to_cv(attrs: dict[str, float], w: int = RES_W, h: int = RES_H) -> tuple[float, float, float, float]:
    """Inverse of usd_projection: recover (fx, fy, cx, cy) from authored attributes."""
    focal = attrs["focalLength"]
    a_h, a_v = attrs["horizontalAperture"], attrs["verticalAperture"]
    fx = focal * w / a_h
    fy = focal * h / a_v
    cx = w / 2.0 - attrs.get("horizontalApertureOffset", 0.0) * w / a_h
    cy = h / 2.0 + attrs.get("verticalApertureOffset", 0.0) * h / a_v
    return fx, fy, cx, cy


def hfov_deg(fx: float, w: int = RES_W) -> float:
    return 2 * math.degrees(math.atan(w / (2 * fx)))


def vfov_deg(fy: float, h: int = RES_H) -> float:
    return 2 * math.degrees(math.atan(h / (2 * fy)))


def describe(name: str) -> str:
    intr = INTRINSICS[name]
    a = usd_projection(intr.fx, intr.fy, intr.cx, intr.cy)
    src = f"borrowed from {intr.borrowed_from}" if intr.borrowed_from else f"SN {intr.serial}"
    return (
        f"{name}: fx/fy {intr.fx:.3f}/{intr.fy:.3f} c {intr.cx:.3f},{intr.cy:.3f} -> "
        f"focal {a['focalLength']:.5f} vAp {a['verticalAperture']:.5f} "
        f"off {a['horizontalApertureOffset']:+.6f}/{a['verticalApertureOffset']:+.6f} "
        f"FOV {hfov_deg(intr.fx):.2f}x{vfov_deg(intr.fy):.2f} deg [{intr.model}] {src}"
    )


if __name__ == "__main__":
    ok = True
    for cam_name, intr in INTRINSICS.items():
        attrs = usd_projection(intr.fx, intr.fy, intr.cx, intr.cy)
        back = usd_to_cv(attrs)
        err = max(abs(back[i] - (intr.fx, intr.fy, intr.cx, intr.cy)[i]) for i in range(4))
        ok &= err < 1e-9
        print(f"[{'PASS' if err < 1e-9 else 'FAIL'}] {describe(cam_name)}  roundtrip {err:.1e}")
    print("ALL PASS" if ok else "FAILURES")
