"""Extract a tileable plywood tabletop texture from the real camera snapshots.

The sim2real cell has to match the real benchtop, so the diffuse map is cut from
the real photos instead of using a stock wood material: candidate patches from
every camera are scored (no cables/hardware, strong *directional* grain), the
lighting gradient is divided out, the grain is amplified so the planks read as
stripes, and the edges are wrap-blended so the result tiles.
"""

import os

import cv2
import numpy as np

SNAP = "/data1/DataCollectionSystemV2/data/snapshots/20260806-164844"
OUT_DIR = "/home/ubuntu/Documents/LabUtopia/assets/sim2real/textures"

# (image, x0, y0, x1, y1) — bare tabletop regions, avoiding robot/cables/hardware
CANDIDATES = [
    ("third_view_orbbec335L_CP2R5530009H.jpg", 470, 300, 630, 460),
    ("third_view_orbbec335L_CP2R5530009H.jpg", 480, 150, 636, 306),
    ("third_view_orbbec335L_CP2R5530009H.jpg", 30, 300, 186, 456),
    ("third_view_orbbec335L_CP2R5530009H.jpg", 330, 300, 486, 456),
    ("second_third_realsense435i_040322071795.jpg", 20, 40, 276, 296),
    ("second_third_realsense435i_040322071795.jpg", 40, 190, 296, 446),
    ("second_third_realsense435i_040322071795.jpg", 330, 340, 586, 470),
    ("left_wrist_orbbec335_CP02653000Z2.jpg", 20, 60, 340, 380),
]

TILE = 1024
GRAIN = 2.4  # amplify the directional grain so the planks read as stripes
COLOR_REF = "third_view_orbbec335L_CP2R5530009H.jpg"  # camera the albedo is matched to


def load(name):
    img = cv2.imread(os.path.join(SNAP, name))
    assert img is not None, name
    return img


os.makedirs(OUT_DIR, exist_ok=True)
images = {name: load(name) for name in {c[0] for c in CANDIDATES}}

best, best_score = None, -1e9
for name, x0, y0, x1, y1 in CANDIDATES:
    patch = images[name][y0:y1, x0:x1]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = cv2.GaussianBlur(g, (0, 0), 9)
    # cables read as mid-grey, not black: test against the patch median instead
    dark = float((g < np.median(g) - 20).mean())
    blob = float((np.abs(g - flat) > 18).mean())  # knots that would visibly repeat
    detail = float(np.std(g - flat))  # how much grain there is
    gx = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)).mean())
    gy = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)).mean())
    aniso = max(gx, gy) / max(min(gx, gy), 1e-3)  # grain is directional, blur is not
    score = -dark * 4000 - blob * 400 + aniso * 4.0 + detail * 0.8
    print(
        f"  {name[:24]:24s} {(x0, y0, x1, y1)!s:22s} dark={dark:.3f} blob={blob:.4f} "
        f"detail={detail:.2f} aniso={aniso:.2f} -> {score:.2f}"
    )
    if score > best_score:
        best, best_score = (name, x0, y0, x1, y1), score

name, x0, y0, x1, y1 = best
print("picked", best)
patch = images[name][y0:y1, x0:x1].astype(np.float32)

# 1. divide out the lighting gradient, amplify the grain, restore the true colour
#    (albedo sampled over the whole reference frame, so a patch sitting in cooler
#     light does not shift the whole benchtop)
ref = images.get(COLOR_REF, images[name])
gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
target = np.median(ref[(gray_ref > 140) & (gray_ref < 245)].reshape(-1, 3), 0).astype(np.float32)
print("tabletop target BGR", target.round(1))

# modulate luminance only — a per-channel ratio would amplify JPEG chroma noise
# into coloured speckle instead of wood grain
lum = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
low = cv2.GaussianBlur(lum, (0, 0), min(patch.shape[:2]) / 12.0)
ratio = (lum / np.maximum(low, 1e-3))[:, :, None]
tex = np.clip((1.0 + (ratio - 1.0) * GRAIN) * target, 0, 255)

# 2. square + upscale
side = min(tex.shape[:2])
tex = cv2.resize(tex[:side, :side], (TILE, TILE), interpolation=cv2.INTER_CUBIC)

# 3. wrap-blend the edges so the map tiles seamlessly
f = TILE // 8
ramp_r = np.linspace(0.0, 1.0, f, dtype=np.float32).reshape(f, 1, 1)
tex[:f, :] = tex[:f, :] * ramp_r + tex[TILE - f :, :][::-1] * (1 - ramp_r)
ramp_c = np.linspace(0.0, 1.0, f, dtype=np.float32).reshape(1, f, 1)
tex[:, :f] = tex[:, :f] * ramp_c + tex[:, TILE - f :][:, ::-1] * (1 - ramp_c)

tex = np.clip(tex, 0, 255).astype(np.uint8)
out = os.path.join(OUT_DIR, "plywood_top_diffuse.png")
cv2.imwrite(out, tex)
print("wrote", out, tex.shape, "mean BGR", tex.reshape(-1, 3).mean(0).round(1))

# debug: 3x3 tiling to eyeball the seams, next to the raw source crop
tiled = np.tile(cv2.resize(tex, (340, 340)), (3, 3, 1))
raw = cv2.resize(images[name][y0:y1, x0:x1], (340, 340))
pad = np.full((tiled.shape[0], 20, 3), 40, np.uint8)
left = np.vstack([raw, np.full((tiled.shape[0] - 340, 340, 3), 40, np.uint8)])
cv2.imwrite("/tmp/table_texture_check.png", np.hstack([left, pad, tiled]))
print("wrote /tmp/table_texture_check.png")
