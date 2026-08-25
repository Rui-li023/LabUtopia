"""Give gripper fingers reliable collision geometry.

A link with ``<visual>`` but no ``<collision>`` converts to a USD prim that renders
normally and collides with nothing. On an arm link that is merely wrong; on a gripper
finger it is fatal and almost invisible: motion planning, IK, FK and the TCP frame are
all still correct, the arm drives to the object and the fingers close straight through
it. The failure looks exactly like a mis-tuned grasp.

ARX X5 and R5 both ship this way -- every link has a collider except ``link7``,
``link8`` and the two pad frames, i.e. the whole gripper. That, not the controller,
is why they scored 0% while reaching the beaker correctly.

The collider is a box at the pad frame rather than a convex hull of the finger mesh:
the ARX finger is a fork whose hull would fill the gap between the jaws and stop the
object ever entering. Its inner face sits exactly on the pad plane, so the jaw opening
matches the mechanism's spec (2 x 0.044 m = 88 mm).

Usage::

    python scripts/urdf_to_usd/add_finger_colliders.py \\
        --urdf third_party/urdf/arx/_generated/arx_x5.urdf --preset arx
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# link -> (pad origin in the link's own frame, box size, half-thickness sign on y).
# The pad origin is read straight off the URDF's own *_pad_joint; the sign says which
# way the gripping face points, so the box can be shifted to put that face on the pad
# plane instead of straddling it.
PRESETS = {
    "arx": {
        "link7": {"origin": (0.05843, -0.0249, 0.0), "size": (0.035, 0.010, 0.045), "face": +1},
        "link8": {"origin": (0.05843, 0.0249, 0.0), "size": (0.035, 0.010, 0.045), "face": -1},
    },
    # The Z1 ships collision meshes, but its single rotating claw imports as a
    # complex convex contact surface which ejects cylindrical labware sideways.
    # These thin pads coincide with the two visible inner faces. The original
    # collision meshes remain in place for the rest of each gripper link.
    "unitree_z1": {
        "gripperStator": {"origin": (0.120, 0.0, -0.012), "size": (0.050, 0.060, 0.006), "face": 0},
        "gripperMover": {"origin": (0.070, 0.0, 0.003), "size": (0.045, 0.060, 0.006), "face": 0},
    },
}


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def has_collision(link: ET.Element) -> bool:
    return link.find("collision") is not None


def add_box_collision(link: ET.Element, origin, size, face, name: str) -> None:
    """Insert a box collider whose gripping face lands on the pad plane.

    Shifting by half the thickness (rather than centring on the pad) keeps the jaw
    opening equal to the mechanism's travel; centring would silently narrow it by one
    thickness and could stop a wide object entering the jaws at all.
    """
    x, y, z = origin
    y = y + face * (size[1] / 2.0)
    col = ET.SubElement(link, "collision", {"name": name})
    ET.SubElement(col, "origin", {"xyz": f"{x} {y} {z}", "rpy": "0 0 0"})
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "box", {"size": f"{size[0]} {size[1]} {size[2]}"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--force", action="store_true", help="Add even if the link already has a collider")
    args = ap.parse_args()

    path = Path(args.urdf).resolve()
    if not path.is_file():
        say(f"ERROR: URDF not found: {path}")
        return 1

    tree = ET.parse(path)
    root = tree.getroot()
    spec = PRESETS[args.preset]
    links = {ln.get("name"): ln for ln in root.findall("link")}

    changed = 0
    for name, cfg in spec.items():
        link = links.get(name)
        if link is None:
            say(f"  {name}: not in this URDF, skipped")
            continue
        collision_name = f"labutopia_{args.preset}_pad"
        if any(collision.get("name") == collision_name for collision in link.findall("collision")):
            say(f"  {name}: {collision_name} already present, left alone")
            continue
        if has_collision(link) and not args.force:
            say(f"  {name}: already has a collider, left alone")
            continue
        add_box_collision(link, cfg["origin"], cfg["size"], cfg["face"], collision_name)
        changed += 1
        say(f"  {name}: added box collider size={cfg['size']} at pad {cfg['origin']}")

    if not changed:
        say("nothing to do")
        return 0

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    say(f"wrote {path} ({changed} collider(s) added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
