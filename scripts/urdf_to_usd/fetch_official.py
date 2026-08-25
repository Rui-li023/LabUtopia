"""Download official NVIDIA Isaac Sim robot USD assets.

NVIDIA generates these with the same URDF importer this pipeline uses -- a converted
ur5e and the official one are face-for-face identical (129347 faces, same OmniPBR
material names, same layered layout). The official copies are still preferable where
they exist: their materials are deduplicated into one shared set instead of one copy
per link, and they ship an extra ``*_robot_schema.usd`` semantic layer.

So: use official assets when NVIDIA publishes the model, and use ``batch.py`` to
convert from URDF only for models they don't.

Assets are laid out as ``<model>/<model>.usd`` plus a sibling ``configuration/``
directory; both must be fetched together or the entry point is an empty shell.

Usage (any Python with requests-free stdlib; no Isaac Sim needed)::

    python scripts/urdf_to_usd/fetch_official.py                 # everything in robots.yaml
    python scripts/urdf_to_usd/fetch_official.py --only ur5e     # a subset
    python scripts/urdf_to_usd/fetch_official.py --list          # show what's on the server
"""

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "robots.yaml"
DEST_ROOT = REPO_ROOT / "assets" / "robots" / "official"

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ISAAC_VERSION = "5.1"
ROBOTS_PREFIX = f"Assets/Isaac/{ISAAC_VERSION}/Isaac/Robots/"


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def list_keys(prefix: str) -> list[str]:
    """List every S3 key under a prefix, following continuation tokens.

    S3 caps a response at 1000 keys; without paging, the Robots listing silently
    truncates partway through the alphabet and models look absent when they exist.
    """
    keys: list[str] = []
    token = None
    for _ in range(50):
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", body)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not match:
            break
        token = match.group(1)
    return keys


def download(key: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", dest)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        say(f"    FAILED {key}: {exc}")
        return False


def fetch_model(vendor: str, model: str, name: str) -> tuple[bool, float]:
    """Fetch one model's USD files. Returns (ok, megabytes)."""
    prefix = f"{ROBOTS_PREFIX}{vendor}/{model}/"
    keys = [k for k in list_keys(prefix) if k.endswith(".usd") and "/.thumbs/" not in k]
    if not keys:
        say(f"  {name}: NOT FOUND on server ({prefix})")
        return False, 0.0

    total = 0
    for key in keys:
        relative = key[len(prefix) :]
        dest = DEST_ROOT / name / relative
        if download(key, dest):
            total += dest.stat().st_size
    say(f"  {name}: {len(keys)} file(s), {total / 1024 / 1024:.1f} MB  <- {vendor}/{model}")
    return True, total / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--only", nargs="*", default=None, help="Only these entry names")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list:
        keys = list_keys(ROBOTS_PREFIX)
        models: dict[str, set[str]] = {}
        for key in keys:
            parts = key.split("/")
            if len(parts) >= 7 and not parts[6].endswith((".usd", ".md", ".txt", ".pdf")):
                models.setdefault(parts[5], set()).add(parts[6])
        for vendor in sorted(models):
            say(f"{vendor:<22} {sorted(models[vendor])}")
        return 0

    data = yaml.safe_load(Path(args.manifest).read_text())
    entries = data.get("official") or []
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["name"] in wanted]

    if not entries:
        say("no official entries selected")
        return 1

    say(f"\nDownloading {len(entries)} official asset(s) -> {DEST_ROOT.relative_to(REPO_ROOT)}\n")
    ok, total = 0, 0.0
    for entry in entries:
        success, size = fetch_model(entry["vendor"], entry["model"], entry["name"])
        ok += success
        total += size

    say(f"\n{ok}/{len(entries)} fetched, {total:.0f} MB total")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
