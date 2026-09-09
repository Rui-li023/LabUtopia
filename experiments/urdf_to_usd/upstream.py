"""Locating the fetched upstream URDF trees, and saying so usefully when they are absent.

`third_party/` is gitignored: the vendor description packages are cloned, not
vendored. Every script that reads a vendor DAE therefore has to cope with the
directory simply not being there on a fresh checkout, and the useful thing to
print is not "not found" but the exact clone command for the one repo needed.
The clone URL and branch already live in `robots.yaml`, so they are read from
there rather than restated here and left to drift.
"""

import os

import yaml
from loguru import logger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS_YAML = os.path.join(REPO_ROOT, "experiments", "urdf_to_usd", "robots.yaml")
DEFAULT_DAE_ROOT = "third_party/urdf"


def _repo_spec(vendor: str) -> dict | None:
    try:
        with open(REPOS_YAML, encoding="utf-8") as handle:
            return (yaml.safe_load(handle) or {}).get("repos", {}).get(vendor)
    except OSError:
        return None


def clone_hint(vendor: str, dae_root: str = DEFAULT_DAE_ROOT) -> str:
    """The `git clone` line that would put *vendor* where a script expects it."""
    spec = _repo_spec(vendor) or {}
    url = spec.get("url", vendor)
    if "://" not in url:
        url = f"https://github.com/{url}.git"
    branch = f"--branch {spec['branch']} " if spec.get("branch") else ""
    return f"git clone --depth 1 {branch}{url} {os.path.join(dae_root, vendor)}"


def require_vendor_tree(vendor: str, dae_root: str) -> str | None:
    """Path to *vendor* under *dae_root*, or None after explaining how to get it."""
    path = os.path.join(dae_root, vendor)
    if os.path.isdir(path):
        return path
    logger.error(f"Upstream mesh tree not found: {path}")
    logger.error(
        "third_party/ is gitignored -- the vendor description packages are cloned, not shipped. Fetch this one with:"
    )
    logger.error(f"    {clone_hint(vendor, dae_root)}")
    logger.error("See experiments/urdf_to_usd/README.md, and robots.yaml for every repo this pipeline uses.")
    return None
