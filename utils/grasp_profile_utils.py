"""Grasp profile configuration loader and query utility.

Loads object grasp profiles from a YAML configuration file and provides
methods to query grasp parameters (gripper width, grasp height, approach
angle, orientation, etc.) with inheritance resolution and default fallbacks.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation as R


# Maximum inheritance depth to prevent circular references
_MAX_INHERIT_DEPTH = 5


class GraspProfileManager:
    """Manages object grasp profiles loaded from a YAML configuration file.

    Supports:
    - Per-object grasp parameter lookup with global default fallback.
    - Inheritance between object profiles (``inherit`` field).
    - Four grasp geometry types: ``axial_symmetric``, ``bilateral``,
      ``fixed``, and ``handle``.
    - Automatic approach-angle computation for round/symmetric objects.

    Example::

        mgr = GraspProfileManager("config/grasp_profiles.yaml")
        width = mgr.get_gripper_width("beaker")        # 0.022
        angle = mgr.compute_approach_angle("beaker", obj_pos, robot_pos)
    """

    def __init__(self, config_path: str) -> None:
        """Load grasp profiles from *config_path*.

        Args:
            config_path: Path to the grasp profiles YAML file.  Both
                absolute and project-relative paths are accepted.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        if not os.path.isabs(config_path):
            # Resolve relative to the project root (two levels up from utils/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Grasp profile config not found: {config_path}")

        with open(config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        self._defaults: Dict[str, Any] = raw.get("defaults", {})
        self._objects: Dict[str, Any] = raw.get("objects", {})

        # Build a lowercase lookup index for case-insensitive matching
        self._name_index: Dict[str, str] = {
            name.lower(): name for name in self._objects
        }

        # Cache for resolved profiles (after inheritance + defaults merge)
        self._resolved_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Loaded grasp profiles: {} objects, config={}",
            len(self._objects),
            config_path,
        )

    # ------------------------------------------------------------------
    # Core resolution
    # ------------------------------------------------------------------

    def get_profile(self, object_name: str) -> Dict[str, Any]:
        """Return the fully resolved grasp profile for *object_name*.

        Resolution order (highest priority first):
        1. Object-specific values
        2. Inherited object values (recursive, up to 5 levels)
        3. Global ``defaults``

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            A dict containing all grasp parameters with defaults filled in.
        """
        key = object_name.lower()
        if key in self._resolved_cache:
            return self._resolved_cache[key]

        profile = self._resolve(key, depth=0)
        self._resolved_cache[key] = profile
        return profile

    def _resolve(self, key: str, depth: int) -> Dict[str, Any]:
        """Recursively resolve inheritance and merge with defaults."""
        if depth > _MAX_INHERIT_DEPTH:
            logger.warning(
                "Inheritance depth exceeded for '{}', stopping at depth {}",
                key,
                depth,
            )
            return copy.deepcopy(self._defaults)

        # Start from defaults
        result = copy.deepcopy(self._defaults)

        # Find the raw object config (case-insensitive)
        canonical = self._name_index.get(key)
        if canonical is None:
            logger.debug("No grasp profile for '{}', using defaults", key)
            return result

        obj_cfg = copy.deepcopy(self._objects[canonical])

        # Handle inheritance
        parent_name = obj_cfg.pop("inherit", None)
        if parent_name is not None:
            parent_profile = self._resolve(parent_name.lower(), depth + 1)
            result = parent_profile  # parent already includes defaults

        # Overlay object-specific values
        _deep_merge(result, obj_cfg)
        return result

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    def get_gripper_width(self, object_name: str) -> float:
        """Return the gripper closing width in meters.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            Gripper width in meters.
        """
        return float(self.get_profile(object_name).get("gripper_width", 0.02))

    def get_grasp_height(
        self, object_name: str, object_size: np.ndarray
    ) -> float:
        """Return the grasp height offset from the object bottom.

        If the profile specifies ``grasp_height`` (absolute), that value is
        returned directly.  Otherwise ``grasp_height_ratio * object_size[2]``
        is used.

        Args:
            object_name: Object name (case-insensitive).
            object_size: Object bounding-box size ``[w, d, h]``.

        Returns:
            Grasp height offset in meters.
        """
        profile = self.get_profile(object_name)
        if "grasp_height" in profile:
            return float(profile["grasp_height"])
        ratio = float(profile.get("grasp_height_ratio", 0.4))
        return object_size[2] * ratio

    def get_pre_grasp_height(
        self, object_name: str, object_size: np.ndarray
    ) -> float:
        """Return the pre-grasp height offset from the object bottom.

        If the profile specifies ``pre_grasp_height`` (absolute), that value
        is returned directly.  Otherwise
        ``pre_grasp_height_ratio * object_size[2]`` is used.

        Args:
            object_name: Object name (case-insensitive).
            object_size: Object bounding-box size ``[w, d, h]``.

        Returns:
            Pre-grasp height offset in meters.
        """
        profile = self.get_profile(object_name)
        if "pre_grasp_height" in profile:
            return float(profile["pre_grasp_height"])
        ratio = float(profile.get("pre_grasp_height_ratio", 0.667))
        return object_size[2] * ratio

    def get_orientation(self, object_name: str) -> np.ndarray:
        """Return the end-effector orientation as a quaternion ``[x, y, z, w]``.

        The profile stores Euler angles in degrees (``euler_xyz``).  This
        method converts them to a unit quaternion using the ``xyz`` intrinsic
        convention.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            Quaternion array of shape ``(4,)`` in ``[x, y, z, w]`` order.
        """
        profile = self.get_profile(object_name)
        orient_cfg = profile.get("orientation", {})
        euler_deg = orient_cfg.get("euler_xyz", [0, 90, 0])
        quat = R.from_euler("xyz", np.radians(euler_deg)).as_quat()  # [x,y,z,w]
        return quat

    def get_lift_height(self, object_name: str) -> float:
        """Return the post-grasp lift height in meters.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            Lift height in meters.
        """
        return float(self.get_profile(object_name).get("lift_height", 0.25))

    def get_pre_offset_x(self, object_name: str) -> float:
        """Return the horizontal approach offset in meters.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            Horizontal offset in meters.
        """
        return float(self.get_profile(object_name).get("pre_offset_x", 0.05))

    def get_pre_offset_z(self, object_name: str) -> float:
        """Return the vertical pre-approach clearance in meters.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            Vertical clearance in meters.
        """
        return float(self.get_profile(object_name).get("pre_offset_z", 0.12))

    # ------------------------------------------------------------------
    # Approach angle computation
    # ------------------------------------------------------------------

    def compute_approach_angle(
        self,
        object_name: str,
        object_pos: np.ndarray,
        robot_base_pos: np.ndarray,
    ) -> float:
        """Compute the optimal horizontal approach angle in radians.

        For ``axial_symmetric`` objects the angle is chosen based on the
        ``preferred_angle`` strategy:

        - ``"auto"`` — angle from robot base toward the object (current
          default behaviour in the codebase).
        - ``"nearest"`` — same as *auto* (reserved for future gripper-pose
          aware selection).
        - ``<float>`` — fixed angle in degrees, converted to radians.

        The result is then adjusted to avoid any ``avoid_angles`` ranges
        defined in the profile.

        For ``fixed`` geometry the angle is derived from the configured
        ``approach_direction`` vector.

        For ``bilateral`` geometry the *auto* angle is snapped to the
        nearest of the two opposite directions along the symmetry plane.

        Args:
            object_name: Object name (case-insensitive).
            object_pos: Object world position ``[x, y, z]``.
            robot_base_pos: Robot base world position ``[x, y, z]``.

        Returns:
            Approach angle in radians (measured counter-clockwise from the
            positive X axis in the XY plane).
        """
        object_pos = np.asarray(object_pos, dtype=np.float64)
        robot_base_pos = np.asarray(robot_base_pos, dtype=np.float64)

        profile = self.get_profile(object_name)
        geom = profile.get("grasp_geometry", {})
        geom_type = geom.get("type", "axial_symmetric")

        if geom_type == "fixed":
            direction = np.asarray(
                geom.get("approach_direction", [1, 0, 0]), dtype=np.float64
            )
            return float(np.arctan2(direction[1], direction[0]))

        # Base angle: direction from robot base to object (XY plane)
        diff = object_pos[:2] - robot_base_pos[:2]
        base_angle = float(np.arctan2(diff[1], diff[0]))

        if geom_type == "bilateral":
            # Snap to the nearest of two opposite directions
            sym_axis = np.asarray(
                geom.get("symmetry_axis", [0, 0, 1]), dtype=np.float64
            )
            # For bilateral, the two valid angles are base_angle and base_angle + pi
            # We pick whichever is closer to the "from robot" direction
            # (they are equivalent, so just return base_angle)
            return base_angle

        # axial_symmetric or handle
        preferred = geom.get("preferred_angle", "auto")

        if isinstance(preferred, (int, float)):
            angle = np.radians(float(preferred))
        else:
            # "auto" or "nearest" — use the robot→object direction
            angle = base_angle

        # Adjust for avoid_angles
        avoid_ranges = geom.get("avoid_angles", [])
        if avoid_ranges:
            angle = self._adjust_for_avoid_angles(angle, avoid_ranges)

        return angle

    def compute_approach_direction(
        self,
        object_name: str,
        object_pos: np.ndarray,
        robot_base_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute a unit approach direction vector in the XY plane.

        This is a convenience wrapper around :meth:`compute_approach_angle`
        that returns a 3-D unit vector ``[cos(a), sin(a), 0]``.

        Args:
            object_name: Object name (case-insensitive).
            object_pos: Object world position ``[x, y, z]``.
            robot_base_pos: Robot base world position ``[x, y, z]``.

        Returns:
            Unit direction vector of shape ``(3,)``.
        """
        angle = self.compute_approach_angle(object_name, object_pos, robot_base_pos)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_for_avoid_angles(
        angle: float, avoid_ranges: List[Dict[str, List[float]]]
    ) -> float:
        """Shift *angle* out of any avoid-angle ranges.

        Each entry in *avoid_ranges* is a dict with a ``"range"`` key
        holding ``[min_deg, max_deg]``.  If *angle* falls inside any such
        range, it is shifted to the nearest boundary.

        Args:
            angle: Candidate angle in radians.
            avoid_ranges: List of ``{"range": [min_deg, max_deg]}`` dicts.

        Returns:
            Adjusted angle in radians.
        """
        angle_deg = np.degrees(angle) % 360

        for avoid in avoid_ranges:
            bounds = avoid.get("range", [])
            if len(bounds) != 2:
                continue
            lo, hi = float(bounds[0]) % 360, float(bounds[1]) % 360

            if lo <= hi:
                in_range = lo <= angle_deg <= hi
            else:
                # Wraps around 360 (e.g. [350, 10])
                in_range = angle_deg >= lo or angle_deg <= hi

            if in_range:
                # Move to the nearest boundary
                dist_lo = _angle_distance_deg(angle_deg, lo)
                dist_hi = _angle_distance_deg(angle_deg, hi)
                if dist_lo <= dist_hi:
                    angle_deg = (lo - 1) % 360
                else:
                    angle_deg = (hi + 1) % 360

        return np.radians(angle_deg)

    def list_objects(self) -> List[str]:
        """Return a list of all configured object names.

        Returns:
            List of canonical object names from the config file.
        """
        return list(self._objects.keys())

    def has_profile(self, object_name: str) -> bool:
        """Check whether a specific profile exists for *object_name*.

        Args:
            object_name: Object name (case-insensitive).

        Returns:
            ``True`` if the object has a dedicated profile entry.
        """
        return object_name.lower() in self._name_index


# ======================================================================
# Module-level helpers
# ======================================================================

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place.

    For nested dicts the merge is recursive.  All other types are replaced.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _angle_distance_deg(a: float, b: float) -> float:
    """Return the shortest angular distance in degrees between *a* and *b*."""
    d = abs(a - b) % 360
    return min(d, 360 - d)
