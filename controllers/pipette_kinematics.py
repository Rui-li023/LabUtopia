"""Local IK recovery for a continuous, already seeded pipette carry path."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def continuous_inverse_kinematics(solver, frame, position, quaternion, seed, bounds=None, *, joint_margin=0.05):
    """Reject global branch jumps; recover locally without relaxing pose limits.

    This only computes a joint target. It never writes simulator poses. The
    local solve is bounded by both physical joint limits and a 0.35 rad trust
    region around the preceding path sample.
    """
    if not np.isfinite(joint_margin) or not 0.05 <= joint_margin <= 0.25:
        raise ValueError("IK joint margin must be in [0.05, 0.25] radians")
    seed = np.asarray(seed, dtype=float)
    solution, ok = solver.compute_inverse_kinematics(
        frame,
        position,
        quaternion,
        warm_start=seed,
        position_tolerance=0.0005,
        orientation_tolerance=0.005,
    )
    if bounds is None:
        bounds = np.asarray(solver.get_cspace_position_limits()).T
    bounds = np.asarray(bounds)
    lower = np.maximum(bounds[:, 0] + joint_margin, seed - 0.35)
    upper = np.minimum(bounds[:, 1] - joint_margin, seed + 0.35)
    if ok and np.all(solution >= lower) and np.all(solution <= upper):
        return np.asarray(solution), True
    if np.any(lower >= upper):
        return seed.copy(), False
    target_rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])

    def residual(joints):
        actual_position, actual_matrix = solver.compute_forward_kinematics(frame, joints)
        angular = (target_rotation * Rotation.from_matrix(actual_matrix).inv()).as_rotvec()
        return np.r_[np.asarray(actual_position).reshape(3) - position, 0.15 * angular, 0.0001 * (joints - seed)]

    result = least_squares(
        residual,
        np.clip(seed, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=40,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    error = residual(result.x)
    valid = np.linalg.norm(error[:3]) <= 0.0005 and np.linalg.norm(error[3:6]) / 0.15 <= 0.005
    return result.x, bool(valid)
