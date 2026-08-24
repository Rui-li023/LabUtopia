# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Piper RMPFlow controller.

Kept only so existing imports keep working. The shared controller now reads the Lula
description and the kinematic base pose from the robot itself, so there is nothing
arm-specific left to override -- duplicating it here is how Piper ended up with a
stale config (planning against ``link6`` instead of the tool centre, and seeding the
base pose from the articulation prim rather than ``base_link``).
"""

from robots.franka.rmpflow_controller import RMPFlowController as _SharedRMPFlowController


class RMPFlowController(_SharedRMPFlowController):
    """Alias of the shared, robot-driven RMPFlow controller."""
