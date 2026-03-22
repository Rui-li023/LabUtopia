# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot definitions for LabUtopia.

This package provides robot classes for manipulation and mobile manipulation tasks.
All robots inherit from BaseRobot which defines a common interface.
"""

from robots.base_robot import BaseRobot
from robots.franka.franka import Franka
from robots.piper.piper import Piper
from robots.ridgebase_franka.ridgebase import Ridgebase

__all__ = ["BaseRobot", "Franka", "Piper", "Ridgebase"]
