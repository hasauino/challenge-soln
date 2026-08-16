"""Data types for trajectory representation."""

from typing import List
from dataclasses import dataclass


@dataclass
class JointState:
    """Joint configuration (6-DOF)."""

    positions: List[float]  # Joint positions in radians


@dataclass
class CartesianPose:
    """End-effector pose in cartesian space."""

    position: List[float]  # [x, y, z] in meters
    orientation: List[float]  # Quaternion [w, x, y, z]


@dataclass
class JointTrajectory:
    """Joint-space trajectory with timing information."""

    waypoints: List[JointState]  # Joint positions for each waypoint
    time_from_start: List[float]  # Timestamp for each waypoint (seconds)
