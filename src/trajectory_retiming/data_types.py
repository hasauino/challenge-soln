"""Data types for trajectory representation."""

from typing import List
from dataclasses import dataclass
import numpy as np


DEFAULT_INIT_VELOCITY = np.zeros(6).tolist()
DEFAULT_INIT_ACCELERATION = np.zeros(6).tolist()


@dataclass
class JointState:
    """Joint configuration (6-DOF)."""

    positions: List[float]  # Joint positions in radians
    velocities: List[float] = None  # Joint speeds in radians/seconds
    accelerations: List[float] = None  # Joint acceleration in radians/seconds^2


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

    def compute_velocities(self):
        if len(self.waypoints) < 2:
            return
        positions = np.array([wp.positions for wp in self.waypoints], dtype=float)
        time_diffs = np.diff(self.time_from_start)

        # v_i = ( p_{i+1} - p_i ) / ( t_{i+1} - t_i ), with the last row left at
        # zero: the arm stops moving at the end of the trajectory
        velocities = np.zeros_like(positions)
        velocities[:-1] = np.diff(positions, axis=0) / time_diffs[:, np.newaxis]

        for waypoint, velocity in zip(self.waypoints, velocities.tolist()):
            waypoint.velocities = velocity

    def compute_acceleration(self):
        if len(self.waypoints) < 3:
            return
        velocities = np.array([wp.velocities for wp in self.waypoints], dtype=float)
        time_diffs = np.diff(self.time_from_start)

        # (t_{i+2} - t_{i}) / 2 expressed as consecutive first differences, except
        # for the waypoint before the last one (i.e. n-2), where the time diff is
        # approximated as the time between the last two waypoints
        centered_time_diffs = np.concatenate(
            [(time_diffs[:-1] + time_diffs[1:]) / 2, time_diffs[-1:]]
        )

        # a_i = ( v_{i+1} - v_i ) / ( (t_{i+2} - t_{i}) / 2 ), with the last row
        # left at zero: the arm is at rest at the end of the trajectory
        accelerations = np.zeros_like(velocities)
        accelerations[:-1] = (
            np.diff(velocities, axis=0) / centered_time_diffs[:, np.newaxis]
        )

        for waypoint, acceleration in zip(self.waypoints, accelerations.tolist()):
            waypoint.accelerations = acceleration
