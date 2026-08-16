"""Robot physical limits and constraints."""

from typing import List
from dataclasses import dataclass, field


@dataclass
class RobotLimits:
    """Physical limits for robot motion."""

    max_joint_velocities: List[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    )  # rad/s for each joint (UR30 defaults)
    max_joint_accelerations: List[float] = field(
        default_factory=lambda: [5.0, 5.0, 5.0, 8.0, 8.0, 8.0]
    )  # rad/s² for each joint (UR30 defaults)
    max_cartesian_speed: float = 0.5  # m/s
