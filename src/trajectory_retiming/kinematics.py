"""Forward kinematics and Jacobian computation using UR30 DH parameters."""

import numpy as np
from .data_types import JointState, CartesianPose

# UR30 Denavit-Hartenberg parameters
# Format: [a, d, alpha] for each joint
UR30_DH_PARAMS = [
    [0.0, 0.2363, np.pi / 2],      # Joint 1
    [-0.6370, 0.0, 0.0],          # Joint 2
    [-0.5037, 0.0, 0.0],          # Joint 3
    [0.0, 0.2010, np.pi / 2],     # Joint 4
    [0.0, 0.1593, -np.pi / 2],   # Joint 5
    [0.0, 0.1543, 0.0],           # Joint 6
]


def dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    """
    Compute Denavit-Hartenberg transformation matrix.

    Args:
        theta: Joint angle (rotation about z-axis)
        a: Link length (translation along x-axis)
        d: Link offset (translation along z-axis)
        alpha: Link twist (rotation about x-axis)

    Returns:
        4x4 homogeneous transformation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    T = np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1],
    ])

    return T


def forward_kinematics(joint_state: JointState) -> CartesianPose:
    """
    Compute end-effector pose from joint configuration using UR30 DH parameters.

    Args:
        joint_state: Joint positions (6-DOF) in radians

    Returns:
        End-effector position and orientation (quaternion [w, x, y, z])
    """
    if len(joint_state.positions) != 6:
        raise ValueError("Joint state must have 6 joint positions")

    # Start with identity transformation
    T = np.eye(4)

    # Compute transformation for each joint
    for i in range(6):
        theta = joint_state.positions[i]
        a, d, alpha = UR30_DH_PARAMS[i]
        T_i = dh_transform(theta, a, d, alpha)
        T = T @ T_i

    # Extract position
    position = T[:3, 3].tolist()

    # Extract rotation matrix and convert to quaternion
    R = T[:3, :3]
    orientation = rotation_matrix_to_quaternion(R)

    return CartesianPose(position=position, orientation=orientation)


def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return [w, x, y, z]


def compute_jacobian(joint_state: JointState) -> np.ndarray:
    """
    Compute 6x6 geometric Jacobian matrix at given joint configuration.

    The Jacobian relates joint velocities to end-effector velocities:
    v = J(q) * q_dot
    where v is [linear_velocity, angular_velocity] and q_dot is joint velocities.

    Args:
        joint_state: Joint positions (6-DOF) in radians

    Returns:
        6x6 Jacobian matrix (first 3 rows: linear velocity, last 3 rows: angular velocity)
    """
    if len(joint_state.positions) != 6:
        raise ValueError("Joint state must have 6 joint positions")

    # Compute forward kinematics for each link
    T_links = []
    T = np.eye(4)

    for i in range(6):
        theta = joint_state.positions[i]
        a, d, alpha = UR30_DH_PARAMS[i]
        T_i = dh_transform(theta, a, d, alpha)
        T = T @ T_i
        T_links.append(T.copy())

    # End-effector position
    p_ee = T_links[5][:3, 3]

    # Initialize Jacobian
    J = np.zeros((6, 6))

    # Compute Jacobian columns
    for i in range(6):
        # Get z-axis (rotation axis) of joint i
        if i == 0:
            z_i = np.array([0, 0, 1])  # Base z-axis
            p_i = np.array([0, 0, 0])  # Base origin
        else:
            z_i = T_links[i - 1][:3, 2]  # z-axis of previous link
            p_i = T_links[i - 1][:3, 3]  # Origin of previous link

        # Linear velocity component: z_i x (p_ee - p_i)
        J[:3, i] = np.cross(z_i, p_ee - p_i)

        # Angular velocity component: z_i
        J[3:, i] = z_i

    return J
