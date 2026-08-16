"""
Trajectory Retiming.

This module provides trajectory retiming functionality to ensure all constraints
are satisfied while minimizing the added time penalty.

Usage:
    Visualize a trajectory (joint positions, Cartesian speed, 3D path):

        python3 -m trajectory_retiming.retimer

    With a custom trajectory file:

        python3 -m trajectory_retiming.retimer --trajectory path/to/trajectory.json
"""

import copy
import json
import numpy as np
from .data_types import JointTrajectory, JointState
from .limits import RobotLimits
from .kinematics import forward_kinematics, compute_jacobian


def load_joint_trajectory_from_json(json_path: str) -> JointTrajectory:
    """Load trajectory from JSON file into JointTrajectory.

    Args:
        json_path: Path to the JSON file containing trajectory data

    Returns:
        JointTrajectory with waypoints and time_from_start
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    waypoints = [
        JointState(positions=wp['positions']) for wp in data['waypoints']
    ]
    return JointTrajectory(
        waypoints=waypoints,
        time_from_start=data['time_from_start'],
    )






def retime_trajectory(
    trajectory: JointTrajectory, limits: RobotLimits
) -> JointTrajectory:
    """
    Retime trajectory to satisfy all motion constraints.

    The retimed trajectory must satisfy:
    - Cartesian end-effector speed ≤ limits.max_cartesian_speed
    - Joint velocities ≤ limits.max_joint_velocities (for each joint)
    - Joint accelerations ≤ limits.max_joint_accelerations (for each joint)
    - All waypoint positions preserved (path geometry unchanged)
    - Monotonically increasing timestamps
    - Minimized total trajectory duration

    In your implementation, clearly document:
    - Your interpolation model (how position/velocity is computed between waypoints)
    - Your algorithm and approach
    - Any trade-offs or assumptions you make

    Args:
        trajectory: Original joint trajectory with waypoints and timing
        limits: Robot physical limits

    Returns:
        Retimed trajectory with adjusted time_from_start values
    """
    trajectory_copy = copy.deepcopy(trajectory)

    # TODO: Implement the retiming algorithm

    return trajectory_copy


def plot_trajectory(trajectory: JointTrajectory) -> None:
    """
    Plot a trajectory object.

    Creates a multi-panel visualization showing:
    - Joint positions over time
    - Joint velocities over time
    - Cartesian speed over time
    - 3D Cartesian end-effector path

    Args:
        trajectory: JointTrajectory object (e.g. from retime_trajectory)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    waypoints = trajectory.waypoints
    time_stamps = np.array(trajectory.time_from_start)

    # Extract joint positions
    num_joints = len(waypoints[0].positions)
    joint_positions = np.array([wp.positions for wp in waypoints])

    # Compute Cartesian positions using FK
    cartesian_positions = []
    for wp in waypoints:
        pose = forward_kinematics(wp)
        cartesian_positions.append(pose.position)
    cartesian_positions = np.array(cartesian_positions)

    # Compute velocities using finite differences: v = delta_pos / delta_t
    delta_t = np.diff(time_stamps)
    delta_t[delta_t == 0] = 1e-9  # Avoid division by zero

    # Joint velocities
    delta_joint_pos = np.diff(joint_positions, axis=0)
    joint_velocities = delta_joint_pos / delta_t[:, np.newaxis]

    # Cartesian velocities and speed using Jacobian
    # v_cartesian = J(q) * q_dot (first 3 rows are linear velocity)
    cartesian_speed = np.zeros(len(waypoints) - 1)
    for i in range(len(waypoints) - 1):
        J = compute_jacobian(waypoints[i])
        # Joint velocities at this segment (q_dot)
        q_dot = joint_velocities[i]
        # Cartesian velocity from Jacobian (first 3 rows are linear velocity)
        v_cartesian = J[:3, :] @ q_dot
        # Speed is the magnitude of linear velocity
        cartesian_speed[i] = np.linalg.norm(v_cartesian)

    # Time stamps for velocities (midpoints between waypoints)
    velocity_time = (time_stamps[:-1] + time_stamps[1:]) / 2

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))

    # Color palette for joints
    colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#9b5de5', '#00bbf9']
    joint_labels = [f'Joint {i+1}' for i in range(num_joints)]

    # Plot 1: Joint positions over time
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(num_joints):
        ax1.plot(time_stamps, joint_positions[:, i],
                 color=colors[i], linewidth=2, label=joint_labels[i])
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Position [rad]', fontsize=11)
    ax1.set_title('Joint Positions', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([time_stamps[0], time_stamps[-1]])

    # Plot 2: Joint velocities over time
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(num_joints):
        ax2.plot(velocity_time, joint_velocities[:, i],
                 color=colors[i], linewidth=2, label=joint_labels[i])
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Velocity [rad/s]', fontsize=11)
    ax2.set_title('Joint Velocities', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([time_stamps[0], time_stamps[-1]])

    # Plot 3: Cartesian speed over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(velocity_time, cartesian_speed, color='#e63946', linewidth=2.5)
    ax3.fill_between(velocity_time, cartesian_speed, alpha=0.3, color='#e63946')
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Speed [m/s]', fontsize=11)
    ax3.set_title('Cartesian End-Effector Speed', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([time_stamps[0], time_stamps[-1]])
    ax3.set_ylim(bottom=0)

    # Add max speed annotation
    max_speed = np.max(cartesian_speed)
    max_speed_idx = np.argmax(cartesian_speed)
    ax3.axhline(y=max_speed, color='#264653', linestyle='--', alpha=0.7)
    ax3.annotate(f'Max: {max_speed:.3f} m/s',
                 xy=(velocity_time[max_speed_idx], max_speed),
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=9, color='#264653')

    # Plot 4: 3D Cartesian path
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(cartesian_positions[:, 0],
             cartesian_positions[:, 1],
             cartesian_positions[:, 2],
             color='#e63946', linewidth=2.5, label='End-effector path')
    ax4.scatter(cartesian_positions[0, 0],
                cartesian_positions[0, 1],
                cartesian_positions[0, 2],
                color='#2a9d8f', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(cartesian_positions[-1, 0],
                cartesian_positions[-1, 1],
                cartesian_positions[-1, 2],
                color='#264653', s=100, marker='s', label='End', zorder=5)
    ax4.set_xlabel('X [m]', fontsize=10)
    ax4.set_ylabel('Y [m]', fontsize=10)
    ax4.set_zlabel('Z [m]', fontsize=10)
    ax4.set_title('Cartesian End-Effector Path', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    """Main entry point for trajectory visualization."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Plot robot trajectory from JSON file'
    )
    parser.add_argument(
        '--trajectory', '-t',
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '..', 'examples', 'p2p_trajectory.json'
        ),
        help='Path to trajectory JSON file'
    )
    args = parser.parse_args()

    print(f"Loading trajectory from: {args.trajectory}")
    trajectory = load_joint_trajectory_from_json(args.trajectory)
    plot_trajectory(trajectory)


if __name__ == "__main__":
    main()
