import numpy as np
import pytest

from trajectory_retiming.data_types import JointState, JointTrajectory

# A 5-waypoint trajectory with deliberately non-uniform time steps so the
# centered time differences used by compute_acceleration are exercised.
TIME_FROM_START = [0.0, 1.0, 3.0, 4.0, 6.0]  # time diffs: [1.0, 2.0, 1.0, 2.0]

# Joint j simply tracks the scalar path below scaled by (j + 1), which keeps the
# expected values easy to derive by hand while still catching axis mix-ups.
SCALAR_POSITIONS = [0.0, 1.0, 6.0, 7.0, 9.0]
JOINT_SCALES = np.arange(1.0, 7.0)  # [1, 2, 3, 4, 5, 6]

# v_i = ( p_{i+1} - p_i ) / ( t_{i+1} - t_i ), last waypoint at rest:
#   1 / 1 = 1.0,  5 / 2 = 2.5,  1 / 1 = 1.0,  2 / 2 = 1.0,  0.0
EXPECTED_SCALAR_VELOCITIES = [1.0, 2.5, 1.0, 1.0, 0.0]

# a_i = ( v_{i+1} - v_i ) / ( (t_{i+2} - t_i) / 2 ), where the centered time
# diffs are [1.5, 1.5, 1.5] and the second-to-last one falls back to the final
# time diff (2.0). Last waypoint at rest:
#   1.5 / 1.5 = 1.0,  -1.5 / 1.5 = -1.0,  0 / 1.5 = 0.0,  -1 / 2 = -0.5,  0.0
EXPECTED_SCALAR_ACCELERATIONS = [1.0, -1.0, 0.0, -0.5, 0.0]


def scale_to_joints(scalar_values):
    """Expand a scalar path into the per-joint 6-DOF values it stands for."""
    return np.outer(scalar_values, JOINT_SCALES)


@pytest.fixture
def trajectory():
    waypoints = [
        JointState(positions=positions.tolist())
        for positions in scale_to_joints(SCALAR_POSITIONS)
    ]
    return JointTrajectory(waypoints=waypoints, time_from_start=list(TIME_FROM_START))


def test_compute_velocities(trajectory):
    trajectory.compute_velocities()

    velocities = np.array([wp.velocities for wp in trajectory.waypoints])
    np.testing.assert_allclose(velocities, scale_to_joints(EXPECTED_SCALAR_VELOCITIES))


def test_compute_velocities_leaves_last_waypoint_at_rest(trajectory):
    trajectory.compute_velocities()

    np.testing.assert_allclose(trajectory.waypoints[-1].velocities, np.zeros(6))


def test_compute_acceleration(trajectory):
    trajectory.compute_velocities()
    trajectory.compute_acceleration()

    accelerations = np.array([wp.accelerations for wp in trajectory.waypoints])
    np.testing.assert_allclose(
        accelerations, scale_to_joints(EXPECTED_SCALAR_ACCELERATIONS), atol=1e-12
    )


def test_compute_acceleration_leaves_last_waypoint_at_rest(trajectory):
    trajectory.compute_velocities()
    trajectory.compute_acceleration()

    np.testing.assert_allclose(trajectory.waypoints[-1].accelerations, np.zeros(6))


def test_compute_velocities_needs_two_waypoints():
    trajectory = JointTrajectory(
        waypoints=[JointState(positions=[0.0] * 6)], time_from_start=[0.0]
    )

    trajectory.compute_velocities()

    assert trajectory.waypoints[0].velocities is None


def test_compute_acceleration_needs_three_waypoints():
    trajectory = JointTrajectory(
        waypoints=[JointState(positions=[0.0] * 6), JointState(positions=[1.0] * 6)],
        time_from_start=[0.0, 1.0],
    )
    trajectory.compute_velocities()

    trajectory.compute_acceleration()

    assert all(wp.accelerations is None for wp in trajectory.waypoints)
