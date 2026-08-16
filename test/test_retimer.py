"""Tests for `retime_trajectory`.

The central assertion throughout is `compute_limit_usage(...) <= 1.0`: every
limit expressed as a fraction of itself, computed with the same finite
differences that `JointTrajectory.compute_velocities` / `compute_acceleration`
use, so the tests check the trajectory the way the README defines it.
"""

import copy
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from trajectory_retiming import retimer
from trajectory_retiming.data_types import JointState, JointTrajectory
from trajectory_retiming.limits import RobotLimits
from trajectory_retiming.retimer import (
    compute_limit_usage,
    load_joint_trajectory_from_json,
    retime_trajectory,
)

EXAMPLE_TRAJECTORY = Path(__file__).parent / "assets" / "p2p_trajectory.json"

# `compute_limit_usage` is a ratio against the limit, so 1.0 is exactly
# saturated. The retimer targets a hair under 1.0 to stay clear of round-off,
# and the spline TOPP-RA plans on differs from the finite-difference model
# these tests measure by a fraction of a percent, so "no slack" means within
# 0.1% rather than exactly 1.0.
FEASIBLE = 1.0
SATURATED = pytest.approx(1.0, abs=1e-3)


@pytest.fixture
def limits():
    return RobotLimits()


@pytest.fixture
def example():
    return load_joint_trajectory_from_json(str(EXAMPLE_TRAJECTORY))


def make_trajectory(positions, duration=1.0) -> JointTrajectory:
    """Build a trajectory from an (n, 6) array with evenly spaced timestamps."""
    positions = np.asarray(positions, dtype=float)
    return JointTrajectory(
        waypoints=[JointState(positions=p.tolist()) for p in positions],
        time_from_start=np.linspace(0.0, duration, len(positions)).tolist(),
    )


def assert_feasible(trajectory, limits):
    """Assert every limit holds, and that the waypoints stay ordered in time."""
    usage = compute_limit_usage(trajectory, limits)
    assert usage['joint_velocity'] <= FEASIBLE
    assert usage['joint_acceleration'] <= FEASIBLE
    assert usage['cartesian_speed'] <= FEASIBLE
    assert np.all(np.diff(trajectory.time_from_start) > 0)
    return usage


def uniform_scaling_duration(trajectory, limits):
    """Duration of the baseline retiming: stretch the whole trajectory evenly.

    Scaling every timestamp by a factor s divides joint velocity and Cartesian
    speed by s and acceleration by s², so this is the smallest feasible uniform
    stretch.
    """
    usage = compute_limit_usage(trajectory, limits)
    scale = max(
        usage['joint_velocity'],
        usage['cartesian_speed'],
        sqrt(usage['joint_acceleration']),
        1.0,
    )
    return usage['duration'] * scale


# --- the provided example ---------------------------------------------------


def test_example_violates_only_the_cartesian_limit(example, limits):
    """Guards the premise the other example tests rest on."""
    usage = compute_limit_usage(example, limits)
    assert usage['cartesian_speed'] > 1.0
    assert usage['joint_velocity'] <= 1.0
    assert usage['joint_acceleration'] <= 1.0


def test_example_is_feasible_after_retiming(example, limits):
    assert_feasible(retime_trajectory(example, limits), limits)


def test_example_preserves_path_geometry(example, limits):
    retimed = retime_trajectory(example, limits)

    np.testing.assert_array_equal(
        [wp.positions for wp in retimed.waypoints],
        [wp.positions for wp in example.waypoints],
    )


def test_example_beats_uniform_time_scaling(example, limits):
    retimed = retime_trajectory(example, limits)

    baseline = uniform_scaling_duration(example, limits)
    assert compute_limit_usage(retimed, limits)['duration'] < baseline


def test_example_saturates_a_limit(example, limits):
    """No slack left: the duration cannot shrink without breaking something."""
    usage = assert_feasible(retime_trajectory(example, limits), limits)

    assert max(usage['joint_velocity'],
               usage['joint_acceleration'],
               usage['cartesian_speed']) == SATURATED


def test_retiming_populates_velocities_and_accelerations(example, limits):
    retimed = retime_trajectory(example, limits)

    assert all(wp.velocities is not None for wp in retimed.waypoints)
    assert all(wp.accelerations is not None for wp in retimed.waypoints)


def test_input_trajectory_is_not_mutated(example, limits):
    before = copy.deepcopy(example)

    retime_trajectory(example, limits)

    assert example.time_from_start == before.time_from_start
    assert [wp.positions for wp in example.waypoints] == \
        [wp.positions for wp in before.waypoints]


# --- trajectories where other limits bind -----------------------------------


def wrist_sweep():
    """Wrist joints sweep far and fast while the end-effector barely moves.

    The Cartesian limit is slack here, so the joint velocity limit binds.
    """
    positions = np.zeros((30, 6))
    positions[:, 3] = np.linspace(0.0, 6.0, 30)
    positions[:, 4] = np.linspace(0.0, 3.0, 30)
    return make_trajectory(positions, duration=1.0)


def wrist_reversal():
    """Short out-and-back on a wrist joint: too little travel to reach the
    velocity limit, so the reversal makes the acceleration limit bind."""
    positions = np.zeros((21, 6))
    positions[:, 4] = np.concatenate([
        np.linspace(0.0, 0.5, 11), np.linspace(0.45, 0.0, 10)
    ])
    return make_trajectory(positions, duration=1.0)


def slow_but_feasible():
    """A trajectory that already respects every limit, with room to spare."""
    time = np.linspace(0.0, 1.0, 40)
    positions = np.column_stack(
        [0.2 * np.sin(2 * np.pi * time * (joint + 1) / 4) for joint in range(6)]
    )
    return make_trajectory(positions, duration=60.0)


@pytest.mark.parametrize(
    "trajectory,expected_binding",
    [
        (wrist_sweep(), 'joint_velocity'),
        (wrist_reversal(), 'joint_acceleration'),
        (slow_but_feasible(), None),
    ],
    ids=["joint-velocity-binding", "joint-acceleration-binding", "already-feasible"],
)
def test_non_cartesian_limits_bind_correctly(trajectory, expected_binding, limits):
    usage = assert_feasible(retime_trajectory(trajectory, limits), limits)

    if expected_binding is not None:
        assert usage[expected_binding] == SATURATED


def test_feasible_trajectory_is_sped_up(limits):
    """Minimising duration means a needlessly slow input comes back faster."""
    trajectory = slow_but_feasible()

    retimed = retime_trajectory(trajectory, limits)

    assert compute_limit_usage(retimed, limits)['duration'] < 60.0


def test_tighter_limits_give_a_longer_trajectory(limits):
    trajectory = wrist_sweep()
    halved = RobotLimits(
        max_joint_velocities=[v / 2 for v in limits.max_joint_velocities],
        max_joint_accelerations=[a / 2 for a in limits.max_joint_accelerations],
        max_cartesian_speed=limits.max_cartesian_speed / 2,
    )

    assert (
        compute_limit_usage(retime_trajectory(trajectory, halved), halved)['duration']
        > compute_limit_usage(retime_trajectory(trajectory, limits), limits)['duration']
    )


# --- edge cases -------------------------------------------------------------


def test_two_waypoints(limits):
    trajectory = make_trajectory([[0.0] * 6, [0.5, -0.3, 0.2, 0.1, 0.4, -0.2]])

    assert_feasible(retime_trajectory(trajectory, limits), limits)


def test_three_waypoints(limits):
    trajectory = make_trajectory(
        [[0.0] * 6, [0.3] * 6, [0.6, 0.1, 0.2, 0.3, 0.4, 0.5]]
    )

    assert_feasible(retime_trajectory(trajectory, limits), limits)


def test_single_waypoint_is_returned_unchanged(limits):
    trajectory = JointTrajectory(
        waypoints=[JointState(positions=[0.1] * 6)], time_from_start=[0.0]
    )

    retimed = retime_trajectory(trajectory, limits)

    assert retimed.time_from_start == [0.0]
    assert retimed.waypoints[0].positions == [0.1] * 6


def test_empty_trajectory_is_returned_unchanged(limits):
    retimed = retime_trajectory(
        JointTrajectory(waypoints=[], time_from_start=[]), limits
    )

    assert retimed.waypoints == []
    assert retimed.time_from_start == []


def test_duplicate_waypoints(limits):
    """A repeated waypoint carries no motion; it must not break the path
    parametrization, and the arm has to be able to stop there."""
    positions = np.zeros((10, 6))
    positions[:, 0] = [0.0, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    assert_feasible(retime_trajectory(make_trajectory(positions), limits), limits)


def test_stationary_trajectory(limits):
    """Every waypoint identical: nothing to slow down, but the timestamps still
    have to come back strictly increasing."""
    trajectory = make_trajectory(np.zeros((5, 6)))

    retimed = retime_trajectory(trajectory, limits)

    assert np.all(np.diff(retimed.time_from_start) > 0)


def test_degenerate_input_timestamps(limits):
    """Zero-length input segments must not leak into the output as zero-length
    segments, nor divide by zero on the fallback path."""
    trajectory = make_trajectory(np.linspace(0.0, 0.6, 8)[:, None] * np.ones(6))
    trajectory.time_from_start = [0.0] * 8

    assert_feasible(retime_trajectory(trajectory, limits), limits)


# --- robustness of the two-stage design -------------------------------------


def test_falls_back_to_a_feasible_timing_when_toppra_fails(
    example, limits, monkeypatch
):
    """The feasibility pass is the safety net: it has to hold on its own."""
    monkeypatch.setattr(retimer, "_solve_with_toppra", lambda *args: None)

    assert_feasible(retime_trajectory(example, limits), limits)


@pytest.mark.parametrize("points_per_segment", [1, 2, 8])
def test_result_is_insensitive_to_grid_resolution(
    example, limits, monkeypatch, points_per_segment
):
    """Justifies GRID_POINTS_PER_SEGMENT = 4: the discretization has converged,
    so a finer grid only costs time."""
    reference = compute_limit_usage(retime_trajectory(example, limits), limits)

    monkeypatch.setattr(retimer, "GRID_POINTS_PER_SEGMENT", points_per_segment)
    monkeypatch.setattr(retimer, "MIN_GRID_POINTS", points_per_segment)
    usage = assert_feasible(retime_trajectory(example, limits), limits)

    assert usage['duration'] == pytest.approx(reference['duration'], rel=1e-3)
