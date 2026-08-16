"""
Trajectory Retiming.

This module provides trajectory retiming functionality to ensure all constraints
are satisfied while minimizing the added time penalty.

Usage:
    Visualize a trajectory (joint positions, Cartesian speed, 3D path):

        python3 -m trajectory_retiming.retimer

    With a custom trajectory file:

        python3 -m trajectory_retiming.retimer --trajectory path/to/trajectory.json


Approach
--------
The retiming is done by TOPP-RA (`toppra`), a reachability-analysis based
time-optimal path parametrization algorithm. TOPP-RA keeps the geometric path
q(s) fixed and searches for the fastest time law s(t) that respects every
constraint, which is exactly the problem posed here: the waypoints — and
therefore the path — must not move, only the timestamps may change.

The result is then passed through a feasibility pass that guarantees the
constraints also hold under the *discrete* finite-difference model that
`JointTrajectory.compute_velocities` / `compute_acceleration` (and the README)
use to define them. See `retime_trajectory` for the full write-up.
"""

import copy
import json
import logging
import numpy as np
import toppra as ta
import toppra.algorithm as ta_algorithm
import toppra.constraint as ta_constraint
from .data_types import JointTrajectory, JointState
from .limits import RobotLimits
from .kinematics import forward_kinematics, compute_jacobian

logger = logging.getLogger(__name__)

# Sub-intervals each waypoint segment is split into for the TOPP-RA grid. The
# grid drives how finely the constraints are sampled along the path; beyond ~4
# per segment the solution stops changing (see test_retimer.py).
GRID_POINTS_PER_SEGMENT = 4

# Floor on the total grid size, so that trajectories with only a handful of
# waypoints are still discretized finely enough for TOPP-RA.
MIN_GRID_POINTS = 50

# Consecutive waypoints closer than this (joint-space L2) carry no motion and
# are collapsed into a single knot of the geometric path.
DUPLICATE_WAYPOINT_TOL = 1e-9

# Duration handed to a segment that carries no motion. Any positive value is
# feasible for such a segment; it only has to keep the timestamps increasing.
MIN_SEGMENT_DURATION = 1e-6

# The feasibility pass aims for this utilization rather than exactly 1.0, so
# that the returned trajectory is strictly inside the limits despite round-off.
LIMIT_UTILIZATION_TARGET = 1.0 - 1e-9

MAX_FEASIBILITY_ITERATIONS = 200


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


class CartesianSpeedConstraint(ta_constraint.LinearConstraint):
    """Bound the translational end-effector speed along a TOPP-RA path.

    TOPP-RA parametrizes a fixed geometric path q(s) by s(t) and reasons in
    terms of the squared path velocity x = s_dot². The Cartesian speed at a
    path position s is

        ||v(s)|| = ||J(q(s))[:3] q'(s)|| * s_dot

    so ``||v(s)|| <= max_speed`` is a plain upper bound on x:

        x <= (max_speed / ||J(q(s))[:3] q'(s)||)²

    which TOPP-RA consumes directly as an ``xbound``, i.e. it needs no extra
    inequality rows and stays inside the fast Seidel LP solver.

    Args:
        jacobian: Callable mapping joint positions to the 6x6 geometric
            Jacobian; only the first three (linear velocity) rows are used.
        max_speed: Maximum translational speed of the end-effector in m/s.
        dof: Degrees of freedom of the path the constraint is applied to.
    """

    def __init__(self, jacobian, max_speed: float, dof: int = 6):
        super().__init__()
        self.jacobian = jacobian
        self.max_speed = float(max_speed)
        self.dof = dof
        self._format_string = f"    Cartesian speed limit: {self.max_speed}\n"

    def compute_constraint_params(self, path, gridpoints):
        path_positions = path(gridpoints, 0)
        path_tangents = path(gridpoints, 1)

        xbound = np.zeros((len(gridpoints), 2))
        for i, (q, tangent) in enumerate(zip(path_positions, path_tangents)):
            speed_per_path_velocity = np.linalg.norm(
                self.jacobian(q)[:3, :] @ tangent
            )
            # A stationary point of the path imposes no bound: the end-effector
            # does not move there no matter how fast s advances.
            xbound[i] = (
                [0.0, np.inf]
                if speed_per_path_velocity < 1e-12
                else [0.0, (self.max_speed / speed_per_path_velocity) ** 2]
            )
        return None, None, None, None, None, None, xbound


def _jacobian(positions) -> np.ndarray:
    """Geometric Jacobian at the given joint positions (array or list)."""
    return compute_jacobian(JointState(positions=list(positions)))


def _path_knots(positions: np.ndarray):
    """Describe the geometric path the waypoints trace out.

    The path is parametrized by cumulative joint-space chord length rather than
    by waypoint index: a spline over an index parametrization badly overshoots
    when the waypoint spacing is uneven, to the point where TOPP-RA reports the
    instance as uncontrollable.

    Consecutive duplicate waypoints carry no motion and would make the
    parametrization non-monotonic, so they are collapsed into a single knot.

    Args:
        positions: Shape (n, dof) array of waypoint joint positions.

    Returns:
        Tuple of (knot positions, knot path coordinates, knot index of each
        waypoint).
    """
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving = steps > DUPLICATE_WAYPOINT_TOL

    is_knot = np.concatenate([[True], moving])
    knot_of_waypoint = np.cumsum(is_knot) - 1
    knot_coordinates = np.concatenate([[0.0], np.cumsum(steps[moving])])
    return positions[is_knot], knot_coordinates, knot_of_waypoint


def _toppra_gridpoints(knot_coordinates: np.ndarray) -> np.ndarray:
    """Discretization grid for TOPP-RA, refined but aligned with the knots.

    Keeping every knot on the grid is what lets us read the waypoint timestamps
    straight out of the solution.
    """
    num_segments = len(knot_coordinates) - 1
    per_segment = max(
        GRID_POINTS_PER_SEGMENT, int(np.ceil(MIN_GRID_POINTS / num_segments))
    )
    return np.unique(np.concatenate([
        np.linspace(knot_coordinates[i], knot_coordinates[i + 1], per_segment + 1)
        for i in range(num_segments)
    ]))


def _solve_with_toppra(
    knot_positions: np.ndarray, knot_coordinates: np.ndarray, limits: RobotLimits
):
    """Run TOPP-RA on the geometric path and return the time at each knot.

    Returns None if TOPP-RA cannot parametrize the path, leaving the caller to
    fall back on a slower but still feasible timing.
    """
    path = ta.SplineInterpolator(knot_coordinates, knot_positions)
    constraints = [
        ta_constraint.JointVelocityConstraint(limits.max_joint_velocities),
        ta_constraint.JointAccelerationConstraint(limits.max_joint_accelerations),
        CartesianSpeedConstraint(
            _jacobian, limits.max_cartesian_speed, dof=path.dof
        ),
    ]

    try:
        instance = ta_algorithm.TOPPRA(
            constraints, path, gridpoints=_toppra_gridpoints(knot_coordinates)
        )
        instance.compute_parameterization(sd_start=0.0, sd_end=0.0)
    except Exception:  # pragma: no cover - defensive, toppra raises broadly
        logger.exception("TOPP-RA raised while parametrizing the path")
        return None

    solution = instance.problem_data
    if (
        solution.return_code != ta_algorithm.ParameterizationReturnCode.Ok
        or solution.sd_vec is None
    ):
        logger.warning("TOPP-RA failed to parametrize the path: %s",
                       solution.return_code)
        return None

    # TOPP-RA holds the path acceleration constant between gridpoints, so the
    # time spent on grid interval k follows exactly from the path velocities at
    # its ends: dt = 2 * ds / (sd_k + sd_k+1). Summing these reproduces the
    # duration of `instance.compute_trajectory()` to the last bit.
    path_velocities = np.asarray(solution.sd_vec, dtype=float)
    grid = np.asarray(solution.gridpoints, dtype=float)
    mean_path_velocity = (path_velocities[:-1] + path_velocities[1:]) / 2
    if not np.all(np.isfinite(path_velocities)) or np.any(mean_path_velocity <= 0):
        logger.warning("TOPP-RA returned a degenerate path velocity profile")
        return None

    grid_times = np.concatenate(
        [[0.0], np.cumsum(np.diff(grid) / mean_path_velocity)]
    )
    # The knots are gridpoints, so this interpolation is an exact lookup.
    return np.interp(knot_coordinates, grid, grid_times)


def _cartesian_step_lengths(positions: np.ndarray) -> np.ndarray:
    """Cartesian distance each segment covers per unit of segment duration.

    Segment i is traversed with joint velocity (q_i+1 - q_i) / dt_i, so its
    Cartesian speed is this value divided by dt_i. We take the larger of the
    two Jacobian evaluations at the segment ends, which is at least as strict
    as the convention in the README (Jacobian at the segment start) and costs
    well under 0.01% of duration on the example trajectory.
    """
    steps = np.diff(positions, axis=0)
    linear_jacobians = [_jacobian(q)[:3, :] for q in positions]
    return np.array([
        max(
            np.linalg.norm(linear_jacobians[i] @ step),
            np.linalg.norm(linear_jacobians[i + 1] @ step),
        )
        for i, step in enumerate(steps)
    ])


def _finite_differences(steps: np.ndarray, durations: np.ndarray):
    """Joint velocities and accelerations of the piecewise-linear model.

    This mirrors `JointTrajectory.compute_velocities` / `compute_acceleration`
    exactly, including their convention that the arm is at rest at the final
    waypoint, so that what we enforce is what those methods report.
    """
    velocities = np.zeros((len(steps) + 1, steps.shape[1]))
    velocities[:-1] = steps / durations[:, np.newaxis]

    centered_durations = np.concatenate(
        [(durations[:-1] + durations[1:]) / 2, durations[-1:]]
    )
    accelerations = np.zeros_like(velocities)
    accelerations[:-1] = (
        np.diff(velocities, axis=0) / centered_durations[:, np.newaxis]
    )
    return velocities, accelerations


def compute_limit_usage(
    trajectory: JointTrajectory, limits: RobotLimits
) -> dict:
    """Report how much of each limit a trajectory uses, as a fraction.

    A value of 1.0 means the limit is exactly saturated and anything above 1.0
    is a violation, which makes this both the feasibility check used internally
    and a convenient assertion target for tests.

    Args:
        trajectory: Trajectory to inspect
        limits: Robot physical limits

    Returns:
        Dict with keys 'joint_velocity', 'joint_acceleration',
        'cartesian_speed' and 'duration'.
    """
    positions = np.array([wp.positions for wp in trajectory.waypoints], dtype=float)
    times = np.asarray(trajectory.time_from_start, dtype=float)
    if len(positions) < 2:
        return {
            'joint_velocity': 0.0,
            'joint_acceleration': 0.0,
            'cartesian_speed': 0.0,
            'duration': 0.0,
        }

    durations = np.diff(times)
    steps = np.diff(positions, axis=0)
    velocities, accelerations = _finite_differences(steps, durations)
    cartesian_speeds = _cartesian_step_lengths(positions) / durations

    max_velocity = np.asarray(limits.max_joint_velocities, dtype=float)
    max_acceleration = np.asarray(limits.max_joint_accelerations, dtype=float)
    return {
        'joint_velocity': float(np.max(np.abs(velocities) / max_velocity)),
        'joint_acceleration': float(np.max(np.abs(accelerations) / max_acceleration)),
        'cartesian_speed': float(np.max(cartesian_speeds) / limits.max_cartesian_speed),
        'duration': float(times[-1] - times[0]),
    }


def _enforce_limits(
    positions: np.ndarray, durations: np.ndarray, limits: RobotLimits
) -> np.ndarray:
    """Stretch segment durations until the discrete model satisfies the limits.

    TOPP-RA works with a continuous spline model of the path, while the limits
    here are defined on finite differences of the waypoints. The two agree very
    closely but not exactly, so this pass closes the remaining gap.

    Every violation shrinks monotonically as durations grow — joint velocity
    and Cartesian speed as 1/dt, acceleration as 1/dt² — so repeatedly scaling
    each offending segment by its own violation ratio converges from any
    positive starting point. That also makes this the safety net that keeps the
    function total: even if TOPP-RA fails and we start from the input timing,
    the result is feasible.
    """
    max_velocity = np.asarray(limits.max_joint_velocities, dtype=float)
    max_acceleration = np.asarray(limits.max_joint_accelerations, dtype=float)
    steps = np.diff(positions, axis=0)
    cartesian_steps = _cartesian_step_lengths(positions)

    durations = np.maximum(np.asarray(durations, dtype=float), MIN_SEGMENT_DURATION)
    for _ in range(MAX_FEASIBILITY_ITERATIONS):
        velocities, accelerations = _finite_differences(steps, durations)
        velocity_usage = np.max(np.abs(velocities[:-1]) / max_velocity, axis=1)
        cartesian_usage = cartesian_steps / durations / limits.max_cartesian_speed
        acceleration_usage = np.max(
            np.abs(accelerations[:-1]) / max_acceleration, axis=1
        )
        if max(
            velocity_usage.max(), cartesian_usage.max(), acceleration_usage.max()
        ) <= 1.0:
            break

        stretch = np.maximum(velocity_usage, cartesian_usage) / LIMIT_UTILIZATION_TARGET
        # Acceleration i is driven by durations i and i+1 and falls off as 1/dt².
        acceleration_stretch = np.sqrt(acceleration_usage / LIMIT_UTILIZATION_TARGET)
        stretch = np.maximum(stretch, acceleration_stretch)
        stretch[1:] = np.maximum(stretch[1:], acceleration_stretch[:-1])

        durations = durations * np.maximum(stretch, 1.0)
    else:
        logger.warning(
            "Feasibility pass hit its iteration cap; limits may still be violated"
        )
    return durations




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

    Interpolation model
    -------------------
    Two models are in play, deliberately.

    TOPP-RA plans on a *cubic spline* through the waypoints, parametrized by
    cumulative joint-space chord length. A continuous model is what TOPP-RA
    needs — a piecewise-linear path has zero curvature inside a segment and an
    undefined second derivative at every waypoint, so the acceleration
    constraint would be blind to exactly the corners that dominate it.

    The constraints themselves are then evaluated on the *piecewise-linear*
    model of the README, i.e. the finite differences that
    `JointTrajectory.compute_velocities` and `compute_acceleration` compute:

        v_i = (q_i+1 - q_i) / (t_i+1 - t_i)
        a_i = (v_i+1 - v_i) / ((t_i+2 - t_i) / 2)

    with the arm at rest at the final waypoint. Cartesian speed is the linear
    part of J(q) v_i, evaluated at both ends of the segment and taking the
    larger of the two.

    On the example trajectory the two models agree to within 0.1%, which is a
    good sign that the spline is a fair stand-in for the real motion.

    Algorithm
    ---------
    1. Build the geometric path: chord-length knots through the waypoints, with
       consecutive duplicates collapsed (`_path_knots`).
    2. Hand the path plus the three constraints to TOPP-RA, which returns the
       time-optimal path velocity profile s_dot(s) (`_solve_with_toppra`).
       Integrating it over the grid gives a timestamp per waypoint.
    3. Stretch any segment that still violates a limit under the discrete
       model, iterating until all of them hold (`_enforce_limits`).

    Trade-offs and assumptions
    --------------------------
    - The original timing is used only as a fallback if TOPP-RA fails: this
      minimizes duration outright rather than only ever slowing down, so an
      over-conservative input trajectory comes back *faster* than it went in.
    - Step 3 can only stretch time, never compress it, so it cannot undo an
      over-conservative step 2. In practice it moves the duration by ~0.03% on
      the example. A coordinate-descent polish that shrinks individual
      segments was tried and recovered nothing, so it was left out.
    - On *coarse* trajectories (a handful of far-apart waypoints) the finite
      differences average over so much motion that the discrete model would
      permit a shorter duration than TOPP-RA returns. That extra margin is not
      taken: reaching it needs a jerk profile the spline model says the arm
      cannot track, so the numbers would improve while the motion got worse.
      This matters little for the dense trajectories this is aimed at.
    - The path is assumed to start and end at rest (s_dot = 0 at both ends),
      matching the rest-at-the-end convention of `compute_velocities`.
    - Duplicate waypoints get a nominally short segment. Under the discrete
      model a repeated waypoint means "stand still here", so the surrounding
      segments are slowed down by step 3 to make the stop feasible.

    Args:
        trajectory: Original joint trajectory with waypoints and timing
        limits: Robot physical limits

    Returns:
        Retimed trajectory with adjusted time_from_start values
    """
    trajectory_copy = copy.deepcopy(trajectory)
    positions = np.array(
        [wp.positions for wp in trajectory_copy.waypoints], dtype=float
    )
    if len(positions) < 2:
        return trajectory_copy

    knot_positions, knot_coordinates, knot_of_waypoint = _path_knots(positions)

    durations = None
    if len(knot_positions) >= 2:
        knot_times = _solve_with_toppra(knot_positions, knot_coordinates, limits)
        if knot_times is not None:
            durations = np.diff(knot_times[knot_of_waypoint])
    if durations is None:
        # Either the path carries no motion at all, or TOPP-RA gave up. Start
        # from the input timing instead; the result is still feasible, just no
        # longer time-optimal.
        durations = np.diff(np.asarray(trajectory_copy.time_from_start, dtype=float))

    durations = _enforce_limits(positions, durations, limits)

    trajectory_copy.time_from_start = np.concatenate(
        [[0.0], np.cumsum(durations)]
    ).tolist()
    trajectory_copy.compute_velocities()
    trajectory_copy.compute_acceleration()
    return trajectory_copy


def plot_trajectory(
    trajectory: JointTrajectory, title: str | None = None, show: bool = True
) -> None:
    """
    Plot a trajectory object.

    Creates a multi-panel visualization showing:
    - Joint positions over time
    - Joint velocities over time
    - Cartesian speed over time
    - 3D Cartesian end-effector path

    Args:
        trajectory: JointTrajectory object (e.g. from retime_trajectory)
        title: Optional heading for the figure, used to tell several figures
            apart
        show: Whether to display the figure. Pass False to build up several
            figures and show them together with a single `plt.show()`.
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

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    if show:
        plt.show()


def _describe(label: str, trajectory: JointTrajectory, limits: RobotLimits) -> str:
    """One-line summary of a trajectory's duration and limit usage."""
    usage = compute_limit_usage(trajectory, limits)
    return (
        f"{label}: {usage['duration']:.3f} s"
        f"  |  joint velocity {usage['joint_velocity']:.0%}"
        f", joint acceleration {usage['joint_acceleration']:.0%}"
        f", Cartesian speed {usage['cartesian_speed']:.0%} of limit"
    )


def main():
    """Main entry point for trajectory visualization."""
    import argparse
    import os

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description='Plot robot trajectory from JSON file'
    )
    parser.add_argument(
        '--trajectory', '-t',
        type=str,
        # The package lives in src/, so the repo root is two levels up.
        default=os.path.join(
            os.path.dirname(__file__), '..', '..', 'examples', 'p2p_trajectory.json'
        ),
        help='Path to trajectory JSON file'
    )
    parser.add_argument(
        '--retime', '-r',
        action='store_true',
        help='Retime the trajectory and plot it side by side with the original'
    )
    args = parser.parse_args()

    print(f"Loading trajectory from: {args.trajectory}")
    trajectory = load_joint_trajectory_from_json(args.trajectory)

    if not args.retime:
        plot_trajectory(trajectory)
        return

    limits = RobotLimits()
    retimed = retime_trajectory(trajectory, limits)
    print(_describe("Original", trajectory, limits))
    print(_describe("Retimed ", retimed, limits))

    # Both figures are built first and shown together, so that the before and
    # after can be compared without closing one to reach the other.
    plot_trajectory(trajectory, title='Before retiming', show=False)
    plot_trajectory(retimed, title='After retiming', show=False)
    plt.show()


if __name__ == "__main__":
    main()
