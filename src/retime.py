import math
import numpy as np
import rospy

from pybullet_tools.utils import get_distance_fn, get_joint_name, clip, get_max_velocity, get_difference_fn, INF, \
    waypoints_from_path

from scipy.interpolate import CubicSpline # LinearNDInterpolator, NearestNDInterpolator, bisplev, bisplrep, splprep


#ARM_SPEED = 0.15*np.pi # radians / sec
ARM_SPEED = 0.2 # percent


def get_joint_names(body, joints):
    return [get_joint_name(body, joint).encode('ascii')  # ,'ignore')
            for joint in joints]

def get_duration_fn(body, joints, velocities=None, norm=INF):
    if velocities is None:
        velocities = np.array([get_max_velocity(body, joint) for joint in joints])
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        distance = np.array(difference_fn(q2, q1))
        duration = np.divide(distance, velocities)
        return np.linalg.norm(duration, ord=norm)
    return fn

################################################################################

def retime_path(robot, joints, path, speed=ARM_SPEED):
    #duration_fn = get_distance_fn(robot, joints)
    duration_fn = get_duration_fn(robot, joints)
    mid_durations = [duration_fn(*pair) for pair in zip(path[:-1], path[1:])]
    durations = [0.] + mid_durations
    time_from_starts = np.cumsum(durations) / speed
    return time_from_starts

def slow_trajectory(robot, joints, path, **kwargs):
    min_fraction = 0.1 # percentage
    ramp_duration = 1.0 # seconds
    # path = waypoints_from_path(path) # Neither moveit or lula benefit from this

    time_from_starts = retime_path(robot, joints, path, **kwargs)
    mid_times = [np.average(pair) for pair in zip(time_from_starts[:-1], time_from_starts[1:])]
    mid_durations = [t2 - t1 for t1, t2 in zip(time_from_starts[:-1], time_from_starts[1:])]
    new_time_from_starts = [0.]
    for mid_time, mid_duration in zip(mid_times, mid_durations):
        time_from_start = mid_time - time_from_starts[0]
        up_fraction = clip(time_from_start / ramp_duration, min_value=min_fraction, max_value=1.)
        time_from_end = time_from_starts[-1] - mid_time
        down_fraction = clip(time_from_end / ramp_duration, min_value=min_fraction, max_value=1.)
        new_fraction = min(up_fraction, down_fraction)
        new_duration = mid_duration / new_fraction
        #print(new_time_from_starts[-1], up_fraction, down_fraction, new_duration)
        new_time_from_starts.append(new_time_from_starts[-1] + new_duration)
    # print(time_from_starts)
    # print(new_time_from_starts)
    # raw_input('Continue?)
    # time_from_starts = new_time_from_starts
    return new_time_from_starts

#def acceleration_limits(robot, joints, path, speed=ARM_SPEED, **kwargs):
#    # TODO: multiple bodies (such as drawer)
#    # The drawers do actually have velocity limits
#    fraction = 0.25
#    duration_fn = get_duration_fn(robot, joints)
#    max_velocities = speed * np.array([get_max_velocity(robot, joint) for joint in joints])
#    max_accelerations = 2*fraction*max_velocities # TODO: fraction
#    difference_fn = get_difference_fn(robot, joints)
#    differences1 = [difference_fn(q2, q1) for q1, q2 in zip(path[:-1], path[1:])]
#    differences2 = [np.array(d2) - np.array(d1) for d1, d2 in zip(differences1[:-1], differences1[1:])] # TODO: circular case

################################################################################

def ensure_increasing(path, time_from_starts):
    assert len(path) == len(time_from_starts)
    for i in reversed(range(1, len(path))):
        if time_from_starts[i-1] == time_from_starts[i]:
            path.pop(i)
            time_from_starts.pop(i)

def spline_parameterization(robot, joints, path, **kwargs):
    from src.issac import ISSAC_FRANKA_FRAME
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    # TODO: could interpolate each DOF independently
    # Univariate interpolation just means that the input is one dimensional (aka time)
    # The output can be arbitrary dimension
    # Bivariate interpolation has a 2D input space

    # Was initially using scipy 0.17.0
    # https://docs.scipy.org/doc/scipy-0.17.0/reference/interpolate.html
    # https://docs.scipy.org/doc/scipy-0.17.0/reference/tutorial/interpolate.html

    # Upgraded to scipy 0.18.0 to use the CubicSpline method
    # sudo pip2 install scipy==0.18.0
    # https://docs.scipy.org/doc/scipy-0.18.0/reference/interpolate.html

    # BPoly.from_derivatives
    # PPoly.from_spline # PPoly.from_bernstein_basis
    #path = list(path)
    #time_from_starts = retime_path(robot, joints, path, **kwargs)
    #time_from_starts = slow_trajectory(robot, joints, path, **kwargs)
    #ensure_increasing(path, time_from_starts)
    # TODO: interpolate through the waypoints
    path, time_from_starts = retime_trajectory(robot, joints, path)
    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(time_from_starts, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)
    # Could resample at this point
    # TODO: could try passing incorrect accelerations (bounded)

    #for i, t in enumerate(time_from_starts):
    #    print(i, t, path[i], positions(t), velocities(t), accelerations(t))
    #wait_for_user('Continue?')

    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/interpolate/_cubic.py#L75-L158
    trajectory = JointTrajectory()
    trajectory.header.frame_id = ISSAC_FRANKA_FRAME
    trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_names = get_joint_names(robot, joints)
    for t in time_from_starts:
        point = JointTrajectoryPoint()
        point.positions = positions(t) # positions alone is insufficient
        point.velocities = velocities(t)
        point.accelerations = accelerations(t) # accelerations aren't strictly needed
        #point.effort = list(np.ones(len(joints)))
        point.time_from_start = rospy.Duration(t)
        trajectory.points.append(point)
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))
    return trajectory

################################################################################

def linear_parameterization(robot, joints, path, speed=ARM_SPEED):
    from src.issac import ISSAC_FRANKA_FRAME
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    distance_fn = get_distance_fn(robot, joints)
    distances = [0] + [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    time_from_starts = np.cumsum(distances) / speed

    # https://en.wikipedia.org/wiki/Finite_difference
    trajectory = JointTrajectory()
    trajectory.header.frame_id = ISSAC_FRANKA_FRAME
    trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_names = get_joint_names(robot, joints)
    for i in range(len(path)):
       point = JointTrajectoryPoint()
       point.positions = list(path[i])
       # Don't need velocities, accelerations, or efforts
       #vector = np.array(path[i]) - np.array(path[i-1])
       #duration = (time_from_starts[i] - time_from_starts[i-1])
       #point.velocities = list(vector / duration)
       #point.accelerations = list(np.ones(len(joints)))
       #point.effort = list(np.ones(len(joints)))
       point.time_from_start = rospy.Duration(time_from_starts[i])
       trajectory.points.append(point)
    return trajectory

################################################################################

def compute_min_duration(distance, max_velocity, acceleration):
    max_ramp_duration = max_velocity / acceleration
    ramp_distance = 0.5 * acceleration * math.pow(max_ramp_duration, 2)
    remaining_distance = distance - 2 * ramp_distance
    if 0 <= remaining_distance:  # zero acceleration
        remaining_time = remaining_distance / max_velocity
        total_time = 2 * max_ramp_duration + remaining_time
    else:
        half_time = np.sqrt(distance / acceleration)
        total_time = 2 * half_time
    return total_time


def compute_ramp_duration(distance, max_velocity, acceleration, duration):
    discriminant = max(0, math.pow(duration * acceleration, 2) - 4 * distance * acceleration)
    velocity = 0.5 * (duration * acceleration - math.sqrt(discriminant))  # +/-
    #assert velocity <= max_velocity
    ramp_time = velocity / acceleration
    predicted_distance = velocity * (duration - 2 * ramp_time) + acceleration * math.pow(ramp_time, 2)
    assert abs(distance - predicted_distance) < 1e-6
    return ramp_time


def compute_position(ramp_time, max_duration, acceleration, t):
    velocity = acceleration * ramp_time
    max_time = max_duration - 2 * ramp_time
    t1 = clip(t, 0, ramp_time)
    t2 = clip(t - ramp_time, 0, max_time)
    t3 = clip(t - ramp_time - max_time, 0, ramp_time)
    #assert t1 + t2 + t3 == t
    return 0.5 * acceleration * math.pow(t1, 2) + velocity * t2 + \
           (velocity * t3 - 0.5 * acceleration * math.pow(t3, 2))


def retime_trajectory(robot, joints, path, velocity_fraction=0.5, acceleration_fraction=1.0, sample_step=None):
    """
    :param robot:
    :param joints:
    :param path:
    :param velocity_fraction: fraction of max_velocity
    :param acceleration_fraction: fraction of velocity_fraction*max_velocity per second
    :param sample_step:
    :return:
    """
    max_velocities = velocity_fraction * np.array([get_max_velocity(robot, joint) for joint in joints])
    accelerations = max_velocities * acceleration_fraction
    path = waypoints_from_path(path)
    difference_fn = get_difference_fn(robot, joints)
    # TODO: more fine grain when moving longer distances

    # Assuming instant changes in accelerations
    waypoints = [path[0]]
    time_from_starts = [0.]
    for q1, q2 in zip(path[:-1], path[1:]):
        differences = difference_fn(q2, q1)
        distances = np.abs(differences)
        duration = 0
        for idx in range(len(joints)):
            total_time = compute_min_duration(distances[idx], max_velocities[idx], accelerations[idx])
            duration = max(duration, total_time)

        time_from_start = time_from_starts[-1]
        if sample_step is not None:
            ramp_durations = [compute_ramp_duration(distances[idx], max_velocities[idx], accelerations[idx], duration)
                              for idx in range(len(joints))]
            directions = np.sign(differences)
            for t in np.arange(sample_step, duration, sample_step):
                positions = []
                for idx in range(len(joints)):
                    distance = compute_position(ramp_durations[idx], duration, accelerations[idx], t)
                    positions.append(q1[idx] + directions[idx] * distance)
                waypoints.append(positions)
                time_from_starts.append(time_from_start + t)
        waypoints.append(q2)
        time_from_starts.append(time_from_start + duration)
    return waypoints, time_from_starts