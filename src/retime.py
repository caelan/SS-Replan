import numpy as np
import rospy

from pybullet_tools.utils import get_distance_fn, get_joint_name, clip
from src.issac import ISSAC_FRANKA_FRAME

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from scipy.interpolate import CubicSpline # LinearNDInterpolator, NearestNDInterpolator, bisplev, bisplrep, splprep


ARM_SPEED = 0.2*np.pi # 0.25


def get_joint_names(body, joints):
    return [get_joint_name(body, joint).encode('ascii')  # ,'ignore')
            for joint in joints]

################################################################################

def slow_trajectory(robot, joints, path, speed=ARM_SPEED):
    fraction = 0.1 # percentage
    ramp_duration = 1.0 # seconds
    # path = waypoints_from_path(path) # Neither moveit or lula benefit from this
    # max_velocities = np.array([get_max_velocity(robot, joint) for joint in joints])

    distance_fn = get_distance_fn(robot, joints)
    mid_distances = [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    distances = [0] + mid_distances
    time_from_starts = np.cumsum(distances) / speed
    # TODO: make this a separate function?

    # TODO: ensure that times are strictly increasing
    mid_times_from_starts = [np.average(pair) for pair in
                             zip(time_from_starts[:-1], time_from_starts[1:])]
    new_time_from_starts = [0.]
    for mid_time_from_start, distance in zip(mid_times_from_starts, mid_distances):
        # TODO: slow down based on distance or time
        up_fraction = clip((mid_time_from_start - time_from_starts[0]) / ramp_duration,
                           min_value=fraction, max_value=1.)
        down_fraction = clip((time_from_starts[-1] - mid_time_from_start) / ramp_duration,
                             min_value=fraction, max_value=1.)
        new_speed = speed * min(up_fraction, down_fraction)
        new_duration = distance / new_speed
        #print(new_time_from_starts[-1], up_fraction, down_fraction, new_duration)
        new_time_from_starts.append(new_time_from_starts[-1] + new_duration)
    # print(time_from_starts)
    # print(new_time_from_starts)
    # raw_input('Continue?)
    # time_from_starts = new_time_from_starts
    return new_time_from_starts

################################################################################

def spline_parameterization(robot, joints, path, **kwargs):
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
    path = list(path)
    time_from_starts = slow_trajectory(robot, joints, path, **kwargs)
    for i in reversed(range(1, len(path))):
        if time_from_starts[i-1] == time_from_starts[i]:
            path.pop(i)
            time_from_starts.pop(i)
    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(time_from_starts, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)
    # Could resample at this point

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
        point.positions = positions(t)
        point.velocities = velocities(t)
        point.accelerations = accelerations(t)
        #point.effort = list(np.ones(len(joints)))
        point.time_from_start = rospy.Duration(t)
        trajectory.points.append(point)
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))
    return trajectory

################################################################################

def linear_parameterization(robot, joints, path, speed=ARM_SPEED):
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