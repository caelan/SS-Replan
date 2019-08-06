from __future__ import print_function

import numpy as np
import time
import math

from src.issac import update_robot, ISSAC_FRANKA_FRAME, lookup_pose, \
    ISSAC_PREFIX, ISSAC_WORLD_FRAME, ISSAC_CARTER_FRAME, update_observer
from pybullet_tools.utils import get_distance_fn, get_joint_name, \
    get_max_force, joint_from_name, point_from_pose, wrap_angle, \
    euler_from_quat, quat_from_pose, circular_difference, \
    get_angle, INF, \
    waypoints_from_path, elapsed_time, get_closest_angle_fn, clip, \
    wait_for_user, draw_pose, remove_handles, get_link_pose
from src.utils import WHEEL_JOINTS
from pddlstream.utils import Verbose

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from rospy import Publisher

import rospy
import moveit_msgs.msg

from actionlib import SimpleActionClient
#from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from scipy.interpolate import CubicSpline  # LinearNDInterpolator, NearestNDInterpolator, bisplev, bisplrep, splprep

# control_msgs/GripperCommandAction
# control_msgs/JointTrajectoryAction
# control_msgs/SingleJointPositionAction
# franka_gripper/MoveAction
# moveit_msgs/ExecuteTrajectoryAction
# moveit_msgs/MoveGroupAction

ARM_SPEED = 0.1*np.pi

################################################################################s

def time_parameterization(robot, joints, path, speed=ARM_SPEED):
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
    distance_fn = get_distance_fn(robot, joints)
    distances = [0] + [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    #time_from_starts = np.cumsum(distances) / speed
    time_from_starts = slow_trajectory(robot, joints, path, speed=speed)

    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(time_from_starts, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)

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
    return trajectory

################################################################################s

def ROSPose(pose):
    point, quat = pose
    return Pose(position=Point(*point), orientation=Quaternion(*quat))

def get_joint_names(body, joints):
    return [get_joint_name(body, joint).encode('ascii')  # ,'ignore')
            for joint in joints]

################################################################################s

def joint_state_control(robot, joints, path, domain, moveit, observer,
                        threshold=0.01, timeout=1.0):
    # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/JointState.html
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py#L398
    #max_velocities = np.array([get_max_velocity(robot, joint) for joint in joints])
    #max_forces = np.array([get_max_force(robot, joint) for joint in joints])
    joint_names = get_joint_names(robot, joints)
    distance_fn = get_distance_fn(robot, joints)
    #difference_fn = get_difference_fn(robot, joints)

    #path = waypoints_from_path(path)
    if len(joints) == 2:
        path = path[-1:]
    for i, target_conf in enumerate(path):
        print('Waypoint {} / {}'.format(i, len(path)))
        velocity = None
        #velocity = list(0.25 * np.array(max_velocities))
        joint_state = JointState(name=joint_names, position=list(target_conf), velocity=velocity)
        moveit.joint_cmd_pub.publish(joint_state)
        #rate = rospy.Rate(1000)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and ((rospy.Time.now() - start_time).to_sec() < timeout):
            with Verbose():
                world_state = update_observer(observer)
            robot_entity = world_state.entities[domain.robot]
            #difference = difference_fn(target_conf, robot_entity.q)
            if distance_fn(target_conf, robot_entity.q) < threshold:
                break
            # ee_frame = moveit.forward_kinematics(joint_state.position)
            # moveit.visualizer.send(ee_frame)
            #rate.sleep()
        else:
            print('Failed to reach set point')
    # TODO: send zero velocity command?
    # TODO: return status
    return None

################################################################################s

def publish_display_trajectory(moveit, plan, frame=ISSAC_FRANKA_FRAME):
    trajectory_start = moveit.robot.get_current_state()
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = trajectory_start
    display_trajectory.trajectory.append(plan)
    display_trajectory.trajectory[0].joint_trajectory.header.frame_id = frame
    moveit.display_trajectory_publisher.publish(display_trajectory)

def slow_trajectory(robot, joints, path, speed=ARM_SPEED):
    # Slow down distance or time
    fraction = 0.1 # percentage
    ramp_duration = 1.0 # seconds
    # path = waypoints_from_path(path) # Neither moveit or lula benefit from this
    # max_velocities = np.array([get_max_velocity(robot, joint) for joint in joints])

    distance_fn = get_distance_fn(robot, joints)
    mid_distances = [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    distances = [0] + mid_distances
    time_from_starts = np.cumsum(distances) / speed
    # TODO: make this a separate function?

    mid_times_from_starts = [np.average(pair) for pair in
                             zip(time_from_starts[:-1], time_from_starts[1:])]
    # https://en.wikipedia.org/wiki/Finite_difference
    new_time_from_starts = [0]
    for mid_time_from_start, distance in zip(mid_times_from_starts, mid_distances):
        up_fraction = clip((mid_time_from_start - time_from_starts[0]) / ramp_duration,
                           min_value=fraction, max_value=1.)
        down_fraction = clip((time_from_starts[-1] - mid_time_from_start) / ramp_duration,
                             min_value=fraction, max_value=1.)
        new_speed = speed * min(up_fraction, down_fraction)
        print(distance, new_speed)
        duration = distance / new_speed
        new_time_from_starts.append(new_time_from_starts[-1] + duration)
    # print(time_from_starts)
    # print(new_time_from_starts)
    # raw_input('Continue?)
    # time_from_starts = new_time_from_starts
    return new_time_from_starts

def moveit_control(robot, joints, path, moveit, observer, **kwargs):
    #path = waypoints_from_path(path)
    #if moveit.use_lula:
    #    speed = 0.5*speed

    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
    action_topic = 'position_joint_trajectory_controller/follow_joint_trajectory'
    # /move_base_simple/goal
    # /execute_trajectory/goal
    # /position_joint_trajectory_controller/command
    client = SimpleActionClient(action_topic, FollowJointTrajectoryAction)
    print('Starting', action_topic)
    client.wait_for_server()
    client.cancel_all_goals()
    print('Finished', action_topic)
    # TODO: create this action client once

    # Moveit's trajectories
    # rostopic echo /execute_trajectory/goal
    # About 1 waypoint per second
    # Start and end velocities are zero
    # Accelerations are about zero (except at the start and end)

    time_from_starts = slow_trajectory(robot, joints, path, **kwargs)
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

    print('Following {} waypoints in {:.3f} seconds'.format(
        len(path), time_from_starts[-1]))
    # path_tolerance, goal_tolerance, goal_time_tolerance
    # http://docs.ros.org/diamondback/api/control_msgs/html/msg/FollowJointTrajectoryGoal.html

    goal = FollowJointTrajectoryGoal(trajectory=trajectory)
    start_time = time.time()
    client.send_goal_and_wait(goal) # send_goal_and_wait
    client.get_result()
    print('Execution took {:.3f} seconds'.format(elapsed_time(start_time)))
    return

    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/interpolator.py
    # Only position, time_from_start, and velocity are used
    plan = RobotTrajectory(joint_trajectory=trajectory)
    if moveit.use_lula:
        world_state = update_observer(observer)
        suppress_all(world_state)
    moveit.verbose = False
    moveit.last_ik = plan.joint_trajectory.points[-1].positions
    #moveit.dilation = 2
    start_time = time.time()
    # /move_group/display_planned_path
    # TODO: display base motions?
    #publish_display_trajectory(moveit, plan) # TODO: get this in the base_link frame
    with Verbose():
        moveit.execute(plan, required_orig_err=0.005, timeout=5.0,
                       publish_display_trajectory=False)
        #moveit.go_local(q=path[-1], ...)
    print('Execution took {:.3f} seconds'.format(elapsed_time(start_time)))

def suppress_all(world_state):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L138
    for name, entity in world_state.entities.items():
       if entity.controllable_object is not None:
           #entity.controllable_object.unsuppress()
           entity.controllable_object.suppress() # Propagate to parents?
       # entity.set_detached()
       #domain.attachments[actor] = goal

def lula_control(world, path, domain, observer, world_state):
    suppress_all(world_state)
    robot_entity = domain.get_robot()
    franka = robot_entity.robot
    #franka.set_speed(speed_level='slow')
    for i, positions in enumerate(path):
       print('{}/{}'.format(i, len(path)))
       timeout = 10.0 if i == len(positions)-1 else 2.0
       franka.end_effector.go_config(positions, err_thresh=0.05,
           wait_for_target=True, wait_time=timeout, verbose=True) # TODO: go_guided/go_long_range
       observer.update()
       update_robot(world, domain, observer)
       #wait_for_duration(1e-3)
       # TODO: attachments

################################################################################s

def base_control(world, goal_values, moveit, observer,
                 timeout=30, sleep=1.0, verbose=False):
    # https://github.mit.edu/caelan/ROS/blob/4f375489a4b3bac7c7a0451fe30e35ba02e6302f/base_navigation.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain_msgs/msg/Goal.msg
    #joints = joints_from_names(world.robot, WHEEL_JOINTS)
    #max_velocities = np.array([get_max_velocity(world.robot, joint) for joint in joints])
    assert len(goal_values) == 3

    unit_forward = np.array([-1, -1]) # Negative seems to be forward
    unit_right = np.array([-1, +1])

    min_speed = 12 # Technically 9.085
    max_speed = 30 # radians per second

    ramp_down_pos = 0.5 # meters
    ramp_down_yaw = math.radians(25) # radians

    #dump_body(world.robot)
    # https://github.mit.edu/caelan/base-trajectory-action/blob/master/src/base_trajectory.cpp
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/scripts/set_base_joint_states.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter.py
    linear_threshold = 0.05
    angular_threshold = math.radians(3)

    # TODO: open-loop odometry (use wheel encoders)
    # TODO: ensure that we are rotating about the correct axis (i.e. base_joints aligned with axis)
    # TODO: subscribe to /robot/joint_states for wheel positions

    # linear/angular
    closet_angle_fn = get_closest_angle_fn(world.robot, world.base_joints)
    reached_goal_pos = False
    goal_pos = np.array(goal_values[:2])
    goal_yaw = goal_values[2]
    goal_pos_error = INF
    goal_yaw_error = INF
    pub = moveit.joint_cmd_pub
    #pub = rospy.Publisher(CONTROL_TOPIC, JointState, queue_size=1)
    #try:
    #rate = rospy.Rate(100)
    start_time = rospy.Time.now()
    while (not rospy.is_shutdown()) and ((rospy.Time.now() - start_time).to_sec() < timeout):
        #world_state = observer.observe()
        #robot_entity = world_state.entities[domain.robot]
        #print(robot_entity.carter_pos, robot_entity.carter_vel) # zeros
        #robot_entity.base_link
        #print(pose_from_tform(robot_entity.pose))

        base_pose = lookup_pose(observer.tf_listener, ISSAC_PREFIX + ISSAC_CARTER_FRAME)
        current_values = np.array(point_from_pose(base_pose))
        current_pos = current_values[:2]
        x, y, _ = point_from_pose(base_pose)
        _, _, current_yaw = map(wrap_angle, euler_from_quat(quat_from_pose(base_pose)))
        if verbose:
            print('x={:.3f}, y={:.3f}, yaw={:.3f}'.format(x, y, current_yaw))

        vector_yaw = get_angle(current_pos, goal_pos)
        movement_yaw, _ = closet_angle_fn(current_values, goal_values)
        movement_yaw_error = abs(circular_difference(movement_yaw, current_yaw))
        delta_pos = goal_pos - current_pos
        goal_pos_error = np.linalg.norm(delta_pos)
        print('Linear error: {:.3f} ({:.3f})'.format(
            goal_pos_error, linear_threshold))
        goal_yaw_error = abs(circular_difference(goal_yaw, current_yaw))
        print('Angular error: {:.1f} ({:.1f})'.format(
            *map(math.degrees, [goal_yaw_error, angular_threshold])))

        target_yaw = None
        if reached_goal_pos or (goal_pos_error < linear_threshold):
            reached_goal_pos = True
            print('Rotating towards goal yaw')
            if goal_yaw_error < angular_threshold:
                break
            target_yaw = goal_yaw
        else:
            if movement_yaw_error < angular_threshold:
                print('Moving towards goal position')
            else:
                print('Rotating towards movement yaw')
                target_yaw = movement_yaw

        if target_yaw is None:
            # target_pos = goal_pos
            if verbose:
                print('Linear delta:', delta_pos)
            pos_fraction = min(1, goal_pos_error / ramp_down_pos)
            speed = (1 - pos_fraction) * min_speed + pos_fraction * max_speed
            sign = math.cos(movement_yaw - vector_yaw)
            joint_velocities = sign * speed * unit_forward
        else:
            delta_yaw = circular_difference(target_yaw, current_yaw)
            if verbose:
                print('Angular delta:', delta_yaw)
            #print(robot_entity.carter_interface) # None
            #print(robot_entity.joints) # Only arm joints
            #print(robot_entity.q, robot_entity.dq)
            #speed = max_speed
            #speed = min(max_speed, 60*np.abs(delta_yaw) / np.pi)
            yaw_fraction = min(1., abs(delta_yaw) / ramp_down_yaw)
            speed = (1 - yaw_fraction)*min_speed + yaw_fraction*max_speed
            joint_velocities = - np.sign(delta_yaw) * speed * unit_right
        if verbose:
            print('Velocities:', joint_velocities.round(3).tolist())
        #world_state = observer.observe()
        #update_robot(world, domain, observer, world_state)
        pub.publish(JointState(header=Header(), name=WHEEL_JOINTS, velocity=list(joint_velocities)))
        #rate.sleep()
    #except KeyboardInterrupt as e:
    #    pass
    #except rospy.ServiceException as e:
    #    rospy.logerr("Service call failed: %s" % e)
    #finally:
    joint_velocities = np.zeros(len(WHEEL_JOINTS))
    if verbose:
        print('Velocities:', joint_velocities.round(3).tolist())
    start_time = rospy.Time.now()
    while (not rospy.is_shutdown()) and ((rospy.Time.now() - start_time).to_sec() < sleep):
        # TODO: actually query the state of the wheel joints
        pub.publish(JointState(header=Header(), name=WHEEL_JOINTS, velocity=list(joint_velocities)))
    print('Final linear error: {:.3f} ({:.3f})'.format(goal_pos_error, linear_threshold))
    print('Final angular error: {:.1f} ({:.1f})'.format(
        *map(math.degrees, [goal_yaw_error, angular_threshold])))
    return False

def follow_base_trajectory(world, path, moveit, observer, **kwargs):
    path = waypoints_from_path(path)
    goal_pub = Publisher("~base_goal", PoseStamped, queue_size=1)
    handles = []
    for i, base_values in enumerate(path):
        print('Waypoint {} / {}'.format(i, len(path)))
        world.set_base_conf(base_values)
        base_pose = get_link_pose(world.robot, world.base_link)
        #base_pose = pose_from_base_values(base_values)
        remove_handles(handles)
        handles.extend(draw_pose(base_pose, length=1))
        pose = PoseStamped()
        pose.header.frame_id = ISSAC_WORLD_FRAME
        pose.pose = ROSPose(base_pose)
        goal_pub.publish(pose)
        #wait_for_user()
        base_control(world, base_values, moveit, observer, **kwargs)

################################################################################s

#FINGER_EFFORT_LIMIT = 20
#FINGER_VELOCITY_LIMIT = 0.2

def open_gripper(robot, moveit, effort=20, sleep=1.0):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L155
    # robot_entity = domain.get_robot()
    # franka = robot_entity.robot
    # gripper = franka.end_effector.gripper
    # gripper.open(speed=.2, actuate_gripper=True, wait=True)
    # update_robot(self.world, domain, observer.observe())
    # time.sleep(1.0)
    #moveit.open_gripper(speed=0.1, sleep=0.2, wait=True)
    joint_state = JointState(name=moveit.gripper.joints, position=moveit.gripper.open_positions)
    if effort is not None:
        gripper_joint = joint_from_name(robot, moveit.gripper.joints[0])
        max_effort = get_max_force(robot, gripper_joint)
        effort = max(0, min(effort, max_effort))
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None

def close_gripper(robot, moveit, effort=20, sleep=1.0):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L218
    # controllable_object is not needed for joint positions
    # TODO: attach_obj
    # robot_entity = domain.get_robot()
    # franka = robot_entity.robot
    # gripper = franka.end_effector.gripper
    # gripper.close(attach_obj=None, speed=.2, force=40., actuate_gripper=True, wait=True)
    # update_robot(self.world, domain, observer, observer.observe())
    # time.sleep(1.0)
    # TODO: only sleep is used by close_gripper and open_gripper...
    #moveit.close_gripper(controllable_object=None, speed=0.1, force=40., sleep=0.2, wait=True)
    joint_state = JointState(name=moveit.gripper.joints, position=moveit.gripper.closed_positions)
    if effort is not None:
        gripper_joint = joint_from_name(robot, moveit.gripper.joints[0])
        max_effort = get_max_force(robot, gripper_joint)
        effort = max(0, min(effort, max_effort))
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None
