from __future__ import print_function

import time
import rospy
import moveit_msgs.msg

from src.retime import spline_parameterization, get_joint_names
from src.issac import update_robot, ISSAC_FRANKA_FRAME, update_observer
from pybullet_tools.utils import get_distance_fn, get_max_force, joint_from_name, elapsed_time, wait_for_user
from pddlstream.utils import Verbose

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from rospy import Publisher

from actionlib import SimpleActionClient
#from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, JointTrajectoryAction, \
    FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal, JointTrajectoryActionGoal, \
    JointTrajectoryGoal, GripperCommandActionGoal, GripperCommandAction, GripperCommandGoal
from lula_controller_msgs.msg import JointPosVelAccCommand

# control_msgs/GripperCommandAction
# control_msgs/JointTrajectoryAction
# control_msgs/SingleJointPositionAction
# franka_gripper/MoveAction
# moveit_msgs/ExecuteTrajectoryAction
# moveit_msgs/MoveGroupAction

def ROSPose(pose):
    point, quat = pose
    return Pose(position=Point(*point), orientation=Quaternion(*quat))

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

def joint_command_control(robot, joints, path, **kwargs):
    # /robot/right_gripper/joint_command
    publisher = Publisher("/lula/robot/limb/right/joint_command", JointPosVelAccCommand, queue_size=1)
    trajectory = spline_parameterization(robot, joints, path, **kwargs)
    print('Following {} waypoints in {:.3f} seconds'.format(
        len(path), trajectory.points[-1].time_from_start.to_sec()))
    wait_for_user('Execute?')

    dialation = 1.0
    for i in range(1, len(trajectory.points)):
        previous = trajectory.points[i-1]
        current = trajectory.points[i]
        duration = dialation*(current.time_from_start - previous.time_from_start)

        command = JointPosVelAccCommand()
        #command.header.frame_id = ISSAC_FRANKA_FRAME # left empty
        #command.header.stamp = rospy.Time(0)
        command.header.stamp = rospy.Time.now()
        command.names = get_joint_names(robot, joints)
        command.id = i
        command.period = duration
        command.t = command.header.stamp
        command.q = current.positions
        command.qd = current.velocities
        command.qdd = current.accelerations
        publisher.publish(command)
        print('Waypoint={} | Duration={:.3f}'.format(i, duration.to_sec()))
        rospy.sleep(duration)

def follow_control(robot, joints, path, **kwargs):
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
    action_topic = '/position_joint_trajectory_controller/follow_joint_trajectory'
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

    trajectory = spline_parameterization(robot, joints, path, **kwargs)
    print('Following {} waypoints in {:.3f} seconds'.format(
        len(path), trajectory.points[-1].time_from_start.to_sec()))
    # path_tolerance, goal_tolerance, goal_time_tolerance
    # http://docs.ros.org/diamondback/api/control_msgs/html/msg/FollowJointTrajectoryGoal.html
    #wait_for_user('Continue?')

    goal = FollowJointTrajectoryGoal(trajectory=trajectory)
    start_time = time.time()
    client.send_goal_and_wait(goal)  # send_goal_and_wait
    # client.get_result()
    print('Execution took {:.3f} seconds'.format(elapsed_time(start_time)))

def moveit_control(robot, joints, path, moveit, observer, **kwargs):
    #path = waypoints_from_path(path)
    #if moveit.use_lula:
    #    speed = 0.5*speed
    #joint_command_control(robot, joints, path, **kwargs)
    follow_control(robot, joints, path, **kwargs)
    return

    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/interpolator.py
    # Only position, time_from_start, and velocity are used
    trajectory = spline_parameterization(robot, joints, path, **kwargs)
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

################################################################################s

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

################################################################################

#FINGER_EFFORT_LIMIT = 20
#FINGER_VELOCITY_LIMIT = 0.2

def move_gripper(position, effort=20):
    # /franka_gripper/grasp
    action_topic = '/franka_gripper/gripper_action'
    client = SimpleActionClient(action_topic, GripperCommandAction)
    print('Starting', action_topic)
    client.wait_for_server()
    client.cancel_all_goals()
    print('Finished', action_topic)

    #goal = GripperCommandActionGoal()
    #goal.header.frame_id = ISSAC_FRANKA_FRAME
    #goal.header.stamp = rospy.Time(0)
    #goal.goal.command.position = position
    #goal.goal.command.max_effort = effort

    goal = GripperCommandGoal()
    print(dir(goal))
    goal.command.position = position
    goal.command.max_effort = effort
    client.send_goal_and_wait(goal) # send_goal_and_wait
    #client.get_result()

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
