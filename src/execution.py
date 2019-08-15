from __future__ import print_function

import time
import numpy as np
import rospy

from src.retime import spline_parameterization, get_joint_names, linear_parameterization
from src.issac import ISSAC_FRANKA_FRAME, update_observer
from pybullet_tools.utils import elapsed_time, wait_for_user
from pddlstream.utils import Verbose

from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, RobotTrajectory, RobotState
from actionlib import SimpleActionClient
#from actionlib_msgs.msg import GoalStatus, GoalState, SimpleClientGoalState
# http://docs.ros.org/jade/api/actionlib/html/classactionlib_1_1SimpleClientGoalState.html
# http://docs.ros.org/kinetic/api/actionlib_msgs/html/msg/GoalStatus.html
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, \
    GripperCommandAction, GripperCommandGoal, JointTolerance

#from lula_franka.franka import FrankaGripper
from lula_franka.franka_gripper_commander import FrankaGripperCommander

# control_msgs/GripperCommandAction
# control_msgs/JointTrajectoryAction
# control_msgs/SingleJointPositionAction
# franka_gripper/MoveAction
# moveit_msgs/ExecuteTrajectoryAction
# moveit_msgs/MoveGroupAction

def publish_display_trajectory(moveit, joint_trajectory, frame=ISSAC_FRANKA_FRAME):
    display_trajectory_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)

    display_trajectory = DisplayTrajectory()
    #display_trajectory.model_id = 'pr2'
    display_trajectory.trajectory_start = moveit.robot.get_current_state()
    robot_trajectory = RobotTrajectory(joint_trajectory=joint_trajectory)
    robot_trajectory.joint_trajectory.header.frame_id = frame
    display_trajectory.trajectory.append(robot_trajectory)
    display_trajectory_pub.publish(display_trajectory)
    # moveit.display_trajectory_publisher.publish(display_trajectory)

    robot_state_pub = rospy.Publisher('/display_robot_state', DisplayRobotState, queue_size=1)
    display_state = DisplayRobotState()
    display_state.state = display_trajectory.trajectory_start
    last_conf = joint_trajectory.points[-1].positions
    joint_state = display_state.state.joint_state
    joint_state.position = list(joint_state.position)
    for joint_name, position in zip(joint_trajectory.joint_names, last_conf):
        joint_index = joint_state.name.index(joint_name)
        joint_state.position[joint_index] = position
    robot_state_pub.publish(display_state)

    return display_trajectory

################################################################################

def franka_control(robot, joints, path, interface, **kwargs):

    #joint_command_control(robot, joints, path, **kwargs)
    #follow_control(robot, joints, path, **kwargs)
    if interface.simulation:
        return moveit_control(robot, joints, path, interface)
        #return joint_state_control(robot, joints, path, interface)

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

    #error_threshold = 1e-3
    #threshold_template = '/position_joint_trajectory_controller/constraints/{}/goal'
    #for name in get_joint_names(robot, joints):
    #    param = threshold_template.format(name)
    #    rospy.set_param(param, error_threshold)
    #    #print(name, rospy.get_param(param))

    trajectory = spline_parameterization(robot, joints, path, **kwargs)
    print('Following {} waypoints in {:.3f} seconds'.format(
        len(path), trajectory.points[-1].time_from_start.to_sec()))
    # path_tolerance, goal_tolerance, goal_time_tolerance
    # http://docs.ros.org/diamondback/api/control_msgs/html/msg/FollowJointTrajectoryGoal.html
    publish_display_trajectory(interface.moveit, trajectory)
    wait_for_user('Continue?')

    goal = FollowJointTrajectoryGoal(trajectory=trajectory)
    #goal.goal_time_tolerance =
    for joint in trajectory.joint_names:
        #goal.path_tolerance.append(JointTolerance(name=joint, position=1e-2)) # position | velocity | acceleration
        goal.goal_tolerance.append(JointTolerance(name=joint, position=1e-3)) # position | velocity | acceleration

    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
    start_time = time.time()
    state = client.send_goal_and_wait(goal)  # send_goal_and_wait
    print('State:', state)
    state = client.get_state()
    print('State:', state)
    print('Status:', client.get_goal_status_text())
    result = client.get_result()
    print('Result:', result)
    # https://docs.ros.org/diamondback/api/actionlib/html/simple__action__client_8py_source.html

    print('Execution took {:.3f} seconds'.format(elapsed_time(start_time)))
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))

################################################################################

def suppress_lula(world_state):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L138
    for name, entity in world_state.entities.items():
       if entity.controllable_object is not None:
           #entity.controllable_object.unsuppress()
           entity.controllable_object.suppress() # Propagate to parents?
       # entity.set_detached()
       #domain.attachments[actor] = goal

def moveit_control(robot, joints, path, interface, **kwargs):
    #path = waypoints_from_path(path)
    #if moveit.use_lula:
    #    speed = 0.5*speed

    # Moveit's trajectories
    # rostopic echo /execute_trajectory/goal
    # About 1 waypoint per second
    # Start and end velocities are zero
    # Accelerations are about zero (except at the start and end)

    # TODO: the moveit trajectory execution is wobbly. Was this always the case?
    moveit = interface.moveit
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/interpolator.py
    # Only position, time_from_start, and velocity are used
    # trajectory = linear_parameterization(robot, joints, path, speed=0.025*np.pi)
    trajectory = spline_parameterization(robot, joints, path, speed=0.05 * np.pi, **kwargs)
    print('Following {} waypoints in {:.3f} seconds'.format(
        len(path), trajectory.points[-1].time_from_start.to_sec()))
    publish_display_trajectory(interface.moveit, trajectory)
    wait_for_user('Continue?')
    # /move_base_simple/goal - no listeners
    # /move_group/goal
    # ABORTED: Cannot execute trajectory since ~allow_trajectory_execution was set to false

    plan = RobotTrajectory(joint_trajectory=trajectory)
    if moveit.use_lula:
        world_state = update_observer(interface.observer)
        suppress_lula(world_state)
    # moveit_msgs/ExecuteTrajectoryActionGoal
    # moveit.group.execute(plan, wait=True)
    # https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    #return

    moveit.verbose = False
    moveit.last_ik = plan.joint_trajectory.points[-1].positions
    moveit.dilation = 0.5
    start_time = time.time()
    # /move_group/display_planned_path
    # TODO: display base motions?
    #publish_display_trajectory(moveit, plan) # TODO: get this in the base_link frame
    with Verbose(False):
        moveit.execute(plan, required_orig_err=0.005, timeout=5.0,
                       publish_display_trajectory=False)
        #moveit.go_local(q=path[-1], ...)
    print('Execution took {:.3f} seconds'.format(elapsed_time(start_time)))

################################################################################

def franka_move_gripper(position, effort=20):
    #gripper = FrankaGripper(is_physical_robot=True)
    #gripper.commander.move(position, speed=.03, wait=True)

    # Should be the same as the following:
    # /home/cpaxton/srl_system/packages/external/lula_franka/lula_franka/franka_gripper_commander.py
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
    goal.command.position = position
    goal.command.max_effort = effort
    client.send_goal(goal) # send_goal | send_goal_and_wait
    client.wait_for_result()
    #client.get_result()
    #rospy.sleep(1.0)

def franka_open_gripper(interface, **kwargs):
    if interface.simulation:
        # https://gitlab-master.nvidia.com/srl/srl_system/compare/9d2455ccd4c97b49f85f86cb2af80fc34e0b66cc...master
        interface.moveit.gripper_cmd_pub = interface.moveit.goal_joint_cmd_pub
        interface.moveit.open_gripper()
    else:
        commander = FrankaGripperCommander()
        commander.open(speed=0.1, wait=True)
        # gripper = FrankaGripper(is_physical_robot=True)
        # return gripper.open(speed=0.1, wait=True)
        # return move_gripper_action(moveit.gripper.open_positions[0], **kwargs)

def franka_close_gripper(interface):
    if interface.simulation:
        interface.moveit.gripper_cmd_pub = interface.moveit.goal_joint_cmd_pub
        interface.moveit.close_gripper()
    else:
        commander = FrankaGripperCommander()
        commander.close(speed=0.1, force=60, wait=True)
        #gripper = FrankaGripper(is_physical_robot=True)
        #return gripper.close(attach_obj=None, speed=0.1, force=40, actuate_gripper=True, wait=True)
        #return move_gripper_action(moveit.gripper.closed_positions[0], effort=50)
