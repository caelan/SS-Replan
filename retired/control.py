import rospy
from lula_controller_msgs.msg import JointPosVelAccCommand
from pddlstream.utils import Verbose
from pybullet_tools.utils import get_distance_fn, wait_for_user, joint_from_name, get_max_force
from rospy import Publisher
from sensor_msgs.msg import JointState
from src.execution import suppress_lula
from src.issac import update_robot, update_observer
from src.retime import get_joint_names, spline_parameterization

################################################################################

def joint_state_control(robot, joints, path, interface,
                        threshold=0.01, timeout=1.0):
    # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/JointState.html
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py#L398
    #max_velocities = np.array([get_max_velocity(robot, joint) for joint in joints])
    #max_forces = np.array([get_max_force(robot, joint) for joint in joints])
    joint_names = get_joint_names(robot, joints)
    distance_fn = get_distance_fn(robot, joints)
    #difference_fn = get_difference_fn(robot, joints)

    # TODO: separate path and goal thresholds
    assert interface.simulation
    # TODO: spline interpolation
    #path = waypoints_from_path(path)
    if len(joints) == 2:
        path = path[-1:]
    for i, target_conf in enumerate(path):
        print('Waypoint {} / {}'.format(i, len(path)))
        velocity = None
        #velocity = list(0.25 * np.array(max_velocities))
        joint_state = JointState(name=joint_names, position=list(target_conf), velocity=velocity)
        interface.moveit.joint_cmd_pub.publish(joint_state)
        #rate = rospy.Rate(1000)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and ((rospy.Time.now() - start_time).to_sec() < timeout):
            with Verbose():
                world_state = update_observer(interface.observer)
            robot_entity = world_state.entities[interface.domain.robot]
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

################################################################################

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

################################################################################

def lula_control(world, path, domain, observer, world_state):
    suppress_lula(world_state)
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

def move_gripper(robot, moveit, position, effort=20, sleep=1.0):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L155
    # robot_entity = domain.get_robot()
    # franka = robot_entity.robot
    # gripper = franka.end_effector.gripper
    # gripper.open(speed=.2, actuate_gripper=True, wait=True)
    # update_robot(self.world, domain, observer.observe())
    # time.sleep(1.0)
    #moveit.open_gripper(speed=0.1, sleep=0.2, wait=True)
    joint_state = JointState(name=moveit.gripper.joints, position=position)
    if effort is not None:
        gripper_joint = joint_from_name(robot, moveit.gripper.joints[0])
        max_effort = get_max_force(robot, gripper_joint)
        effort = max(0, min(effort, max_effort))
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None

def open_gripper(robot, moveit, **kwargs):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L155
    # gripper.open(speed=.2, actuate_gripper=True, wait=True)
    # robot_entity = domain.get_robot()
    # franka = robot_entity.robot
    # gripper = franka.end_effector.gripper
    # TODO: only sleep is used by close_gripper and open_gripper...
    #moveit.open_gripper(speed=0.1, sleep=0.2, wait=True)
    return move_gripper(robot, moveit, moveit.gripper.open_positions, **kwargs)

def close_gripper(robot, moveit, **kwargs):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L218
    # gripper.close(attach_obj=None, speed=.2, force=40., actuate_gripper=True, wait=True)
    #moveit.close_gripper(controllable_object=None, speed=0.1, force=40., sleep=0.2, wait=True)
    return move_gripper(robot, moveit, moveit.gripper.closed_positions, **kwargs)
