import rospy
from lula_controller_msgs.msg import JointPosVelAccCommand
from pybullet_tools.utils import wait_for_user, joint_from_name, get_max_force
from rospy import Publisher
from sensor_msgs.msg import JointState
from src.isaac.execution import suppress_lula
from src.isaac.issac import update_robot
from src.retime import get_joint_names, spline_parameterization

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
