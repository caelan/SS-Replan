import numpy as np

from issac import update_robot, ISSAC_REFERENCE_FRAME
from pybullet_tools.utils import get_distance_fn, get_joint_name, \
    get_max_velocity, get_max_force, get_difference_fn

def get_joint_names(body, joints):
    return [get_joint_name(body, joint).encode('ascii')  # ,'ignore')
            for joint in joints]

def joint_state_control(robot, joints, path, domain, moveit, observer,
                        threshold=0.01, timeout=2.0):
    # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/JointState.html
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py#L398
    from sensor_msgs.msg import JointState
    import rospy
    max_velocities = np.array([get_max_velocity(robot, joint) for joint in joints])
    max_forces = np.array([get_max_force(robot, joint) for joint in joints])
    joint_names = get_joint_names(robot, joints)
    distance_fn = get_distance_fn(robot, joints)
    difference_fn = get_difference_fn(robot, joints)
    for target_conf in path:
        rate = rospy.Rate(1000)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and ((rospy.Time.now() - start_time).to_sec() < timeout):
            world_state = observer.observe()
            robot_entity = world_state.entities[domain.robot]
            difference = difference_fn(target_conf, robot_entity.q)
            if distance_fn(target_conf, robot_entity.q) < threshold:
                break
            #velocity = None
            velocity = list(0.25*np.array(max_velocities))
            joint_state = JointState(name=joint_names, position=list(target_conf), velocity=velocity)
            # ee_frame = moveit.forward_kinematics(joint_state.position)
            # moveit.visualizer.send(ee_frame)
            moveit.joint_cmd_pub.publish(joint_state)
            rate.sleep()
        else:
            print('Failed to reach set point')
    # moveit.execute(plan, required_orig_err=0.05, timeout=20.0,
    #               publish_display_trajectory=True)
    # TODO: return status
    return None

def moveit_control(robot, joints, path):
    from trajectory_msgs.msg import JointTrajectoryPoint
    from moveit_msgs.msg import RobotTrajectory
    import rospy

    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/interpolator.py
    # Only position, time_from_start, and velocity are used
    plan = RobotTrajectory()
    plan.joint_trajectory.header.frame_id = ISSAC_REFERENCE_FRAME
    plan.joint_trajectory.header.stamp = rospy.Time(0)
    plan.joint_trajectory.joint_names = get_joint_names(robot, joints)

    speed = 0.1
    distance_fn = get_distance_fn(robot, joints)
    distances = [0] + [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    time_from_starts = np.cumsum(distances) / speed
    # print(time_from_starts)

    for i in range(1, len(path)):
        point = JointTrajectoryPoint()
        point.positions = list(path[i])
        # Don't need velocities, accelerations, or efforts
        vector = np.array(path[i]) - np.array(path[i-1])
        duration = (time_from_starts[i] - time_from_starts[i-1])
        point.velocities = list(vector / duration)
        point.accelerations = list(np.ones(len(joints)))
        point.effort = list(np.ones(len(joints)))
        point.time_from_start = rospy.Duration(time_from_starts[i])
        plan.joint_trajectory.points.append(point)

def lula_control(world, path, domain, observer, world_state):
    robot_entity = domain.get_robot()
    for name, entity in world_state.entities.items():
       if entity.controllable_object is not None:
           #entity.controllable_object.unsuppress()
           entity.controllable_object.suppress() # Propagate to parents?
       # entity.set_detached()
       #domain.attachments[actor] = goal
    franka = robot_entity.robot
    for i, positions in enumerate(path):
       print('{}/{}'.format(i, len(path)))
       timeout = 10.0 if i == len(positions)-1 else 2.0
       franka.end_effector.go_config(positions, err_thresh=0.05,
           wait_for_target=True, wait_time=timeout, verbose=True) # TODO: go_guided/go_long_range
       update_robot(world, domain, observer, observer.observe())
       #wait_for_duration(1e-3)
       # TODO: attachments

################################################################################s

FINGER_EFFORT_LIMIT = 20
FINGER_VELOCITY_LIMIT = 0.2

def open_gripper(moveit, effort=None, sleep=0.2):
    # robot_entity = domain.get_robot()
    # franka = robot_entity.robot
    # gripper = franka.end_effector.gripper
    # gripper.open(speed=.2, actuate_gripper=True, wait=True)
    # update_robot(self.world, domain, observer.observe())
    # time.sleep(1.0)
    #moveit.open_gripper(speed=0.1, sleep=0.2, wait=True)
    from sensor_msgs.msg import JointState
    import rospy
    joint_state = JointState(name=moveit.gripper.joints, position=moveit.gripper.open_positions)
    if effort is not None:
        effort = min(effort, FINGER_EFFORT_LIMIT)
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None

def close_gripper(moveit, effort=None, sleep=0.2):
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
    from sensor_msgs.msg import JointState
    import rospy
    joint_state = JointState(name=moveit.gripper.joints, position=moveit.gripper.closed_positions)
    if effort is not None:
        effort = min(effort, FINGER_EFFORT_LIMIT)
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None
