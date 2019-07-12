import numpy as np

from src.issac import update_robot, ISSAC_REFERENCE_FRAME, lookup_pose, \
    ISSAC_PREFIX, ISSAC_CARTER_FRAME, CONTROL_TOPIC
from pybullet_tools.utils import get_distance_fn, get_joint_name, \
    get_max_force, joint_from_name, point_from_pose, wrap_angle, \
    euler_from_quat, quat_from_pose, dump_body, circular_difference, \
    joints_from_names, get_max_velocity, get_distance, get_angle, INF, waypoints_from_path, HideOutput
from src.utils import WHEEL_JOINTS
from pddlstream.utils import Verbose

def get_joint_names(body, joints):
    return [get_joint_name(body, joint).encode('ascii')  # ,'ignore')
            for joint in joints]

def joint_state_control(robot, joints, path, domain, moveit, observer,
                        threshold=0.01, timeout=1.0):
    # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/JointState.html
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py#L398
    from sensor_msgs.msg import JointState
    import rospy
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
                world_state = observer.observe()
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

def control_base(goal_values, moveit, observer, timeout=INF):
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    import rospy
    #joints = joints_from_names(world.robot, WHEEL_JOINTS)
    #max_velocities = np.array([get_max_velocity(world.robot, joint) for joint in joints])

    unit_forward = np.array([-1, -1]) # Negative seems to be forward
    unit_right = np.array([-1, +1])

    min_speed = 12 # Technically 9.085
    max_speed = 30

    ramp_down_pos = 0.25
    ramp_down_yaw = np.pi / 8

    #dump_body(world.robot)
    # https://github.mit.edu/caelan/base-trajectory-action/blob/master/src/base_trajectory.cpp
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/scripts/set_base_joint_states.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter.py
    linear_threshold = 0.04
    angular_threshold = np.pi / 64

    # TODO: open-loop odometry (use wheel encoders)
    # TODO: ensure that we are rotating about the correct axis (i.e. base_joints aligned with axis)

    # linear/angular
    reached_goal_pos = False
    goal_pos = np.array(goal_values[:2])
    goal_yaw = goal_values[2]

    #pub = moveit.joint_cmd_pub
    pub = rospy.Publisher(CONTROL_TOPIC, JointState, queue_size=1)
    try:
        rate = rospy.Rate(100)
        start_time = rospy.Time.now()
        while (not rospy.is_shutdown()) and ((rospy.Time.now() - start_time).to_sec() < timeout):
            #world_state = observer.observe()
            #robot_entity = world_state.entities[domain.robot]
            #print(robot_entity.carter_pos, robot_entity.carter_vel)
            #robot_entity.base_link
            #print(pose_from_tform(robot_entity.pose))

            base_pose = lookup_pose(observer.tf_listener, ISSAC_PREFIX + ISSAC_CARTER_FRAME)
            pos = np.array(point_from_pose(base_pose)[:2])
            _, _, yaw = map(wrap_angle, euler_from_quat(quat_from_pose(base_pose)))

            x, y, _ = point_from_pose(base_pose)
            print('x={:.3f}, y={:.3f}, yaw={:.3f}'.format(x, y, yaw))

            movement_yaw = get_angle(pos, goal_pos)
            movement_yaw_error = abs(circular_difference(movement_yaw, yaw))
            delta_pos = goal_pos - pos
            pos_error = np.linalg.norm(delta_pos)
            goal_yaw_error = abs(circular_difference(goal_yaw, yaw))
            print('Linear error: {:.5f} ({:.5f})'.format(pos_error, linear_threshold))
            print('Angular error: {:.5f} ({:.5f})'.format(goal_yaw_error, angular_threshold))

            target_yaw = None
            if reached_goal_pos or (pos_error < linear_threshold):
                reached_goal_pos = True
                print('rotating towards goal yaw')
                if goal_yaw_error < angular_threshold:
                    return True
                target_yaw = goal_yaw
            else:
                if movement_yaw_error < angular_threshold:
                    print('moving towards goal position')
                else:
                    print('moving towards movement yaw')
                    target_yaw = movement_yaw

            if target_yaw is None:
                # target_pos = goal_pos
                print('Linear delta:', delta_pos)
                pos_fraction = min(1, pos_error / ramp_down_pos)
                speed = (1 - pos_fraction) * min_speed + pos_fraction * max_speed
                joint_velocities = speed * unit_forward
            else:
                delta_yaw = circular_difference(target_yaw, yaw)
                print('Angular delta:', delta_yaw)
                #print(robot_entity.carter_interface) # None
                #print(robot_entity.joints) # Only arm joints
                #print(robot_entity.q, robot_entity.dq)
                #speed = min(max_speed, 60*np.abs(delta_yaw) / np.pi)
                yaw_fraction = min(1., abs(delta_yaw) / ramp_down_yaw)
                speed = (1 - yaw_fraction)*min_speed + yaw_fraction*max_speed
                #speed = max_speed
                joint_velocities = - np.sign(delta_yaw) * speed * unit_right

            print('Velocities:', joint_velocities.round(3).tolist())
            #world_state = observer.observe()
            #update_robot(world, domain, observer, world_state)
            pub.publish(JointState(header=Header(), name=WHEEL_JOINTS, velocity=list(joint_velocities)))
            #rate.sleep()
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
    finally:
        joint_velocities = np.zeros(len(WHEEL_JOINTS))
        print('Velocities:', joint_velocities.round(3).tolist())
        pub.publish(JointState(header=Header(), name=WHEEL_JOINTS, velocity=list(joint_velocities)))
    return False

################################################################################s

#FINGER_EFFORT_LIMIT = 20
#FINGER_VELOCITY_LIMIT = 0.2

def open_gripper(robot, moveit, effort=20, sleep=1.0):
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
        gripper_joint = joint_from_name(robot, moveit.gripper.joints[0])
        max_effort = get_max_force(robot, gripper_joint)
        effort = max(0, min(effort, max_effort))
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None

def close_gripper(robot, moveit, effort=20, sleep=1.0):
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
        gripper_joint = joint_from_name(robot, moveit.gripper.joints[0])
        max_effort = get_max_force(robot, gripper_joint)
        effort = max(0, min(effort, max_effort))
        joint_state.effort = [effort] * len(moveit.gripper.joints)
    moveit.joint_cmd_pub.publish(joint_state)
    if 0. < sleep:
        rospy.sleep(sleep)
    return None
