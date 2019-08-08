import math
import numpy as np
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from rospy import Publisher
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from pybullet_tools.utils import get_closest_angle_fn, INF, point_from_pose, wrap_angle, euler_from_quat, \
    quat_from_pose, get_angle, circular_difference, waypoints_from_path, get_link_pose, remove_handles, draw_pose
from src.issac import lookup_pose, ISSAC_PREFIX, ISSAC_CARTER_FRAME, ISSAC_WORLD_FRAME
from src.utils import WHEEL_JOINTS

def ROSPose(pose):
    point, quat = pose
    return Pose(position=Point(*point), orientation=Quaternion(*quat))

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
