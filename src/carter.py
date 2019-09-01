from __future__ import print_function

import numpy as np
import time
import math

import rospy
from pybullet_tools.utils import elapsed_time, pose2d_from_pose, get_joint_positions, \
    multiply, invert, unit_from_theta, pose_from_pose2d, quat_angle_between

from src.issac import get_base_pose
from src.deepim import mean_pose_deviation, wait_until_frames_stabilize



# Middle carter pose: [ 25.928  11.541   2.4  ]
# BB8 carter pose: [ 17.039  13.291  -2.721]

HOME_BASE_POSE = [31.797, 9.118, -0.12]
#INDIGO_BASE_POSE = [33.1, 7.789, 0.0]
INDIGO_BASE_POSE = [33.05, 7.789, 0.0]

CARRY_CONF = [0.020760029206411876, -1.0611899273529857, -0.052402929133539944, -2.567198461037754,
              -0.06013280179334339, 1.5917587080266737, -2.3553707114303295]

def seed_dart_with_carter(interface):
    robot_entity = interface.domain.get_robot()
    #robot_entity.unfix_bases() # suppressor.deactivate() => unfix
    start_time = time.time()
    timeout = 5
    while elapsed_time(start_time) < timeout:
        interface.carter.pub_disable_deadman_switch.publish(True) # must send repeatedly
        rospy.sleep(0.01)
    interface.carter.pub_disable_deadman_switch.publish(False)
    #robot_entity.fix_bases() # suppressor.activate() => fix

# navigation.control.lqr
ISAAC_SIGHT = {
    "gain_speed": 10.0,
    "min_distance": 0.0,
    "num_controls_to_check": 25,
    "speed_gradient_min_distance": 0.1,
    "target_distance": 0.0, # 0.01
}

def command_carter(interface, goal_pose, timeout=30):
    # pose_deadman_topic = '/isaac/disable_deadman_switch'
    # velocity_deadman_topic = '/isaac/enable_ros_segway_cmd'
    # carter.move_to(goal_pose) # recursion bug
    robot_entity = interface.domain.get_robot()
    #robot_entity.unfix_bases()  # suppressor.deactivate() => unfix
    start_time = time.time()
    reached_goal = False
    carter = interface.carter
    carter.current_goal = goal_pose

    history = []
    print('Initial pose:', np.array(carter.current_pose).round(3))
    print('Goal pose:', np.array(goal_pose).round(3))
    while elapsed_time(start_time) < timeout:
        carter.update_current_error()
        print('Error:', np.array(carter.pose_error).round(3))
        history.append(np.array(carter.current_pose))
        if carter.at_goal():
            print('Within goal region!')
            # TODO: wrap angle
            if np.std([pose[2] for pose in history[-10:]]) < math.radians(2):
                break
        #else:
        #    history = []
        carter.pub_disable_deadman_switch.publish(True)  # must send repeatedly
        carter.move_to_async(goal_pose)  # move_to_async | move_to_safe
        rospy.sleep(0.01)
    else:
        reached_goal = False

    start_time = time.time()
    rest_pose = np.array(carter.current_pose)
    print('Stopping carter')
    total = np.zeros(rest_pose.shape)
    num = 0
    while elapsed_time(start_time) < 10.0:
        total += carter.current_pose
        num += 1
        average = total / num
        print('Running average:', average.round(3))
        rest_pose = average
        #rest_pose = carter.current_pose
        carter.move_to_async(rest_pose)  # move_to_async | move_to_safe
        carter.pub_disable_deadman_switch.publish(True)
        rospy.sleep(0.01)

    success = wait_until_frames_stabilize(interface, frames=[robot_entity.current_root])

    #carter.pub_disable_deadman_switch.publish(False)
    #robot_entity.fix_bases()  # suppressor.activate() => fix
    # Towards the kitchen is +x (yaw=0)
    # fix base of Panda with DART is overwritten by the published message
    return reached_goal


def command_carter_to_pybullet_goal(interface, goal_pose2d, **kwargs):
    world = interface.world
    pybullet_from_current = pose_from_pose2d(get_joint_positions(world.robot, world.base_joints))
    isaac_from_current = pose_from_pose2d(interface.carter.current_pose)
    isaac_from_pybullet = multiply(isaac_from_current, invert(pybullet_from_current))
    pybullet_from_goal = pose_from_pose2d(goal_pose2d)
    isaac_from_goal = multiply(isaac_from_pybullet, pybullet_from_goal)
    isaac_from_goal2d = pose2d_from_pose(isaac_from_goal)
    #distance_fn = get_nonholonomic_distance_fn(world.robot, world.base_joints,
    #                                           linear_velocity=0.25, angular_velocity=np.pi/2)
    #duration = distance_fn(pybullet_from_current, pybullet_from_goal)
    #timeout = 2*duration
    return command_carter(interface, isaac_from_goal2d, **kwargs) # TODO: customize based on the straight-line distance


def heading_waypoint(carter, distance):
    x, y, theta = carter.current_pose # current_velocity
    pos = np.array([x, y])
    goal_pos = pos + distance * unit_from_theta(theta)
    goal_pose = np.append(goal_pos, [theta])
    return goal_pose

################################################################################

def test_carter(interface):
    carter = interface.carter
    # /isaac_navigation2D_status
    # /isaac_navigation2D_request

    assert carter is not None
    initial_pose = carter.current_pose
    print('Carter pose:', initial_pose)
    x, y, theta = initial_pose  # current_velocity
    #goal_pose = np.array([x, y, 0])
    #goal_pose = heading_stuff(carter, -1.0)
    #goal_pose = np.array(INDIGO_BASE_POSE)
    goal_pose = np.array(HOME_BASE_POSE)
    command_carter(interface, goal_pose, timeout=120)

    #while True:
    #    print('Carter pose:', carter.current_pose)
    #    rospy.sleep(0.1)

    #carter.move_to_openloop(goal_pose)
    # move_to_open_loop | move_to_safe_followed_by_openloop

    #carter.simple_move(-0.1) # simple_move | simple_stop
    # rospy.sleep(2.0)
    # carter.simple_stop()
    #domain.get_robot().carter_interface = interface.carter
    # domain.get_robot().unsuppress_fixed_bases()

    # /sim/tf to get all objects
    # https://gitlab-master.nvidia.com/srl/srl_system/blob/722d127a016c9105ec68a33902a73480c36b31ac/packages/isaac_bridge/scripts/sim_tf_relay.py
    # sim_tf_relay.py

    # roslaunch isaac_bridge sim_franka.launch cooked_sim:=true config:=panda_full lula:=false world:=franka_leftright_kitchen_ycb_world.yaml
    # https://gitlab-master.nvidia.com/srl/srl_system/blob/fb94253c60b1bd1308a37c1aeb9dc4a4c453c512/packages/isaac_bridge/launch/sim_franka.launch
    # packages/external/lula_franka/config/worlds/franka_center_right_kitchen.sim.yaml
    # packages/external/lula_franka/config/worlds/franka_center_right_kitchen.yaml
