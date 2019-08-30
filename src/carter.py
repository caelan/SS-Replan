import numpy as np
import time

import rospy
from pybullet_tools.utils import elapsed_time, pose2d_from_pose, get_joint_positions, multiply, invert, \
    unit_from_theta, get_nonholonomic_distance_fn



# Middle carter pose: [ 25.928  11.541   2.4  ]
# BB8 carter pose: [ 17.039  13.291  -2.721]

HOME_BASE_POSE = [31.797, 9.118, -0.12]
#INDIGO_BASE_POSE = [33.1, 7.789, 0.0]
INDIGO_BASE_POSE = [33.05, 7.789, 0.0]

CARRY_CONF = [0.020760029206411876, -1.0611899273529857, -0.052402929133539944, -2.567198461037754,
              -0.06013280179334339, 1.5917587080266737, -2.3553707114303295]

def seed_dart_with_carter(interface):
    robot_entity = interface.domain.get_robot()
    robot_entity.unfix_bases() # suppressor.deactivate() => unfix
    start_time = time.time()
    timeout = 5
    while elapsed_time(start_time) < timeout:
        interface.carter.pub_disable_deadman_switch.publish(True) # must send repeatedly
        rospy.sleep(0.01)
    interface.carter.pub_disable_deadman_switch.publish(False)
    robot_entity.fix_bases() # suppressor.activate() => fix


def command_carter(interface, goal_pose, timeout=30):
    # pose_deadman_topic = '/isaac/disable_deadman_switch'
    # velocity_deadman_topic = '/isaac/enable_ros_segway_cmd'
    # carter.move_to(goal_pose) # recursion bug
    robot_entity = interface.domain.get_robot()
    robot_entity.unfix_bases()  # suppressor.deactivate() => unfix
    start_time = time.time()
    reached_goal = False
    carter = interface.carter
    while elapsed_time(start_time) < timeout:
        print('Error:', np.array(carter.pose_error).round(3))
        if carter.at_goal():
            print('At goal configuration!')
            reached_goal = True
            break
        carter.pub_disable_deadman_switch.publish(True)  # must send repeatedly
        carter.move_to_async(goal_pose)  # move_to_async | move_to_safe
        rospy.sleep(0.01)

    start_time = time.time()
    while elapsed_time(start_time) < 1.0:
        carter.move_to_async(carter.current_pose)  # move_to_async | move_to_safe
        carter.pub_disable_deadman_switch.publish(False)
        rospy.sleep(0.01)
    # TODO: wait until it doesn't move for a certain amount of time

    carter.pub_disable_deadman_switch.publish(False)
    robot_entity.fix_bases()  # suppressor.activate() => fix
    # Towards the kitchen is +x (yaw=0)
    # fix base of Panda with DART is overwritten by the published message
    return reached_goal


def command_carter_to_pybullet_goal(interface, goal_pose2d, **kwargs):
    world = interface.world
    pybullet_from_current = pose2d_from_pose(get_joint_positions(world.robot, world.base_joints))
    isaac_from_current = pose2d_from_pose(interface.carter.current_pose)
    isaac_from_pybullet = multiply(isaac_from_current, invert(pybullet_from_current))
    pybullet_from_goal = pose2d_from_pose(goal_pose2d)
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
    goal_pose = np.array(INDIGO_BASE_POSE)
    #goal_pose = np.array(HOME_BASE_POSE)
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
