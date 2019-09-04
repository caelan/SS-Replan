#!/usr/bin/env python2

from __future__ import print_function

import sys
import os
import rospy
import traceback
import numpy as np
import math
import time

from collections import defaultdict

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

#import brain_ros.moveit
#brain_ros.moveit.USE_MOVEIT = False

from brain_ros.kitchen_domain import KitchenDomain
#from brain_ros.demo_kitchen_domain import KitchenDomain as DemoKitchenDomain
#from grasps import *
from brain_ros.ros_world_state import RosObserver
from isaac_bridge.carter import Carter
from src.carter import command_carter, HOME_BASE_POSE, test_carter

#import kitchen_poses
#kitchen_poses.supported_ycb_objects[:] = []

from pybullet_tools.utils import LockRenderer, wait_for_user, elapsed_time, \
    point_from_pose, set_camera_pose, \
    link_from_name, get_link_pose, get_joint_positions
from pddlstream.utils import Verbose

from src.deepim import DeepIM, mean_pose_deviation, get_pose_distance
#from retired.perception import test_deepim
from src.policy import run_policy
from src.interface import Interface
from src.command import execute_commands, iterate_commands
from src.isaac_task import TRIAL_MANAGER_TASKS, set_isaac_sim, \
    simulation_setup
from src.utils import JOINT_TEMPLATE, SPAM, SUGAR, CHEEZIT, YCB_OBJECTS, INDIGO_COUNTER, \
    TOP_DRAWER, TOP_GRASP, LEFT_DOOR, BOTTOM_DRAWER, SIDE_GRASP
from src.visualization import add_markers
from src.issac import observe_world, kill_lula, update_robot_conf, \
    load_objects, display_kinect, update_objects, RIGHT, LEFT, get_base_pose
from src.world import World
from src.task import Task
from src.execution import franka_open_gripper
from run_pybullet import create_parser

from examples.discrete_belief.dist import UniformDist


# TODO: prevent grasp if open
# TODO: ignore grasped objects in the update
# TODO: avoid redetects on hard surfaces

def wait_for_dart_convergence(interface, detected, min_updates=10, timeout=10.0):
    # TODO: move to a dart class
    start_time = time.time()
    history = defaultdict(list)
    num_updates = 0
    while elapsed_time(start_time) < timeout:
        #rospy.sleep(1e-2)
        num_updates += 1
        print('Update: {} | Time: {}'.format(num_updates, elapsed_time(start_time)))
        world_state = interface.update_state()
        observation = update_objects(interface, world_state, detected)
        # TODO: could include robot pose here
        # print(observation)
        for name, pose in observation.items():
            history[name].extend(pose)
        if num_updates < min_updates:
            continue
        success = True
        for name in history:
            if min_updates <= len(history[name]):
                pos_deviation, ori_deviation = mean_pose_deviation(history[name][-min_updates:])
                print('{}) position deviation: {:.3f} meters | orientation deviation: {:.3f} degrees'.format(
                    name, pos_deviation, math.degrees(ori_deviation)))
                success &= (pos_deviation <= 0.005) and (ori_deviation <= math.radians(1))
        if success:
            return True
    # TODO: return last observation?
    return False

def planning_loop(interface):
    args = interface.args
    initial_base_pose = get_base_pose(interface.observer)
    # TODO: wait until the robot's base orientation is correct

    def observation_fn(belief):
        # TODO: test if visibility is good enough
        # TODO: sort by distance from camera
        assert interface.deepim is not None # TODO: IsaacSim analog
        if belief.holding is not None:
            interface.stop_tracking(belief.holding)
        detected = interface.perception.detect_all(belief.placed)
        start_time = time.time()
        converged = wait_for_dart_convergence(interface, detected)
        print('Stabilized: {} | Time: {:.3f}'.format(converged, elapsed_time(start_time)))
        global initial_base_pose
        initial_base_pose = get_base_pose(interface.observer)
        #rospy.sleep(5.0)
        # Wait until convergence
        #interface.localize_all()
        return observe_world(interface, visible=detected)

    def transition_fn(belief, commands):
        sim_state = belief.sample_state()
        if args.watch or args.record:
            wait_for_user()
            # simulate_plan(sim_state.copy(), commands, args)
            iterate_commands(sim_state.copy(), commands)
            wait_for_user()
        sim_state.assign()
        if args.teleport or args.cfree:
            print('Some constraints were ignored. Skipping execution!')
            return False
        current_base_pose = get_base_pose(interface.observer)
        global initial_base_pose
        translation, rotation = get_pose_distance(initial_base_pose, current_base_pose)
        #if (0.01 < translation) or ():
        #    return False


        # TODO: could calibrate closed-loop relative to the object
        # Terminate if failed to pick up
        success = execute_commands(interface, commands)
        update_robot_conf(interface)
        return success

    return run_policy(interface.task, args, observation_fn, transition_fn)

################################################################################

#TARGET_DISTANCE = 0.25 # match up with Isaac Sight
TARGET_DISTANCE = 0.05

def real_setup(domain, world, args):
    # TODO: detect if lula is active via rosparam
    observer = RosObserver(domain)
    perception = DeepIM(domain, sides=[RIGHT], obj_types=YCB_OBJECTS)
    prior = {
        SPAM: UniformDist([TOP_DRAWER, BOTTOM_DRAWER]), # INDIGO_COUNTER
        #SPAM: UniformDist([INDIGO_COUNTER]),  # INDIGO_COUNTER
        SUGAR: UniformDist([INDIGO_COUNTER]),
        #CHEEZIT: UniformDist([INDIGO_COUNTER]),
    }
    goal_drawer = TOP_DRAWER # TOP_DRAWER | BOTTOM_DRAWER | LEFT_DOOR
    task = Task(world, prior=prior, teleport_base=True,
                grasp_types=[TOP_GRASP, SIDE_GRASP],
                #goal_detected=[SPAM],
                goal_holding=SPAM,
                #goal_on={SPAM: goal_drawer},
                #goal_closed=[],
                #goal_closed=[JOINT_TEMPLATE.format(goal_drawer)],
                #goal_closed=[JOINT_TEMPLATE.format(drawer) for drawer in [TOP_DRAWER, BOTTOM_DRAWER]],
                #goal_open=[JOINT_TEMPLATE.format(goal_drawer)],
                movable_base=not args.fixed,
                goal_aq=world.carry_conf, #.values,
                #goal_cooked=[SPAM],
                #return_init_aq=True,
                return_init_bq=True)

    robot_entity = domain.get_robot()
    if not args.fixed:
        # TODO: these thresholds not used by Isaac
        carter = Carter(goal_threshold_tra=TARGET_DISTANCE,
                        goal_threshold_rot=math.radians(15.),
                        vel_threshold_lin=0.01,
                        vel_threshold_ang=math.radians(1.0))
        robot_entity.carter_interface = carter
    interface = Interface(args, task, observer, deepim=perception)
    if interface.carter is not None:
        #initial_pose = interface.carter.current_pose
        initial_pose = HOME_BASE_POSE # INDIGO_BASE_POSE | HOME_BASE_POSE
        goal_distance = np.linalg.norm(np.array(HOME_BASE_POSE)[:2] - np.array(interface.carter.current_pose)[:2])
        if 0.5 < goal_distance:
            command_carter(interface, initial_pose)
        # Carter more likely to creep forward when not near cabinet

    #robot_entity.fix_bases()
    robot_entity.unfix_bases()
    return interface


################################################################################

#   File "/home/cpaxton/srl_system/workspace/src/brain/src/brain_ros/ros_world_state.py", line 397, in update_msg
#     self.gripper = msg.get_positions([self.gripper_joint])[0]
# TypeError: 'NoneType' object has no attribute '__getitem__'

def main():
    parser = create_parser()
    parser.add_argument('-execute', action='store_true',
                        help="When enabled, uses the real robot_entity")
    parser.add_argument('-jump', action='store_true',
                        help="When enabled, skips base control")
    parser.add_argument('-lula', action='store_true',
                        help='When enabled, uses LULA instead of JointState control')
    parser.add_argument('-problem', default=TRIAL_MANAGER_TASKS[2], #choices=TRIAL_MANAGER_TASKS,
                        help='The name of the task')
    parser.add_argument('-watch', action='store_true',
                        help='When enabled, plans are visualized in PyBullet before executing in IsaacSim')
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    #args.watch |= args.execute
    # TODO: reobserve thee same scene until receive good observation

    # srl_system/packages/isaac_bridge/configs/ycb_table_config.json
    # srl_system/packages/isaac_bridge/configs/ycb_table_graph.json
    # srl_system/packages/isaac_bridge/configs/panda_full_config.json
    # srl_system/packages/isaac_bridge/configs/panda_full_graph.json
    # alice/assets/maps/seattle_map_res02_181214.config.json

    # https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/brain/src/brain_ros/lula_policies.py#L464
    rospy.init_node("STRIPStream")

    with Verbose(False):
        domain = KitchenDomain(sim=not args.execute, sigma=0, lula=args.lula)
        #domain = DemoKitchenDomain(sim=not args.execute, use_carter=True) # TODO: broken
    robot_entity = domain.get_robot()
    #robot_entity.get_motion_interface().remove_obstacle() # TODO: doesn't remove
    #robot_entity.fix_bases()
    #robot_entity.unfix_bases()
    #print(dump_dict(robot_entity))
    #test_deepim(domain)

    # /home/cpaxton/srl_system/workspace/src/external/lula_franka
    world = World(use_gui=True) # args.visualize)
    if args.fixed:
        target_point = point_from_pose(get_link_pose(
            world.kitchen, link_from_name(world.kitchen, 'indigo_tmp')))
        offset = np.array([1, -1, 1])
        camera_point = target_point + offset
        set_camera_pose(camera_point, target_point)
    if args.execute:
        interface = real_setup(domain, world, args)
    else:
        interface = simulation_setup(domain, world, args)
    #seed_dart_with_carter(interface)
    franka_open_gripper(interface)
    #interface.localize_all()
    #interface.update_state()
    #test_carter(interface)
    #return

    # Can disable lula world objects to improve speed
    # Adjust DART to get a better estimate for the drawer joints
    #interface.localize_all()
    #wait_for_user()
    #print('Entities:', sorted(world_state.entities))
    with LockRenderer(lock=True):
        # Used to need to do expensive computation before localize_all
        # due to the LULA overhead (e.g. loading complex meshes)
        load_objects(interface.task)
        observe_world(interface, visible=set())
        print('Base conf:', get_joint_positions(world.robot, world.base_joints))
        print('Arm conf:', get_joint_positions(world.robot, world.arm_joints))
        print('Gripper conf:', get_joint_positions(world.robot, world.gripper_joints))
        for side in [LEFT] if interface.simulation else [RIGHT]: # TODO: simulation naming inconsistency
            display_kinect(interface, side=side)
        #if LEFT in world.cameras:
        #    del world.cameras[LEFT]
        if interface.simulation:  # TODO: move to simulation instead?
            set_isaac_sim(interface)
        world._update_initial()
        add_markers(interface.task, inverse_place=False)
    #wait_for_user()

    #test_carter(interface)
    #return

    #base_control(world, [2.0, 0, -3*np.pi / 4], domain.get_robot().get_motion_interface(), observer)
    #return

    success = planning_loop(interface)
    print('Success:', success)
    world.destroy()

################################################################################

if __name__ == '__main__':
    #main()
    try:
        main()
    except: # BaseException as e:
        traceback.print_exc()
        #raise e
    finally:
        kill_lula()

# srl@vgilligan:~$ find . -name panda_control_moveit_rviz.launch
# ./srl_system/packages/panda_moveit_config/launch/panda_control_moveit_rviz.launch
# ./catkin_ws/src/panda_moveit_config/launch/panda_control_moveit_rviz.launch

# 3 real robot control options:
# 1) LULA + RMP
# 2) Position joint trajectory controller
# 3) LULA backend directly

# Running in IsaacSim
# 1) roslaunch isaac_bridge sim_franka.launch cooked_sim:=true config:=panda_full lula:=false

# Running the carter
# cpaxton@lokeefe:~$ ssh srl@carter
# srl@carter:~$ cd ~/deploy/srl/carter-pkg
# srl@carter:~/deploy/srl/carter-pkg$ ./apps/carter/carter -r srl -m seattle_map_res02_181214
# cpaxton@lokeefe:~/alice$ bazel run apps/samples/navigation_rosbridge

# commander.robot_remote
# navigation.control.lqr

# Running DART
# cpaxton@lokeefe:~$ franka world franka_center_right_kitchen.yaml
# cpaxton@lokeefe:~$ roslaunch lula_dart kitchen_dart_kinect1_kinect2.launch
# cpaxton@lokeefe:~/srl_system/workspace/src/brain/src/brain_ros$ rosrun lula_dart object_administrator --detect --obj=00_potted_meat_can

# Running on the real robot w/o LULA
# cpaxton@lokeefe:~$ roscore
# 1) srl@vgilligan:~$ roslaunch franka_controllers start_control.launch
# 2) srl@vgilligan:~$ cd ~/catkin_ws/src/panda_moveit_config/launch
# 2) srl@vgilligan:~/catkin_ws/src/panda_moveit_config/launch$ roslaunch panda_control_moveit_rviz.launch load_gripper:=True robot_ip:=172.16.0.2
# 3) srl@vgilligan:~$ ./srl_system/workspace/src/brain/relay.sh
# 3) cpaxton@lokeefe:~/srl_system/workspace/src/external/lula_franka$ franka viz (REQUIRED!!!)
# 4) killall move_group franka_control_node local_controller

# Running on the real robot w/ lula
# 1) franka_backend
# 2) roslaunch panda_moveit_config start_moveit.launch
# 3) ...

# Adjusting impedance thresholds to allow contact
# /franka_control/set_cartesian_impedance
# /franka_control/set_force_torque_collision_behavior
# /franka_control/set_full_collision_behavior
# /franka_control/set_joint_impedance
# srl@vgilligan:~/srl_system/workspace/src/third_party/franka_controllers/scripts
# rosed franka_controllers set_parameters

# /objects/prior_pose/00_potted_meat_can (geometry_msgs/PoseStamped)
# /objects/prior_pose/00_potted_meat_can/attachment (geometry_msgs/PoseStamped)
# /objects/prior_pose/00_potted_meat_can/attachment/decayable_weight (lula_dart/DecayableWeight)
# /objects/prior_pose/00_potted_meat_can/attachment/regulator (lula_dart/ModelPosePriorRegulator)
# /objects/prior_pose/00_potted_meat_can/attachment/status (lula_dart/ModelPosePriorStatus)
# /objects/prior_pose/00_potted_meat_can/attachment/weight
# /objects/prior_pose/00_potted_meat_can/attachment/weight/status
# /objects/prior_pose/00_potted_meat_can/decayable_weight
# /objects/prior_pose/00_potted_meat_can/heartbeat (std_msgs/String)
# /objects/prior_pose/00_potted_meat_can/inertia (geometry_msgs/PoseStamped)
# /objects/prior_pose/00_potted_meat_can/inertia/decayable_weight
# /objects/prior_pose/00_potted_meat_can/inertia/regulator
# /objects/prior_pose/00_potted_meat_can/inertia/status
# /objects/prior_pose/00_potted_meat_can/inertia/weight
# /objects/prior_pose/00_potted_meat_can/inertia/weight/status
# /objects/prior_pose/00_potted_meat_can/regulator
# /objects/prior_pose/00_potted_meat_can/status
# /objects/prior_pose/00_potted_meat_can/weight
# /objects/prior_pose/00_potted_meat_can/weight/status
