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
#brain_ros.moveit.USE_MOVEIT = True

from brain_ros.kitchen_domain import KitchenDomain
#from brain_ros.demo_kitchen_domain import KitchenDomain as DemoKitchenDomain
#from grasps import *
from brain_ros.sim_test_tools import TrialManager
from brain_ros.ros_world_state import RosObserver
from isaac_bridge.carter import Carter

from pybullet_tools.utils import LockRenderer, wait_for_user, unit_from_theta, elapsed_time, \
    get_distance, quat_angle_between, quat_from_pose, point_from_pose, set_camera_pose, \
    link_from_name, get_link_pose
from pddlstream.utils import Verbose

from src.deepim import DeepIM, mean_pose_deviation, Segmentator, FullObserver
#from retired.perception import test_deepim
from src.policy import run_policy
from src.interface import Interface
from src.command import execute_commands, iterate_commands
from src.parse_brain import task_from_trial_manager, create_trial_args, TRIAL_MANAGER_TASKS
from src.utils import JOINT_TEMPLATE, SPAM, MUSTARD, TOMATO_SOUP, SUGAR, CHEEZIT, YCB_OBJECTS, ECHO_COUNTER, \
    INDIGO_COUNTER, TOP_DRAWER, BOTTOM_DRAWER
from src.visualization import add_markers
from src.issac import observe_world, kill_lula, update_robot_conf, \
    load_objects, display_kinect, dump_dict, update_objects, RIGHT, SIDES, LEFT
from src.update_isaac import update_isaac_sim, set_isaac_camera
from src.world import World
from src.task import Task, SPAM_POSE2D, pose2d_on_surface, sample_placement, close_all_doors
from src.execution import franka_open_gripper
from run_pybullet import create_parser

from examples.discrete_belief.dist import DDist, UniformDist, DeltaDist


# TODO: prevent grasp if open
# TODO: ignore grasped objects in the update
# TODO: avoid redetects on hard surfaces

def wait_for_dart_convergence(interface, detected, min_updates=10, timeout=10.0):
    start_time = time.time()
    history = defaultdict(list)
    num_updates = 0
    while elapsed_time(start_time) < timeout:
        num_updates += 1
        print('Update: {} | Time: {}'.format(num_updates, elapsed_time(start_time)))
        world_state = interface.update_state()
        observation = update_objects(interface, world_state, detected)
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
        # TODO: could calibrate closed-loop relative to the object
        # Terminate if failed to pick up
        success = execute_commands(interface, commands)
        update_robot_conf(interface)
        return success

    return run_policy(interface.task, args, observation_fn, transition_fn)

################################################################################

def test_carter(interface):
    carter = interface.carter
    # /isaac_navigation2D_status
    # /isaac_navigation2D_request

    assert carter is not None
    carter_pose = carter.current_pose
    print('Carter pose:', carter_pose)
    x, y, theta = carter_pose  # current_velocity
    pos = np.array([x, y])
    goal_pos = pos - 1.0 * unit_from_theta(theta)
    goal_pose = np.append(goal_pos, [theta])
    #goal_pose = np.append(pos, [0.])
    goal_pose = np.array([33.1, 7.789, 0.0])

    #pose_deadman_topic = '/isaac/disable_deadman_switch'
    #velocity_deadman_topic = '/isaac/enable_ros_segway_cmd'
    # carter.move_to(goal_pose) # recursion bug
    robot_entity = interface.domain.get_robot()
    robot_entity.unfix_bases() # suppressor.deactivate() => unfix
    start_time = time.time()
    timeout = 100
    while elapsed_time(start_time) < timeout:
        carter.pub_disable_deadman_switch.publish(True) # must send repeatedly
        carter.move_to_async(goal_pose)  # move_to_async | move_to_safe
        rospy.sleep(0.01)
    carter.pub_disable_deadman_switch.publish(False)
    robot_entity.fix_bases() # suppressor.activate() => fix
    # Towards the kitchen is +x (yaw=0)
    # fix base of Panda with DART is overwritten by the published message

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

#   File "/home/cpaxton/srl_system/workspace/src/brain/src/brain_ros/ros_world_state.py", line 397, in update_msg
#     self.gripper = msg.get_positions([self.gripper_joint])[0]
# TypeError: 'NoneType' object has no attribute '__getitem__'

def set_isaac_sim(interface):
    assert interface.simulation
    task = interface.task
    world = task.world
    #close_all_doors(world)
    if task.movable_base:
        world.set_base_conf([2.0, 0, -np.pi / 2])
        # world.set_initial_conf()
    else:
        for name, dist in task.prior.items():
            if name in task.prior:
                surface = task.prior[name].sample()
                sample_placement(world, name, surface, learned=True)
            else:
                sample_placement(world, name, ECHO_COUNTER, learned=False)
        # pose2d_on_surface(world, SPAM, INDIGO_COUNTER, pose2d=SPAM_POSE2D)
        # pose2d_on_surface(world, CHEEZIT, INDIGO_COUNTER, pose2d=CRACKER_POSE2D)
    update_isaac_sim(interface, world)
    # wait_for_user()

################################################################################

def simulation_setup(domain, world, args):
    # TODO: forcibly reset robot arm configuration
    # trial_args = parse.parse_kitchen_args()
    trial_args = create_trial_args()
    with Verbose(False):
        trial_manager = TrialManager(trial_args, domain, lula=args.lula)
    observer = trial_manager.observer
    #set_isaac_camera(trial_manager.sim_manager)
    trial_manager.set_camera(randomize=False)

    task_name = args.problem.replace('_', ' ')
    if task_name in TRIAL_MANAGER_TASKS:
        task = task_from_trial_manager(world, trial_manager, task_name, fixed=args.fixed)
    else:
        prior = {
            SPAM: UniformDist([TOP_DRAWER, BOTTOM_DRAWER]),
            #SPAM: UniformDist([INDIGO_COUNTER]),
            SUGAR: UniformDist([INDIGO_COUNTER]),
            CHEEZIT: UniformDist([INDIGO_COUNTER]),
        }
        goal_drawer = TOP_DRAWER  # TOP_DRAWER | BOTTOM_DRAWER
        task = Task(world, prior=prior, teleport_base=True,
                    # goal_detected=[SPAM],
                    goal_holding=SPAM,
                    #goal_on={SPAM: goal_drawer},
                    # goal_closed=[],
                    # goal_closed=[JOINT_TEMPLATE.format(goal_drawer)],
                    # goal_open=[JOINT_TEMPLATE.format(goal_drawer)],
                    movable_base=not args.fixed,
                    goal_aq=world.carry_conf,  # .values,
                    # return_init_aq=True,
                    return_init_bq=True)

    perception = FullObserver(domain) if args.observable else Segmentator(domain)
    interface = Interface(args, task, observer, trial_manager=trial_manager, deepim=perception)
    if args.jump:
        robot_entity = domain.get_robot()
        robot_entity.carter_interface = interface.sim_manager
    return interface

################################################################################

def real_setup(domain, world, args):
    # TODO: detect if lula is active via rosparam
    observer = RosObserver(domain)
    perception = DeepIM(domain, sides=[RIGHT], obj_types=YCB_OBJECTS)
    prior = {
        SPAM: UniformDist([TOP_DRAWER, BOTTOM_DRAWER]), # INDIGO_COUNTER
        #SPAM: UniformDist([INDIGO_COUNTER]),  # INDIGO_COUNTER
        SUGAR: UniformDist([INDIGO_COUNTER]),
        CHEEZIT: UniformDist([INDIGO_COUNTER]),
    }
    goal_drawer = BOTTOM_DRAWER # TOP_DRAWER | BOTTOM_DRAWER
    task = Task(world, prior=prior, teleport_base=True,
                #goal_detected=[SPAM],
                #goal_holding=SPAM,
                goal_on={SPAM: goal_drawer},
                #goal_closed=[],
                #goal_closed=[JOINT_TEMPLATE.format(goal_drawer)],
                goal_closed=[JOINT_TEMPLATE.format(drawer) for drawer in [TOP_DRAWER, BOTTOM_DRAWER]],
                #goal_open=[JOINT_TEMPLATE.format(goal_drawer)],
                movable_base=not args.fixed,
                goal_aq=world.carry_conf, #.values,
                #return_init_aq=True,
                return_init_bq=True)

    if not args.fixed:
        carter = Carter(goal_threshold_tra=0.10,
                        goal_threshold_rot=math.radians(15.),
                        vel_threshold_lin=0.01,
                        vel_threshold_ang=math.radians(1.0))
        robot_entity = domain.get_robot()
        robot_entity.carter_interface = carter
        #robot_entity.fix_bases()
    return Interface(args, task, observer, deepim=perception)


################################################################################

def main():
    parser = create_parser()
    parser.add_argument('-execute', action='store_true',
                        help="When enabled, uses the real robot_entity")
    parser.add_argument('-fixed', action='store_true',
                        help="When enabled, fixes the robot_entity's base")
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
    # TODO: samples from the belief distribution likely don't have the init flag

    # TODO: populate with initial objects even if not observed
    # TODO: reobserve thee same scene until receive good observation
    # TODO: integrate with deepim

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
    robot_entity.fix_bases()
    #robot_entity.unfix_bases()
    #print(dump_dict(robot_entity))
    #test_deepim(domain)
    #return

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
        for side in [LEFT] if interface.simulation else [RIGHT]: # TODO: simulation naming inconsistency
            display_kinect(interface, side=side)
        #if LEFT in world.cameras:
        #    del world.cameras[LEFT]
        if interface.simulation:  # TODO: move to simulation instead?
            set_isaac_sim(interface)
        world._update_initial()
        add_markers(interface.task, inverse_place=False)
    #wait_for_user()

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
# srl@carter:~/deploy/srl/carter-pkg$ ./apps/carter/carter -r 2 -m seattle_map_res02_181214
# cpaxton@lokeefe:~/alice$ bazel run apps/samples/navigation_rosbridge

# Running DART
# cpaxton@lokeefe:~$ franka world franka_center_right_kitchen.yaml
# cpaxton@lokeefe:~$ roslaunch lula_dart kitchen_dart_kinect1_kinect2.launch
# cpaxton@lokeefe:~/srl_system/workspace/src/brain/src/brain_ros$ rosrun lula_dart object_administrator --detect --j=00_potted_meat_can

# Running on the real robot w/o LULA
# cpaxton@lokeefe:~$ roscore
# 1) srl@vgilligan:~$ roslaunch franka_controllers start_control.launch
# 2) srl@vgilligan:~/srl_system/packages/panda_moveit_config/launch$ roslaunch panda_control_moveit_rviz.launch load_gripper:=True robot_ip:=172.16.0.2
# 3) srl@vgilligan:~$ ~/srl_system/workspace/src/brain/relay.sh
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
