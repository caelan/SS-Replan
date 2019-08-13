#!/usr/bin/env python2

from __future__ import print_function

import sys
import os
import rospy
import traceback
import numpy as np
import math

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from brain_ros.kitchen_domain import KitchenDomain
#from brain_ros.demo_kitchen_domain import KitchenDomain as DemoKitchenDomain
#from grasps import *
from brain_ros.sim_test_tools import TrialManager
from brain_ros.ros_world_state import RosObserver
from isaac_bridge.carter import Carter

from pybullet_tools.utils import LockRenderer, set_camera_pose, wait_for_user, unit_from_theta

from src.interface import Interface
from src.command import execute_commands
from src.parse_brain import task_from_trial_manager, create_trial_args, TASKS
from src.observation import create_observable_belief
from src.visualization import add_markers
from src.issac import update_world, kill_lula, update_isaac_sim, dump_dict
from src.world import World
from run_pybullet import create_parser
from src.planner import solve_pddlstream, simulate_plan, commands_from_plan, extract_plan_prefix
from src.problem import pdddlstream_from_problem
from src.task import Task, CRACKER_POSE2D, BOX_POSE2D, pose2d_on_surface, sample_placement
from src.replan import get_plan_postfix, make_wild_skeleton

SPAM = 'potted_meat_can'
MUSTARD = 'mustard_bottle'
TOMATO_SOUP = 'tomato_soup_can'
SUGAR = 'sugar_box'
CHEEZIT = 'cracker_box'

YCB_OBJECTS = [SPAM, MUSTARD, TOMATO_SOUP, SUGAR, CHEEZIT]

ECHO_COUNTER = 'echo'
INDIGO_COUNTER = 'indigo_tmp'
TOP_DRAWER = 'indigo_drawer_top'
JOINT_TEMPLATE = '{}_joint'

################################################################################

def planning_loop(interface):
    args = interface.args
    domain = interface.domain
    world = interface.world
    belief = create_observable_belief(world)
    last_skeleton = None
    while True:
        # The difference in state is that this one is only used for visualization
        state = world.get_initial_state() # TODO: create from belief for holding
        problem = pdddlstream_from_problem(belief, collisions=not args.cfree, teleport=args.teleport)

        wait_for_user('Plan?')
        plan, cost, evaluations = solve_pddlstream(problem, args, skeleton=last_skeleton)
        if (plan is None) and (last_skeleton is not None):
            plan, cost, evaluations = solve_pddlstream(problem, args)

        plan_prefix = extract_plan_prefix(plan)
        print('Prefix:', plan_prefix)
        commands = commands_from_plan(world, plan_prefix)
        print('Commands:', commands)
        if args.watch or args.record:
            # TODO: operate on real state
            simulate_plan(state.copy(), commands, args, record=args.record)
        wait_for_user()

        state.assign()
        if (commands is None) or args.teleport or args.cfree:
            return False
        if not commands:
            return True

        #wait_for_user()
        execute_commands(interface, commands)
        plan_postfix = get_plan_postfix(plan, plan_prefix)
        last_skeleton = make_wild_skeleton(plan_postfix)
        interface.localize_all()
        update_world(world, domain, interface.observer)


################################################################################

def test_carter(domain, carter):
    x, y, theta = carter.current_pose  # current_velocity
    pos = np.array([x, y])
    goal_pos = pos + 0.2 * unit_from_theta(theta)
    goal_pose = np.append(goal_pos, [theta])
    # goal_pose = np.append(pos, [0.])

    # carter.move_to(goal_pose) # recursion bug
    carter.move_to_safe(goal_pose)  # move_to_async | move_to_safe
    # move_to_open_loop | move_to_safe_followed_by_openloop

    # carter.simple_move(0.1) # simple_move | simple_stop
    # rospy.sleep(2.0)
    # carter.simple_stop()
    domain.get_robot().carter_interface = carter
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
    parser.add_argument('-problem', default=TASKS[2], choices=TASKS,
                        help='The name of the task')
    parser.add_argument('-watch', action='store_true',
                        help='When enabled, plans are visualized in PyBullet before executing in IsaacSim')
    args = parser.parse_args()
    task_name = args.problem.replace('_', ' ')
    np.set_printoptions(precision=3, suppress=True)
    args.watch |= args.execute

    # srl_system/packages/isaac_bridge/configs/ycb_table_config.json
    # srl_system/packages/isaac_bridge/configs/ycb_table_graph.json
    # srl_system/packages/isaac_bridge/configs/panda_full_config.json
    # srl_system/packages/isaac_bridge/configs/panda_full_graph.json
    # alice/assets/maps/seattle_map_res02_181214.config.json

    # https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/brain/src/brain_ros/lula_policies.py#L464
    rospy.init_node("STRIPStream")
    #with HideOutput():
    #if args.execute:
    #    domain = DemoKitchenDomain(sim=not args.execute, use_carter=True)
    #else:
    domain = KitchenDomain(sim=not args.execute, sigma=0, lula=args.lula)
    robot_entity = domain.get_robot()
    robot_entity.suppress_fixed_bases() # Not as much error?
    #robot_entity.unsuppress_fixed_bases() # Significant error
    # Significant error without either
    #print(dump_dict(robot_entity))
    # TODO: forcibly reset robot configuration

    world = World(use_gui=True) # args.visualize)
    # /home/cpaxton/srl_system/workspace/src/external/lula_franka

    if args.execute:
        observer = RosObserver(domain)
        task = Task(world,
                    objects=[SPAM, SUGAR], #, CHEEZIT],
                    #goal_holding=[SPAM],
                    goal_on={SPAM: TOP_DRAWER},
                    #goal_closed=[],
                    goal_closed=[JOINT_TEMPLATE.format(TOP_DRAWER)], #, 'indigo_drawer_bottom_joint'],
                    #goal_open=[JOINT_TEMPLATE.format(TOP_DRAWER)],
                    movable_base=not args.fixed,
                    return_init_bq=True, return_init_aq=True)

        if not args.fixed:
            carter = Carter(goal_threshold_tra=0.10,
                            goal_threshold_rot=math.radians(15.),
                            vel_threshold_lin=0.01,
                            vel_threshold_ang=math.radians(1.0))
            robot_entity.carter_interface = carter
            robot_entity.unsuppress_fixed_bases()
        interface = Interface(args, task, observer)
    else:
        #trial_args = parse.parse_kitchen_args()
        trial_args = create_trial_args()
        trial_manager = TrialManager(trial_args, domain, lula=args.lula)
        observer = trial_manager.observer

        # TODO: could make the camera follow the robot_entity around
        ##camera_point = Point(4.95, -9.03, 2.03)
        #camera_point = Point(4.5, -9.5, 2.)
        #camera_pose = Pose(camera_point, Euler(roll=-3*np.pi/4))
        #set_isaac_camera(sim_manager, camera_pose)
        #trial_manager.set_camera(randomize=False)

        # Need to reset at the start
        task = task_from_trial_manager(world, trial_manager, task_name, fixed=args.fixed, objects=YCB_OBJECTS)
        #trial_manager.disable() # Disables collisions
        interface = Interface(args, task, observer, trial_manager=trial_manager)
        if args.jump:
            robot_entity.carter_interface = interface.sim_manager

    # Can disable lula world objects to improve speed
    # Adjust DART to get a better estimate for the drawer joints
    #localize_all(world_state)
    #wait_for_user()
    #print('Entities:', sorted(world_state.entities))
    with LockRenderer(lock=True):
        # Need to do expensive computation before localize_all
        # Such as loading the meshes
        update_world(world, domain, observer)
        interface.localize_all()
        update_world(world, domain, observer)
        #close_all_doors(world)
        if interface.simulation and task.movable_base:
            world.set_base_conf([2.0, 0, -np.pi/2])
            #world.set_initial_conf()
            update_isaac_sim(interface, world)
        world.update_initial()
        add_markers(world, inverse_place=False)

    if interface.simulation:
        pose2d_on_surface(world, SPAM, INDIGO_COUNTER, pose2d=BOX_POSE2D)
        pose2d_on_surface(world, CHEEZIT, INDIGO_COUNTER, pose2d=CRACKER_POSE2D)
        for name in [MUSTARD, TOMATO_SOUP, SUGAR]:
            sample_placement(world, name, ECHO_COUNTER, learned=False)
        update_isaac_sim(interface, world)
        world.update_initial()
        wait_for_user()

    #base_control(world, [2.0, 0, -3*np.pi / 4], domain.get_robot().get_motion_interface(), observer)
    #return

    success = planning_loop(interface)
    print('Success:', success)
    world.destroy()

    # /tracker/axe/joint_states
    # /tracker/baker/joint_states

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

# 3 real robot control options:
# 1) LULA + RMP
# 2) Position joint trajectory controller
# 3) LULA backend directly

# Running in IsaacSim
# 1) roslaunch isaac_bridge sim_franka.launch cooked_sim:=true config:=panda_full lula:=false

# Running on the real robot w/o LULA
# 1) roslaunch franka_controllers lula_control.launch
# 2) roslaunch panda_moveit_config panda_control_moveit_rviz.launch load_gripper:=True robot_ip:=172.16.0.2
# 3) killall move_group franka_control_node local_controller

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
