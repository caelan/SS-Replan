#!/usr/bin/env python

import sys
import os
import rospy
import traceback
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

import brain_ros.kitchen_domain as kitchen_domain
from brain_ros.sim_test_tools import TrialManager
from brain_ros.ros_world_state import RosObserver

from pybullet_tools.utils import LockRenderer, set_camera_pose, WorldSaver, \
    wait_for_user, wait_for_duration, Pose, Point, Euler

from src.issac import update_world, kill_lula, update_isaac_sim, set_isaac_camera, update_isaac_robot
from src.world import World
from run_pybullet import create_parser
from src.planner import solve_pddlstream, simulate_plan, commands_from_plan, extract_plan_prefix
from src.problem import pdddlstream_from_problem
from src.task import Task, close_all_doors
from src.execution import base_control

from pddlstream.language.constants import Not, And

TASKS = [
    'open_bottom', 'open_top', 'pick_spam',
    'put_away', # tomato_soup_can
    'put_spam',
]

SPAM = 'potted_meat_can'
MUSTARD = 'mustard_bottle'
TOMATO_SOUP = 'mustard_bottle'
SUGAR = 'mustard_bottle'
CHEEZIT = 'cracker_box'

YCB_OBJECTS = [SPAM, MUSTARD, TOMATO_SOUP, SUGAR, CHEEZIT]

TOP_DRAWER = 'indigo_drawer_top'

# cage_handle_from_drawer = ([0.28, 0.0, 0.0], [0.533, -0.479, -0.501, 0.485])

# Detection
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/object_administrator.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/fixed_base_suppressor.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py#L182
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L470

################################################################################

def task_from_trial_manager(world, trial_manager, task_name, fixed=False):
    objects, goal, plan = trial_manager.get_task(task=task_name, reset=True)
    goals = [(h.format(o), v) for h, v in goal for o in objects]
    print('Goals:', goals)
    #regex = re.compile(r"(\w+)\((\)\n")
    task = Task(world, movable_base=not fixed)
    init = []
    goal_literals = []

    # TODO: use the task plan to constrain solution
    # TODO: include these within task instead
    for head, value in goals:
        predicate, arguments = head.strip(')').split('(')
        args = [arg.strip() for arg in arguments.split(',')]
        if predicate == 'on_counter':
            obj, = args
            surface = 'indigo_tmp'
            atom = ('On', obj, surface)
        elif predicate == 'is_free':
            atom = ('HandEmpty',)
        elif predicate == 'gripper_closed':
            assert value is False
            value = True
            atom = ('AtGConf', world.open_gq)
        elif predicate == 'cabinet_is_open':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'open')
        elif predicate == 'cabinet_is_closed':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'closed')
        elif predicate == 'in_drawer':
            obj, surface = args
            # TODO: ensure that it actually is a surface?
            init.append(('Stackable', obj, surface))
            atom = ('On', obj, surface)
        else:
            raise NotImplementedError(predicate)
        goal_literals.append(atom if value else Not(atom))
    return task, init, goal_literals

################################################################################

def create_trial_args(**kwargs):
    args = lambda: None # Dummy class
    args.side = 'right'
    args.drawer = 'top'
    args.script_timeout = None
    args.no_planning = True
    args.debug_planner = False
    args.pause = False
    args.image = 'img%02d.png'
    args.max_count = 999999
    args.disrupt = False
    args.linear = False
    args.replan = False
    args.seed = None
    args.image_topic = '/sim/left_color_camera/image'
    args.iter = 1
    args.max_t = 3*60
    args.randomize_textures = 0.
    args.randomize_camera = 0.
    args.sigma = 0.
    args.p_sample = 0.
    args.lula_collisions = False
    args.babble = False
    # TODO: use setattr
    return args

################################################################################

def planning_loop(domain, observer, state, args, additional_init=[], additional_goals=[]):
    robot_entity = domain.get_robot()
    moveit = robot_entity.get_motion_interface() # equivalently robot_entity.planner
    world = state.world # One world per state
    #task = world.task # One task per world

    # TODO: track the plan cost
    while True:
        # TODO: Isaac class for these things

        world_state = observer.observe()
        update_world(world, domain, observer, world_state)
        problem = pdddlstream_from_problem(state, collisions=not args.cfree, teleport=args.teleport)
        problem[-2].extend(additional_init)
        problem = problem[:-1] + (And(problem[-1], *additional_goals),)
        saver = WorldSaver()
        solution = solve_pddlstream(problem, args)
        plan, cost, evaluations = solution
        plan_prefix = extract_plan_prefix(plan, defer=args.defer)
        print('Prefix:', plan_prefix)
        commands = commands_from_plan(world, plan_prefix)
        print('Commands:', commands)
        if args.watch or args.record:
            simulate_plan(state.copy(), commands, args)
        wait_for_user()
        saver.restore()
        if (commands is None) or args.teleport:
            return False
        if not commands:
            return True

        #wait_for_user()
        for command in commands:
            command.execute(domain, moveit, observer, state)

################################################################################

def main():
    parser = create_parser()
    parser.add_argument('-execute', action='store_true',
                        help="When enabled, uses the real robot")
    parser.add_argument('-fixed', action='store_true',
                        help="When enabled, fixes the robot's base")
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
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)
    use_lula = args.execute or args.lula

    rospy.init_node("STRIPStream")
    #with HideOutput():
    domain = kitchen_domain.KitchenDomain(sim=not args.execute, sigma=0, lula=use_lula)

    world = World(use_gui=True) # args.visualize)
    set_camera_pose(camera_point=[2, 0, 2])

    if args.execute:
        observer = RosObserver(domain)
        sim_manager = None
        additional_init, additional_goals = [], []
        task = Task(world, goal_on={SPAM: TOP_DRAWER},
                    movable_base=not args.fixed)
    else:
        #trial_args = parse.parse_kitchen_args()
        trial_args = create_trial_args()
        trial_manager = TrialManager(trial_args, domain, lula=use_lula)
        observer = trial_manager.observer
        sim_manager = trial_manager.sim
        #camera_point = Point(4.95, -9.03, 2.03)
        camera_point = Point(4.5, -9.5, 2.)
        camera_pose = Pose(camera_point, Euler(roll=-3*np.pi/4))
        # TODO: could make the camera follow the robot around
        set_isaac_camera(sim_manager, camera_pose)
        #trial_manager.set_camera(randomize=False)
        # Need to reset at the start
        task, additional_init, additional_goals = task_from_trial_manager(
            world, trial_manager, task_name, fixed=args.fixed)
        if args.jump:
            domain.get_robot().carter_interface = sim_manager
        #trial_manager.disable() # Disables collisions

    world_state = observer.observe() # domain.root
    with LockRenderer():
        update_world(world, domain, observer, world_state)
        #close_all_doors(world)
        if (sim_manager is not None) and task.movable_base:
            world.set_base_conf([2.0, 0, -np.pi/2])
            #world.set_initial_conf()
            update_isaac_sim(domain, observer, sim_manager, world)
        world.update_initial()
    wait_for_duration(duration=0.1)
    state = world.get_initial_state()
    # TODO: initial robot base conf is in collision

    #wait_for_user()
    #return
    #base_control(world, [2.0, 0, -3*np.pi / 4], domain.get_robot().get_motion_interface(), observer)
    #wait_for_user()
    #return

    success = planning_loop(domain, observer, state, args,
                            additional_init=additional_init, additional_goals=additional_goals)
    print('Success:', success)
    world.destroy()
    # roslaunch isaac_bridge sim_franka.launch cooked_sim:=true config:=panda_full lula:=false

if __name__ == '__main__':
    #main()
    try:
        main()
    except: # BaseException as e:
        traceback.print_exc()
        #raise e
    finally:
        kill_lula()
