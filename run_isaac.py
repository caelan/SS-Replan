#!/usr/bin/env python

import sys
import os

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

import rospy
import traceback
import numpy as np

import brain_ros.kitchen_domain as kitchen_domain
from brain_ros.sim_test_tools import TrialManager

from pybullet_tools.utils import LockRenderer, set_camera_pose, WorldSaver, \
    get_max_velocity, get_max_force, wait_for_user, get_camera, dump_body, link_from_name, \
    point_from_pose, get_yaw, get_pitch, get_link_pose, draw_pose, multiply, \
    invert, set_joint_positions, set_base_values

from src.issac import update_world, kill_lula, update_isaac_sim
from src.world import World
from run_pybullet import solve_pddlstream, create_parser, simulate_plan
from src.problem import pdddlstream_from_problem
from src.execution import open_gripper, control_base
from src.task import Task

from pddlstream.language.constants import Not, And

NONE = 'none'

TASKS = [
    'open_bottom', 'open_top', 'pick_spam',
    'put_away', # tomato_soup_can
    'put_spam',
    NONE,
]

################################################################################

def goal_formula_from_goal(world, goals, plan):
    #regex = re.compile(r"(\w+)\((\)\n")
    task = Task(world, movable_base=False, fixed_base=True)
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
    return task, init, And(*goal_literals)

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

def main():
    parser = create_parser()
    #parser.add_argument('-fixed', action='store_true',
    #                    help="When enabled, fixes the robot's base")
    parser.add_argument('-lula', action='store_true',
                        help='When enabled, uses LULA instead of JointState control')
    parser.add_argument('-problem', default=TASKS[2], choices=TASKS,
                        help='The name of the task')
    parser.add_argument('-watch', action='store_true',
                        help='When enabled, plans are visualized in PyBullet before executing in IsaacSim')
    args = parser.parse_args()
    task = args.problem.replace('_', ' ')
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)

    rospy.init_node("test")
    #with HideOutput():
    domain = kitchen_domain.KitchenDomain(sim=True, sigma=0, lula=args.lula)

    #trial_args = parse.parse_kitchen_args()
    trial_args = create_trial_args()
    trial_manager = TrialManager(trial_args, domain, lula=args.lula)
    observer = trial_manager.observer
    sim_manager = trial_manager.sim
    trial_manager.set_camera(randomize=False)

    # Need to reset at the start
    if task != NONE:
        objects, goal, plan = trial_manager.get_task(task=task, reset=True)

    # TODO: why can't I use this earlier?
    robot_entity = domain.get_robot()
    moveit = robot_entity.get_motion_interface() # equivalently robot_entity.planner
    #world_state = observer.observe() # domain.root

    world = World(use_gui=True) # args.visualize)
    set_camera_pose(camera_point=[1, -1, 2])

    if task != NONE:
        goals = [(h.format(o), v) for h, v in goal for o in objects]
        print('Goals:', goals)
        task, additional_init, goal_formula = goal_formula_from_goal(world, goals, plan)
    # TODO: fixed_base_suppressors
    #trial_manager.disable() # Disables collisions

    world_state = observer.observe() # domain.root
    with LockRenderer():
        update_world(world, domain, observer, world_state)
        #base_positions = [1.0, 0, np.pi]
        #set_base_values(world.robot, base_positions)
        #world.set_initial_conf()
        #update_isaac_sim(domain, observer, sim_manager, world)
    wait_for_user()

    #goal_values = [1.5, -1, np.pi]
    #control_base(goal_values, moveit, observer)
    if task == NONE:
        return

    #cage_handle_from_drawer = ([0.28, 0.0, 0.0], [0.533, -0.479, -0.501, 0.485])
    #drawer_link = link_from_name(world.kitchen, 'indigo_drawer_top')
    #draw_pose(multiply(get_link_pose(world.kitchen, drawer_link), (cage_handle_from_drawer)))

    # TODO: initial robot base conf is in collision
    problem = pdddlstream_from_problem(task, collisions=not args.cfree, teleport=args.teleport)
    problem[-2].extend(additional_init)
    problem = problem[:-1] + (And(problem[-1], goal_formula),)
    saver = WorldSaver()
    commands = solve_pddlstream(world, problem, args)
    if args.watch or args.record:
        simulate_plan(world, commands, args)
    else:
        wait_for_user()
    if (commands is None) or args.teleport:
        return
    #wait_for_user()
    saver.restore()

    #sim_manager.pause() # Careful! This actually does pause the system
    #rospy.sleep(1.) # Small sleep might be needed
    #sim_manager.pause() # The second invocation resumes

    # roslaunch isaac_bridge sim_franka.launch cooked_sim:=true config:=panda_full lula:=false
    for command in commands:
        command.execute(domain, moveit, observer)
    world.destroy()

if __name__ == '__main__':
    #main()
    try:
        main()
    except: # BaseException as e:
        traceback.print_exc()
        #raise e
    finally:
        kill_lula()