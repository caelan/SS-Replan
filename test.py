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

from pybullet_tools.utils import LockRenderer, set_camera_pose, WorldSaver, get_max_velocity, get_max_force, wait_for_user

from issac import update_world, kill_lula
from world import World
from run import solve_pddlstream, create_parser, simulate_plan
from problem import pdddlstream_from_problem
from execution import open_gripper

from pddlstream.language.constants import Not, And

def constraints_from_plan(plan):
    # TODO: use the task plan to constrain solution
    raise NotImplementedError()

def goal_formula_from_goal(goals):
    #regex = re.compile(r"(\w+)\((\)\n")
    # TODO: return initial literals as well
    init = []
    goal_literals = []
    for head, value in goals:
        predicate, arguments = head.strip(')').split('(')
        args = [arg.strip() for arg in arguments.split(',')]
        if predicate == 'on_counter':
            obj, = args
            surface = 'indigo_tmp'
            atom = ('On', obj, surface)
        #elif predicate == '???':
        #    atom = ('Holding', obj)
        elif predicate == 'is_free':
            #arm, = args
            atom = ('HandEmpty',)
        #elif predicate == 'gripper_closed':
        #    #arm, = args
        #    atom = None
        elif predicate == 'cabinet_is_open':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'open')
        elif predicate == 'cabinet_is_closed':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'closed')
        elif predicate == 'in_drawer':
            obj, drawer = args
            surface = '{}_joint'.format(drawer)
            init.append(('Stackable', obj, surface))
            atom = ('On', obj, surface)
        else:
            print('Skipping {}={}'.format(head, value))
            continue
            #raise NotImplementedError(predicate)
        goal_literals.append(atom if value else Not(atom))
    return init, And(*goal_literals)

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

TASKS = [
    'open_bottom', 'open_top', 'pick_spam',
    'put_away', # tomato_soup_can
    'put_spam',
]

################################################################################

def main():
    parser = create_parser()
    parser.add_argument('-fixed', action='store_true',
                        help='TBD')
    parser.add_argument('-lula', action='store_true',
                        help='TBD')
    parser.add_argument('-problem', default=TASKS[2], choices=TASKS,
                        help='TBD')
    parser.add_argument('-watch', action='store_true',
                        help='TBD')
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
    trial_manager.set_camera(randomize=False)

    # Need to reset at the start
    objects, goal, plan = trial_manager.get_task(task=task, reset=True)
    goals = [(h.format(o), v) for h, v in goal for o in objects]
    print(goals)
    additional_init, goal_formula = goal_formula_from_goal(goals)
    # TODO: fixed_base_suppressors
    #trial_manager.disable() # Disables collisions

    # TODO: why can't I use this earlier?
    robot_entity = domain.get_robot()
    moveit = robot_entity.get_motion_interface() # equivalently robot_entity.planner
    open_gripper(moveit)
    #world_state = observer.observe() # domain.root

    world = World(use_gui=True) # args.visualize)
    #print([get_max_velocity(world.robot, joint) for joint in world.arm_joints])
    #print([get_max_force(world.robot, joint) for joint in world.arm_joints])
    #print([get_max_velocity(world.robot, joint) for joint in world.gripper_joints])
    #print([get_max_force(world.robot, joint) for joint in world.gripper_joints])
    #return

    set_camera_pose(camera_point=[1, -1, 2])

    world_state = observer.observe() # domain.root
    with LockRenderer():
        update_world(world, domain, observer, world_state)

    # TODO: initial robot base conf is in collision
    problem = pdddlstream_from_problem(world, movable_base=False, fixed_base=True,
                                       collisions=not args.cfree, teleport=args.teleport)
    problem[-2].extend(additional_init)
    problem = problem[:-1] + (goal_formula,)
    saver = WorldSaver()
    commands = solve_pddlstream(world, problem, args)
    if args.watch:
        simulate_plan(world, commands, args)
    else:
        wait_for_user()
    if commands is None:
        return
    #wait_for_user()
    saver.restore()

    #sim_manager = trial_manager.sim
    #sim_manager.pause() # Careful! This actually does pause the system
    #rospy.sleep(1.) # Small sleep might be needed
    #sim_manager.pause() # The second invocation resumes

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