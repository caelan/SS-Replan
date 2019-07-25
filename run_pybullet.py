#!/usr/bin/env python2

from __future__ import print_function

import sys
import argparse
import os
import numpy as np


sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, INF, LockRenderer
from src.visualization import add_markers
from src.observation import test_observation
from src.planner import VIDEO_FILENAME, solve_pddlstream, simulate_plan, commands_from_plan, extract_plan_prefix
from src.world import World
from src.problem import pdddlstream_from_problem
from src.task import stow_block, detect_block
from src.replan import make_wild_skeleton, compute_plan_cost, get_plan_postfix


#from src.debug import dump_link_cross_sections, test_rays

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', default='focused', choices=['incremental', 'focused'],
                        help='Specifies the planning algorithm that should be used')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-defer', action='store_true',
                        help='When enabled, defers evaluation of motion planning streams.')
    parser.add_argument('-optimal', action='store_true',
                        help='Runs in an anytime mode')
    #parser.add_argument('-problem', default='test_block',
    #                    help='The name of the problem to solve.')
    parser.add_argument('-max_time', default=120, type=int,
                        help='The max computation time')
    parser.add_argument('-record', action='store_true',
                        help='When enabled, records and saves a video ({})'.format(VIDEO_FILENAME))
    parser.add_argument('-seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-teleport', action='store_true',
                        help='When enabled, motion planning is skipped')
    parser.add_argument('-unit', action='store_true',
                        help='When enabled, uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes the planning world '
                             'rather than the simulated world (for debugging).')
    return parser
    # TODO: get rid of funky orientations by dropping them from some height

################################################################################

def run_deterministic(task, args):
    world = task.world
    state = world.get_initial_state()
    problem = pdddlstream_from_problem(state,
        collisions=not args.cfree, teleport=args.teleport)
    solution = solve_pddlstream(problem, args)
    plan, cost, evaluations = solution
    plan_prefix = extract_plan_prefix(plan, defer=args.defer)
    commands = commands_from_plan(world, plan_prefix)
    simulate_plan(state, commands, args)
    wait_for_user()

################################################################################

def run_stochastic(task, args):
    # TODO: relax hard constraints threshold after some time
    # Soft constraints tell you when you succeed
    # Hard constraints are nice because they allow the solver to prune
    # Constrain to use the previous plan skeleton
    # Allow a plan skeleton to be short-cutted (don't need to move base twice)
    # Technically all values change upon each observation
    #last_cost = INF # TODO: update the remaining cost (removing attempted actions)
    # The nice thing about having a correct belief model is that you actually know what cost makes progress
    last_skeleton = None
    last_cost = INF
    world = task.world
    state = world.get_initial_state()
    # TODO: make this a generic policy
    while True:
        problem = pdddlstream_from_problem(state,
            collisions=not args.cfree, teleport=args.teleport)
        plan, cost, evaluations = solve_pddlstream(problem, args, skeleton=last_skeleton)
        # TODO: first attempt cheaper path
        # TODO: store history of stream evaluations
        if (plan is None) and (last_skeleton is not None):
            #print('Failure')
            #return False
            #wait_for_user('Failure')
            plan, cost, evaluations = solve_pddlstream(problem, args)
        if plan is None:
            print('Failure')
            return False
        if not plan:
            break
        plan_prefix = extract_plan_prefix(plan, defer=args.defer)
        print('Prefix:', plan_prefix)
        commands = commands_from_plan(world, plan_prefix)
        simulate_plan(state, commands, args)
        plan_postfix = get_plan_postfix(plan, plan_prefix)
        last_skeleton = make_wild_skeleton(plan_postfix)
        #last_skeleton = make_exact_skeleton(plan_postfix)
        last_cost = compute_plan_cost(plan_postfix)
        assert compute_plan_cost(plan_prefix) + last_cost == cost
        if not plan_postfix:
            break
    print('Success')
    return True

################################################################################

def main():
    parser = create_parser()
    args = parser.parse_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)
    world = World(use_gui=True)

    #task = stow_block(world)
    task = detect_block(world)
    #task = relocate_block(world)
    if not args.record:
        with LockRenderer():
            add_markers(world, inverse_place=False)
    #wait_for_user()

    #test_observation(world, entity_name='big_red_block0')
    #return

    if args.defer:
        run_stochastic(task, args)
    else:
        run_deterministic(task, args)
    world.destroy()

if __name__ == '__main__':
    main()

