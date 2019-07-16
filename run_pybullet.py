#!/usr/bin/env python2

import sys
import argparse
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user
from src.planner import VIDEO_FILENAME, solve_pddlstream, simulate_plan, commands_from_plan
from src.world import World
from src.problem import pdddlstream_from_problem
from src.task import stow_block
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

def run_deteriministic(task, args):
    world = task.world
    problem = pdddlstream_from_problem(task,
        collisions=not args.cfree, teleport=args.teleport)
    solution = solve_pddlstream(problem, args)
    plan, cost, evaluations = solution
    commands = commands_from_plan(world, plan, args.defer)
    simulate_plan(world, commands, args)
    wait_for_user()

def run_stochastic(task, args):
    world = task.world
    while True:
        problem = pdddlstream_from_problem(task,
            collisions=not args.cfree, teleport=args.teleport)
        solution = solve_pddlstream(problem, args)
        plan, cost, evaluations = solution
        commands = commands_from_plan(world, plan, defer=args.defer)
        if commands is None:
            return False
        if not commands:
            return True
        simulate_plan(world, commands, args)

################################################################################

def main():
    parser = create_parser()
    args = parser.parse_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)
    world = World(use_gui=True)

    #test_rays(point_from_pose(world_from_zed_left), world.get_body(entity_name))
    #test_observation(world, entity_name, world_from_zed_left)
    #return

    task = stow_block(world)
    if args.defer:
        run_stochastic(task, args)
    else:
        run_deteriministic(task, args)
    world.destroy()

if __name__ == '__main__':
    main()

