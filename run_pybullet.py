#!/usr/bin/env python2

import sys
import argparse
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from src.planner import VIDEO_FILENAME, solve_pddlstream, simulate_plan
from src.world import World
from src.problem import pdddlstream_from_problem
from src.task import stow_block
#from src.debug import dump_link_cross_sections, test_rays

#from examples.pybullet.pr2.run import post_process

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', default='focused', choices=['incremental', 'focused'],
                        help='Specifies the planning algorithm that should be used')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
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
    problem = pdddlstream_from_problem(task,
        collisions=not args.cfree, teleport=args.teleport)
    commands = solve_pddlstream(world, problem, args)
    simulate_plan(world, commands, args)
    world.destroy()

if __name__ == '__main__':
    main()

