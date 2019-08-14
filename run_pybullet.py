#!/usr/bin/env python2

from __future__ import print_function

import sys
import argparse
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, LockRenderer, \
    get_random_seed, get_numpy_seed
from src.command import create_state
from src.visualization import add_markers
from src.belief import create_observable_belief
from src.observe import observe_pybullet
#from src.debug import test_observation
from src.planner import VIDEO_TEMPLATE, iterate_commands, \
    solve_pddlstream, simulate_plan, commands_from_plan
from src.world import World
from src.problem import pdddlstream_from_problem
from src.task import TASKS
from src.policy import run_policy
#from src.debug import dump_link_cross_sections, test_rays

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anytime', action='store_true',
                        help='Runs in an anytime mode')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    #parser.add_argument('-defer', action='store_true',
    #                    help='When enabled, defers evaluation of motion planning streams.')
    parser.add_argument('-deterministic', action='store_true',
                        help='Treats actions as fully deterministic')
    parser.add_argument('-observable', action='store_true',
                        help='Treats the state as fully observable')
    parser.add_argument('-max_time', default=3*60, type=int,
                        help='The max computation time')
    parser.add_argument('-record', action='store_true',
                        help='When enabled, records and saves a video at {}'.format(
                            VIDEO_TEMPLATE.format('<problem>')))
    #parser.add_argument('-seed', default=None,
    #                    help='The random seed to use.')
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

def run_plan(task, real_state, args):
    world = task.world
    belief = create_observable_belief(world)
    #wait_for_user('Start?')

    problem = pdddlstream_from_problem(belief,
        collisions=not args.cfree, teleport=args.teleport)
    solution = solve_pddlstream(problem, args, replan_actions={})
    plan, cost, evaluations = solution
    commands = commands_from_plan(world, plan)
    simulate_plan(real_state, commands, args)
    wait_for_user()

################################################################################

def main():
    task_names = [fn.__name__ for fn in TASKS]
    print('Tasks:', task_names)
    parser = create_parser()
    parser.add_argument('-problem', default=task_names[0], choices=task_names,
                        help='The name of the problem to solve.')
    args = parser.parse_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    #set_random_seed(None) # Doesn't ensure deterministic
    #set_numpy_seed(None)
    print('Random seed:', get_random_seed())
    print('Numpy seed:', get_numpy_seed())

    np.set_printoptions(precision=3, suppress=True)
    world = World(use_gui=True)
    task_fn_from_name = {fn.__name__: fn for fn in TASKS}
    task_fn = task_fn_from_name[args.problem]

    task = task_fn(world)
    world.update_initial()
    if not args.record:
        with LockRenderer():
            add_markers(world, inverse_place=False)
    #wait_for_user()
    # TODO: FD instantiation is slightly slow to a deepcopy
    # 4650801/25658    2.695    0.000    8.169    0.000 /home/caelan/Programs/srlstream/pddlstream/pddlstream/algorithms/skeleton.py:114(do_evaluate_helper)
    #test_observation(world, entity_name='big_red_block0')
    #return

    real_state = create_state(world)
    if args.deterministic and args.observable:
        run_plan(task, real_state, args)
    else:
        observation_fn = lambda: observe_pybullet(world)
        # restore real_state just in case?
        transition_fn = lambda belief, commands: iterate_commands(real_state, commands)
        # simulate_plan(real_state, commands, args)
        run_policy(task, args, observation_fn, transition_fn)
    world.destroy()
    # TODO: make the sink extrude from the mesh

if __name__ == '__main__':
    main()
