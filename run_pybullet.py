#!/usr/bin/env python2

from __future__ import print_function

import sys
import argparse
import os
import numpy as np


sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, INF, LockRenderer, \
    get_random_seed, get_numpy_seed, set_numpy_seed, set_random_seed, print_separator, \
    VideoSaver
from src.visualization import add_markers
from src.observation import create_observable_belief, \
    transition_belief_update, create_surface_belief, UniformDist, observe_scene
from src.utils import ZED_LEFT_SURFACES
from src.debug import test_observation
from src.planner import VIDEO_TEMPLATE, DEFAULT_TIME_STEP, iterate_plan, \
    solve_pddlstream, simulate_plan, commands_from_plan, extract_plan_prefix
from src.world import World
from src.problem import pdddlstream_from_problem
from src.task import stow_block, detect_block, TASKS
from src.replan import make_wild_skeleton, compute_plan_cost, get_plan_postfix, reuse_facts


#from src.debug import dump_link_cross_sections, test_rays

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anytime', action='store_true',
                        help='Runs in an anytime mode')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-defer', action='store_true',
                        help='When enabled, defers evaluation of motion planning streams.')
    parser.add_argument('-observable', action='store_true',
                        help='Treats the state as fully observable')
    parser.add_argument('-max_time', default=3*60, type=int,
                        help='The max computation time')
    parser.add_argument('-record', action='store_true',
                        help='When enabled, records and saves a video at {}'.format(
                            VIDEO_TEMPLATE.format('<problem>')))
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
    belief = create_observable_belief(world)
    #wait_for_user('Start?')

    problem = pdddlstream_from_problem(belief,
        collisions=not args.cfree, teleport=args.teleport)
    solution = solve_pddlstream(problem, args)
    plan, cost, evaluations = solution
    commands = commands_from_plan(world, plan)
    simulate_plan(state, commands, args, record=args.record)
    wait_for_user()

################################################################################

def run_stochastic(task, args):
    # Soft constraints tell you when you succeed
    # Hard constraints are nice because they allow the solver to prune
    # Constrain to use the previous plan skeleton
    # Allow a plan skeleton to be short-cutted (don't need to move base twice)
    # Technically all values change upon each observation
    #last_cost = INF # TODO: update the remaining cost (removing attempted actions)
    # The nice thing about having a correct belief model is that you actually know what cost makes progress
    world = task.world
    state = world.get_initial_state()
    if args.observable:
        belief = create_observable_belief(world)
    else:
        belief = create_surface_belief(world, UniformDist(ZED_LEFT_SURFACES))
    # TODO: when no collisions, the robot doesn't necessarily stand in reach of the surface
    print('Prior:', belief)
    video = None
    if args.record:
        wait_for_user('Start?')
        video = VideoSaver(VIDEO_TEMPLATE.format(args.problem))

    previous_facts = []
    previous_skeleton = None
    #previous_cost = INF
    # TODO: make this a generic policy
    while True:
        print_separator(n=50)
        observation = observe_scene(world)
        print('Observation:', observation)
        belief.update(observation)
        print('Belief:', belief)
        belief.draw()
        #wait_for_user('Plan?')
        problem = pdddlstream_from_problem(belief, additional_init=previous_facts, fixed_base=True,
                                           collisions=not args.cfree, teleport=args.teleport)
        print_separator(n=25)
        plan = None
        if previous_skeleton is not None:
            print('Skeleton:', previous_skeleton)
            print('Reused facts:', sorted(previous_facts, key=lambda f: f[0]))
            # The search my get stuck otherwise
            plan, cost, certificate = solve_pddlstream(
                problem, args, max_time=15, skeleton=previous_skeleton)
            if plan is None:
                wait_for_user('Failed to adhere to plan')

        # TODO: store history of stream evaluations
        if plan is None:
            # TODO: could reusing the same problem be troublesome?
            # TODO: could require that this run be done without fixed base
            print_separator(n=25)
            cost = INF # Attempt to find a cheaper plan
            plan, cost, certificate = solve_pddlstream(problem, args, max_time=args.max_time, max_cost=cost)
        if plan is None:
            print('Failure')
            return False
        print('Preimage:', sorted(certificate.preimage_facts, key=lambda f: f[0]))
        if not plan:
            break
        print_separator(n=25)
        plan_prefix = extract_plan_prefix(plan)
        print('Prefix:', plan_prefix)
        commands = commands_from_plan(world, plan_prefix)
        if not video: # Video doesn't include planning time
            wait_for_user()
        iterate_plan(state, commands, time_step=DEFAULT_TIME_STEP)
        #simulate_plan(state, commands, args)
        transition_belief_update(belief, plan_prefix)

        plan_postfix = get_plan_postfix(plan, plan_prefix)
        previous_skeleton = make_wild_skeleton(plan_postfix)
        #previous_skeleton = make_exact_skeleton(plan_postfix)
        #last_cost = compute_plan_cost(plan_postfix)

        #previous_facts = []
        #previous_facts = certificate.preimage_facts # includes fluents!
        previous_facts = reuse_facts(problem, certificate, previous_skeleton)
        #assert compute_plan_cost(plan_prefix) + last_cost == cost
        #if not plan_postfix:
        #    break
    print('Success')
    if video:
        video.restore()
    return True

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
    # TODO: objects lift up for some reason

    task = task_fn(world)
    if not args.record:
        with LockRenderer():
            add_markers(world, inverse_place=False)
    #wait_for_user()
    # TODO: FD instantiation is slightly slow to a deepcopy

    #test_observation(world, entity_name='big_red_block0')
    #return
    if args.observable:
        run_deterministic(task, args)
    else:
        run_stochastic(task, args)
    world.destroy()

if __name__ == '__main__':
    main()

