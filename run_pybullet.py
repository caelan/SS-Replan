#!/usr/bin/env python2

import sys
import argparse
import cProfile
import pstats
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, LockRenderer, WorldSaver, VideoSaver

from src.world import World
from src.problem import pdddlstream_from_problem
from src.command import State, Wait, execute_plan
from src.task import stow_block
#from src.debug import dump_link_cross_sections, test_rays

#from examples.pybullet.pr2.run import post_process
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import print_solution
from pddlstream.utils import INF
from pddlstream.language.stream import StreamInfo
from pddlstream.algorithms.constraints import PlanConstraints

VIDEO_FILENAME = 'video.mp4'

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

def solve_pddlstream(world, problem, args, debug=False):
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', stream_map.keys())

    stream_info = {
        # TODO: check if already on the stove
        'compute-pose-kin': StreamInfo(p_success=0.5, eager=True),
        'compute-angle-kin': StreamInfo(p_success=0.5, eager=True),
        'test-door': StreamInfo(p_success=0, eager=True),
        'plan-pick': StreamInfo(),
        'plan-pull': StreamInfo(),
        'plan-base-motion': StreamInfo(overhead=1e3),
        'plan-arm-motion': StreamInfo(overhead=1e2),
        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, negate=True),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, negate=True),
        'test-cfree-traj-pose': StreamInfo(p_success=1e-1, negate=True),
        # 'Distance': FunctionInfo(p_success=0.99, opt_fn=lambda q1, q2: BASE_CONSTANT),
        # 'MoveCost': FunctionInfo(lambda t: BASE_CONSTANT),
    }

    # skeleton = [
    #     ('calibrate', [WILD, WILD, WILD]),
    #     ('move_base', [WILD, WILD, WILD]),
    #     ('pull', ['indigo_drawer_top_joint', WILD, WILD,
    #               'indigo_drawer_top', WILD, WILD, WILD, WILD, WILD  ]),
    #     ('move_base', [WILD, WILD, WILD]),
    #     ('pick', ['big_red_block0', WILD, WILD, WILD,
    #               'indigo_drawer_top', WILD, WILD, WILD, WILD]),
    #     ('move_base', [WILD, WILD, WILD]),
    # ]
    #constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    constraints = PlanConstraints()

    success_cost = 0 if args.optimal else INF
    planner = 'max-astar' if args.optimal else 'ff-astar'
    search_sample_ratio = 1 # TODO: could try decreasing
    max_planner_time = 10

    pr = cProfile.Profile()
    pr.enable()
    with LockRenderer(lock=not args.visualize):
        saver = WorldSaver()
        if args.algorithm == 'focused':
            # TODO: option to only consider costs during local optimization
            # effort_weight = 0 if args.optimal else 1
            effort_weight = 1e-3 if args.optimal else 1
            solution = solve_focused(problem, constraints=constraints, stream_info=stream_info,
                                     planner=planner, max_planner_time=max_planner_time,
                                     unit_costs=args.unit, success_cost=success_cost,
                                     max_time=args.max_time, verbose=True, debug=debug,
                                     unit_efforts=True, effort_weight=effort_weight,
                                     # bind=True, max_skeletons=None,
                                     search_sample_ratio=search_sample_ratio)
        elif args.algorithm == 'incremental':
            solution = solve_incremental(problem, constraints=constraints,
                                         planner=planner, max_planner_time=max_planner_time,
                                         unit_costs=args.unit, success_cost=success_cost,
                                         max_time=args.max_time, verbose=True, debug=debug)
        else:
            raise ValueError(args.algorithm)
        saver.restore()

    # print([(s.cost, s.time) for s in SOLUTIONS])
    # print(SOLUTIONS)
    print_solution(solution)
    plan, cost, evaluations = solution
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(25)  # cumtime | tottime
    commands = commands_from_plan(world, plan)
    return commands

################################################################################

def commands_from_plan(world, plan):
    if plan is None:
        return None
    # TODO: propagate the state
    commands = []
    for action, params in plan:
        if action in ['move_base', 'move_arm', 'move_gripper', 'pick', 'pull', 'calibrate']:
            commands.extend(params[-1].commands)
        elif action == 'place':
            commands.extend(params[-1].reverse().commands)
        elif action in ['cook']:
            commands.append(Wait(world, steps=100))
        else:
            raise NotImplementedError(action)
    return commands

def simulate_plan(world, commands, args):
    if commands is None:
        wait_for_user()
        return
    initial_state = State(savers=[WorldSaver()], attachments=world.initial_attachments.values())
    wait_for_user()
    time_step = None if args.teleport else 0.02
    if args.record:
        with VideoSaver(VIDEO_FILENAME):
            execute_plan(world, initial_state, commands, time_step=time_step)
    else:
        execute_plan(world, initial_state, commands, time_step=time_step)
    wait_for_user()

################################################################################

def main():
    # TODO: handle relative poses for drawers
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

