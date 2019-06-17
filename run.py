#!/usr/bin/env python3

#from __future__ import printfunction, divide

import sys
import argparse
import random
import cProfile
import pstats

sys.path.extend([
    '/home/caelan/Programs/pddlstream',
    '/home/caelan/Programs/pddlstream/examples/pybullet/utils',
])


from pybullet_tools.utils import wait_for_user, sample_placement, link_from_name, \
    LockRenderer, WorldSaver, user_input, wait_for_duration, draw_base_limits, aabb_union, get_aabb, \
    get_bodies, VideoSaver
from utils import World, get_block_path, BLOCK_SIZES, BLOCK_COLORS, SURFACES
from problem import pdddlstream_from_problem
from command import State, Wait

#from examples.pybullet.pr2.run import post_process
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import print_solution
from pddlstream.utils import INF
from pddlstream.language.stream import StreamInfo


################################################################################

def commands_from_plan(world, plan):
    if plan is None:
        return None
    # TODO: propagate the state
    commands = []
    for action, params in plan:
        if action in ['move_base', 'move_arm', 'pick']:
            commands.extend(params[-1].commands)
        elif action == 'place':

            print(params[-1].reverse().commands)
            commands.extend(params[-1].reverse().commands)
        elif action in ['cook']:
            commands.append(Wait(world, steps=100))
        else:
            raise NotImplementedError(action)
    return commands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', default='focused',
                        help='Specifies the algorithm')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-optimal', action='store_true',
                        help='Runs in an anytime mode')
    #parser.add_argument('-problem', default='test_block',
    #                    help='The name of the problem to solve.')
    parser.add_argument('-seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-max_time', default=120, type=int,
                        help='The max time')
    parser.add_argument('-teleport', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-unit', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()
    #if args.seed is not None:
    #    set_seed(args.seed)

    world = World(use_gui=True)
    for joint in world.kitchen_joints:
        #world.open_door(joint)
        world.close_door(joint)
    world.open_gripper()
    #print(world.door_links)

    block_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    world.add_body(block_name, get_block_path(block_name))

    initial_surface = random.choice(SURFACES)
    print('Initial surface:', initial_surface)
    pose = sample_placement(world.get_body(block_name), world.kitchen,
                            bottom_link=link_from_name(world.kitchen, initial_surface))
    assert pose is not None

    full_aabb = aabb_union(get_aabb(body) for body in get_bodies() if body != world.floor)
    #draw_aabb(full_aabb)
    lower, upper = full_aabb
    base_limits = (lower[:2], upper[:2])
    draw_base_limits(base_limits)
    #wait_for_user()

    pddlstream_problem = pdddlstream_from_problem(world, base_limits=base_limits,
                                                  collisions=not args.cfree, teleport=args.teleport)

    #for joint in world.kitchen_joints:
    #    joint_from_name(world.kitchen, joint)
    #test_grasps(world, block_name)

    _, _, _, stream_map, init, goal = pddlstream_problem
    print('Init:', init)
    print('Goal:', goal)
    # print('Streams:', stream_map.keys())

    stream_info = {
        # TODO: check if already on the stove
        'inverse-kinematics': StreamInfo(),
        'plan-base-motion': StreamInfo(overhead=1e1),
        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, negate=True),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, negate=True),
        'test-cfree-traj-pose': StreamInfo(p_success=1e-1, negate=True),  # TODO: this applies to arm and base trajs
        # 'test-cfree-traj-grasp-pose': StreamInfo(negate=True),
        #'Distance': FunctionInfo(p_success=0.99, opt_fn=lambda q1, q2: BASE_CONSTANT),
        # 'MoveCost': FunctionInfo(lambda t: BASE_CONSTANT),
    }

    success_cost = 0 if args.optimal else INF
    planner = 'ff-astar' if args.optimal else 'ff-wastar1'
    search_sample_ratio = 2
    max_planner_time = 10

    pr = cProfile.Profile()
    pr.enable()
    with LockRenderer(lock=not args.visualize):
        saver = WorldSaver()
        if args.algorithm == 'focused':
            # TODO: option to only consider costs during local optimization
            # effort_weight = 0 if args.optimal else 1
            effort_weight = 1e-3 if args.optimal else 1
            solution = solve_focused(pddlstream_problem, stream_info=stream_info,
                                     planner=planner, max_planner_time=max_planner_time, debug=False,
                                     unit_costs=args.unit, success_cost=success_cost,
                                     max_time=args.max_time, verbose=True,
                                     unit_efforts=True, effort_weight=effort_weight,
                                     # bind=True, max_skeletons=None,
                                     search_sample_ratio=search_sample_ratio)
        elif args.algorithm == 'incremental':
            solution = solve_incremental(pddlstream_problem,
                                         planner=planner, max_planner_time=max_planner_time,
                                         unit_costs=args.unit, success_cost=success_cost,
                                         max_time=args.max_time, verbose=True)
        else:
            raise ValueError(args.algorithm)
        saver.restore()

    #print([(s.cost, s.time) for s in SOLUTIONS])
    #print(SOLUTIONS)
    print_solution(solution)
    plan, cost, evaluations = solution
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(25)  # cumtime | tottime

    if plan is None:
        wait_for_user()
        world.destroy()
        return

    commands = commands_from_plan(world, plan)
    state = State()
    wait_for_user()

    time_step = None if args.teleport else 0.02
    with VideoSaver('video.mp4'):
        for i, command in enumerate(commands):
            print('\nCommand {:2}: {}'.format(i, command))
            # TODO: skip to end
            # TODO: downsample
            for j, _ in enumerate(command.iterate(world, state)):
                state.derive()
                if j == 0:
                    continue
                if time_step is None:
                    wait_for_duration(1e-2)
                    user_input('Command {:2} | step {:2} | Next?'.format(i, j))
                else:
                    wait_for_duration(time_step)

    wait_for_user()
    world.destroy()

if __name__ == '__main__':
    main()

