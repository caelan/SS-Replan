#!/usr/bin/env python2

import sys
import argparse
import cProfile
import pstats
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.pr2_utils import get_viewcone
from pybullet_tools.utils import wait_for_user, LockRenderer, WorldSaver, VideoSaver, \
    get_pose, RED, apply_alpha, set_pose, multiply, step_simulation, set_all_static, \
    Euler, Pose, Point, stable_z, link_from_name

from observation import KITCHEN_FROM_ZED_LEFT, CAMERA_MATRIX, DEPTH, test_observation
from utils import get_block_path, BLOCK_SIZES, BLOCK_COLORS, \
    DRAWER_JOINTS, joint_from_name, get_ycb_obj_path, COUNTERS, DRAWERS, CABINETS
from world import World
from problem import pdddlstream_from_problem
from command import State, Wait, execute_plan
from stream import get_stable_gen

#from examples.pybullet.pr2.run import post_process
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import print_solution
from pddlstream.utils import INF
from pddlstream.language.stream import StreamInfo
from pddlstream.algorithms.constraints import PlanConstraints, WILD

def create_args():
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
    parser.add_argument('-record', action='store_true',
                        help='Records a video')
    parser.add_argument('-teleport', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-unit', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    return parser.parse_args()
    # TODO: get rid of funky orientations by dropping them from some height

################################################################################

def solve_pddlstream(world, problem, args, debug=False):
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    # print('Streams:', stream_map.keys())

    stream_info = {
        # TODO: check if already on the stove
        'compute-pose-kin': StreamInfo(p_success=0.5, eager=True),
        'compute-angle-kin': StreamInfo(p_success=0.5, eager=True),
        'test-door': StreamInfo(p_success=0, eager=True),
        'plan-pick': StreamInfo(),
        'plan-pull': StreamInfo(),
        'plan-base-motion': StreamInfo(overhead=1e2),
        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, negate=True),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, negate=True),
        'test-cfree-traj-pose': StreamInfo(p_success=1e-1, negate=True),
        # 'Distance': FunctionInfo(p_success=0.99, opt_fn=lambda q1, q2: BASE_CONSTANT),
        # 'MoveCost': FunctionInfo(lambda t: BASE_CONSTANT),
    }

    skeleton = [
        ('calibrate', [WILD, WILD, WILD]),
        ('move_base', [WILD, WILD, WILD]),
        ('pull', ['indigo_drawer_top_joint', WILD, WILD,
                  'indigo_drawer_top', WILD, WILD, WILD, WILD, WILD  ]),
        ('move_base', [WILD, WILD, WILD]),
        ('pick', ['big_red_block0', WILD, WILD, WILD,
                  'indigo_drawer_top', WILD, WILD, WILD, WILD]),
        ('move_base', [WILD, WILD, WILD]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    #constraints = PlanConstraints()

    #success_cost = 0 if args.optimal else INF
    success_cost = INF
    planner = 'max-astar' if args.optimal else 'ff-wastar1'
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
    pstats.Stats(pr).sort_stats('cumtime').print_stats(25)  # cumtime | tottime
    commands = commands_from_plan(world, plan)
    return commands

################################################################################

def commands_from_plan(world, plan):
    if plan is None:
        return None
    # TODO: propagate the state
    commands = []
    for action, params in plan:
        if action in ['move_base', 'move_arm', 'pick', 'pull', 'calibrate']:
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
    initial_state = State(savers=[WorldSaver()], attachments=dict(world.initial_attachments))
    wait_for_user()
    time_step = None if args.teleport else 0.02
    if args.record:
        with VideoSaver('video.mp4'):
            execute_plan(world, initial_state, commands, time_step=time_step)
    else:
        execute_plan(world, initial_state, commands, time_step=time_step)
    wait_for_user()

################################################################################

def main():
    args = create_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)

    world = World(use_gui=True)
    #for joint in world.kitchen_joints:
    #for name in LEFT_VISIBLE:
    for name in DRAWER_JOINTS[1:2]:
        joint = joint_from_name(world.kitchen, name)
        world.open_door(joint)
        #world.close_door(joint)
    world.open_gripper()
    #dump_link_cross_sections(world, link_name='indigo_tmp')
    #wait_for_user()

    entity_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    entity_path = get_block_path(entity_name)
    #entity_name = 'potted_meat_can'
    #entity_path = get_ycb_obj_path(entity_name)
    world.add_body(entity_name, entity_path)

    #obstruction_name = 'cracker_box'
    #obstruction_path = get_ycb_obj_path(obstruction_name)
    #world.add_body(obstruction_name, obstruction_path, color=np.ones(4))
    #obstruction_body = world.get_body(obstruction_name)
    #z = stable_z(obstruction_body, world.kitchen, link_from_name(world.kitchen, COUNTERS[0]))
    #set_pose(obstruction_body, Pose(Point(0.2, 1.2, z), Euler(yaw=np.pi/4)))

    #test_grasps(world, entity_name)
    set_all_static()

    # TODO: could intersect convex with half plane
    world_from_zed_left = multiply(get_pose(world.kitchen), KITCHEN_FROM_ZED_LEFT)
    cone_body = get_viewcone(depth=DEPTH, camera_matrix=CAMERA_MATRIX, color=apply_alpha(RED, 0.1))
    set_pose(cone_body, world_from_zed_left)
    step_simulation()
    #test_rays(point_from_pose(world_from_zed_left), world.get_body(entity_name))
    #test_observation(world, entity_name, world_from_zed_left)
    #return

    #surface_name = random.choice(DRAWERS)
    surface_name = DRAWERS[1] # COUNTERS | DRAWERS | SURFACES | CABINETS
    #surface_name = 'indigo_tmp' # hitman_drawer_top_joint | hitman_tmp | indigo_tmp
    print('Initial surface:', surface_name)
    with WorldSaver():
        placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)
        pose, = next(placement_gen(entity_name, surface_name), (None,))
    assert pose is not None
    pose.assign()

    problem = pdddlstream_from_problem(
        world, close_doors=True, return_home=True,
        collisions=not args.cfree, teleport=args.teleport)
    commands = solve_pddlstream(world, problem, args)
    simulate_plan(world, commands, args)
    world.destroy()

if __name__ == '__main__':
    main()

