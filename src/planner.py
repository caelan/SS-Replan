from __future__ import print_function

import cProfile
import pstats
import math

from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.language.constants import print_solution
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.language.rule import RULES
from pddlstream.language.object import Object, OptimisticObject

from pddlstream.utils import INF
from pybullet_tools.utils import LockRenderer, WorldSaver, wait_for_user, VideoSaver
from src.command import Wait, iterate_plan

VIDEO_FILENAME = 'video.mp4'
REPLAN_ACTIONS = {'calibrate'}


def solve_pddlstream(problem, args, skeleton=None, max_cost=INF, debug=False):
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', stream_map.keys())
    RULES[:] = []
    Object.reset()
    OptimisticObject.reset()

    opt_gen_fn = PartialInputs(unique=False)
    stream_info = {
        'test-gripper': StreamInfo(p_success=0, eager=True),
        'test-door': StreamInfo(p_success=0, eager=True),
        'test-near-pose': StreamInfo(p_success=0, eager=True),
        'test-near-joint': StreamInfo(p_success=0, eager=True),

        # TODO: need to be careful about conditional effects
        'compute-pose-kin': StreamInfo(opt_gen_fn=PartialInputs(unique=True),
                                       p_success=0.5, eager=True),
        #'compute-angle-kin': StreamInfo(p_success=0.5, eager=True),
        #'sample-pose': StreamInfo(opt_gen_fn=opt_gen_fn),
        #'sample-nearby-pose': StreamInfo(opt_gen_fn=opt_gen_fn),
        #'sample-grasp': StreamInfo(opt_gen_fn=opt_gen_fn),

        'plan-pick': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
        #'fixed-plan-pick': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
        'plan-pull': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
        #'fixed-plan-pull': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
        #'plan-calibrate-motion': StreamInfo(opt_gen_fn=opt_gen_fn),
        'plan-base-motion': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e3, defer=True),
        'plan-arm-motion': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e2, defer=True),
        #'plan-gripper-motion': StreamInfo(opt_gen_fn=opt_gen_fn),

        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, negate=True),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, negate=True),
        'test-cfree-traj-pose': StreamInfo(p_success=1e-1, negate=True),

        'test-ofree-ray-pose': StreamInfo(p_success=1e-3, negate=True),
        'test-ofree-ray-grasp': StreamInfo(p_success=1e-3, negate=True),

        #'Distance': FunctionInfo(p_success=0.99, opt_fn=lambda bq1, bq2: BASE_CONSTANT),
        #'MoveCost': FunctionInfo(lambda t: BASE_CONSTANT),
    }
    #print(set(stream_map) - set(stream_info))
    replan_actions = REPLAN_ACTIONS if args.defer else set()
    skeletons = None if skeleton is None else [skeleton]
    constraints = PlanConstraints(skeletons=skeletons, max_cost=max_cost, exact=True)

    success_cost = 0 if args.anytime else INF
    planner = 'ff-astar' if args.anytime else 'ff-wastar1'
    search_sample_ratio = 1 # TODO: could try decreasing
    max_planner_time = 10

    pr = cProfile.Profile()
    pr.enable()
    saver = WorldSaver()
    with LockRenderer(lock=not args.visualize):
        # TODO: option to only consider costs during local optimization
        # effort_weight = 0 if args.anytime else 1
        effort_weight = 1e-3 if args.anytime else 1
        #effort_weight = 0
        solution = solve_focused(problem, constraints=constraints, stream_info=stream_info,
                                 replan_actions=replan_actions,
                                 initial_complexity=5,
                                 planner=planner, max_planner_time=max_planner_time,
                                 unit_costs=args.unit, success_cost=success_cost,
                                 max_time=args.max_time, verbose=True, debug=debug,
                                 unit_efforts=True, effort_weight=effort_weight, max_effort=INF,
                                 # bind=True, max_skeletons=None,
                                 search_sample_ratio=search_sample_ratio)
        saver.restore()

    # print([(s.cost, s.time) for s in SOLUTIONS])
    # print(SOLUTIONS)
    print_solution(solution)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(25)  # cumtime | tottime
    return solution

################################################################################

def extract_plan_prefix(plan, defer=False):
    if plan is None:
        return None
    prefix = []
    for action in plan:
        name, args = action
        prefix.append(action)
        if defer and (name in REPLAN_ACTIONS):
            break
    return prefix

DEFAULT_TIME_STEP = 0.02

def commands_from_plan(world, plan):
    if plan is None:
        return None
    # TODO: propagate the state
    commands = []
    for action, params in plan:
        if action in ['move_base', 'move_arm', 'move_gripper', 'pick', 'pull']:
            commands.extend(params[-1].commands)
        elif action == 'detect':
            commands.append(params[-1])
        elif action == 'place':
            commands.extend(params[-1].reverse().commands)
        elif action in ['cook', 'calibrate']:
            # TODO: calibrate action that uses fixed_base_suppressor
            steps = int(math.ceil(2.0 / DEFAULT_TIME_STEP))
            commands.append(Wait(world, steps=steps))
        else:
            raise NotImplementedError(action)
    return commands

################################################################################

def simulate_plan(state, commands, args, time_step=DEFAULT_TIME_STEP):
    wait_for_user()
    if commands is None:
        return
    time_step = None if args.teleport else time_step
    if args.record:
        with VideoSaver(VIDEO_FILENAME):
            iterate_plan(state, commands, time_step=time_step)
    else:
        iterate_plan(state, commands, time_step=time_step)
