from __future__ import print_function

import time
import traceback

from pybullet_tools.utils import wait_for_duration, wait_for_user, \
    print_separator, INF, elapsed_time
from pddlstream.utils import get_peak_memory_in_kb, str_from_object
from pddlstream.language.constants import Certificate, PDDLProblem
from src.belief import create_observable_belief, transition_belief_update, create_observable_pose_dist
from src.planner import solve_pddlstream, extract_plan_prefix, commands_from_plan
from src.problem import pdddlstream_from_problem, get_streams
from src.replan import get_plan_postfix, make_exact_skeleton, reuse_facts, OBSERVATION_ACTIONS, \
    STOCHASTIC_ACTIONS, make_wild_skeleton
from src.utils import BOWL

# TODO: max time spent reattempting streams flag (might not be needed actually)
# TODO: process binding blows up for detect_drawer
UNCONSTRAINED_FIXED_BASE = False
MAX_RESTART_TIME = 2*60
REPLAN_ITERATIONS = 1 # 1 | INF
REPLAN_FAILURES = False


def random_restart(belief, args, problem, max_time=INF, max_iterations=INF,
                   max_planner_time=INF, **kwargs):
    domain_pddl, constant_map, _, _, init, goal_formula = problem
    start_time = time.time()
    task = belief.task
    world = task.world
    iterations = 0
    while (elapsed_time(start_time) < max_time) and (iterations < max_iterations):
        iterations += 1
        try:
            stream_pddl, stream_map = get_streams(world, teleport_base=task.teleport_base,
                                                  collisions=not args.cfree, teleport=args.teleport)
            problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
            remaining_time = min(max_time - elapsed_time(start_time), max_planner_time)
            plan, plan_cost, certificate = solve_pddlstream(belief, problem, args, max_time=remaining_time, **kwargs)
            if plan is not None:
                return plan, plan_cost, certificate
        except KeyboardInterrupt as e:
            raise e
        except MemoryError as e:
            traceback.print_exc()
            if not REPLAN_FAILURES:
                raise e
            break
        except Exception as e:
            traceback.print_exc()
            if not REPLAN_FAILURES:
                raise e
        # FastDownward translator runs out of memory
    return None, INF, Certificate(all_facts=[], preimage_facts=[])

def run_policy(task, args, observation_fn, transition_fn, constrain=True, defer=True, # serialize=True,
               max_time=10*60, max_constrained_time=1.5*60, max_unconstrained_time=INF):
    replan_actions = OBSERVATION_ACTIONS if args.deterministic else STOCHASTIC_ACTIONS
    defer_actions = replan_actions if defer else set()
    world = task.world
    if args.observable:
        # TODO: problematic if not observable
        belief = create_observable_belief(world)  # Fast
    else:
        belief = task.create_belief()
    if BOWL in task.objects: # TODO: hack for bowl
        belief.pose_dists[BOWL] = create_observable_pose_dist(world, BOWL)
    belief.liquid.update(task.init_liquid)
    print('Prior:', belief)

    previous_facts = []
    previous_skeleton = None
    total_start_time = time.time()
    plan_time = 0
    achieved_goal = False
    num_iterations = num_constrained = num_unconstrained = num_successes = 0
    num_actions = num_commands = total_cost = 0
    while elapsed_time(total_start_time) < max_time:
        print_separator(n=50)
        num_iterations += 1
        # TODO: could allow this to be an arbitrary belief transformation
        observation = observation_fn(belief)
        print('Observation:', observation)
        belief.update(observation)
        print('Belief:', belief)
        belief.draw()

        #wait_for_user('Plan?')
        fixed_base = UNCONSTRAINED_FIXED_BASE or not task.movable_base or not constrain
        plan_start_time = time.time()
        plan, plan_cost = None, INF
        if constrain and (previous_skeleton is not None):
            # TODO: could constrain by comparing to the previous plan cost
            num_constrained += 1
            print_separator(n=25)
            print('Skeleton:', previous_skeleton)
            print('Reused facts:', sorted(previous_facts, key=lambda f: f[0]))
            problem = pdddlstream_from_problem(belief, additional_init=previous_facts,
                                               collisions=not args.cfree, teleport=args.teleport)
            planning_time = min(max_time - elapsed_time(total_start_time), max_constrained_time) #, args.max_time)
            plan, plan_cost, certificate = random_restart(belief, args, problem, max_time=planning_time, max_iterations=1,
                                                          skeleton=previous_skeleton, replan_actions=defer_actions)
            if plan is None:
                print('Failed to solve with plan constraints')
                #wait_for_user()
        #elif not fixed_base:
        #    num_unconstrained += 1
        #    problem = pdddlstream_from_problem(belief, additional_init=previous_facts,
        #                                       collisions=not args.cfree, teleport=args.teleport)
        #    print_separator(n=25)
        #    planning_time = min(max_time - elapsed_time(total_start_time)) # , args.max_time)
        #    plan, plan_cost, certificate = solve_pddlstream(belief, problem, args, max_time=planning_time,
        #                                                    replan_actions=defer_actions)
        #    if plan is None:
        #        print('Failed to solve when allowing fixed base')

        if plan is None:
            # TODO: might be helpful to add additional facts here in the future
            num_unconstrained += 1 # additional_init=previous_facts,
            problem = pdddlstream_from_problem(belief, fixed_base=fixed_base,
                                               collisions=not args.cfree, teleport=args.teleport)
            print_separator(n=25)
            planning_time = min(max_time - elapsed_time(total_start_time), max_unconstrained_time) #, args.max_time)
            plan, plan_cost, certificate = random_restart(belief, args, problem, max_time=planning_time,
                                                          max_planner_time=MAX_RESTART_TIME,
                                                          max_iterations=REPLAN_ITERATIONS,
                                                          max_cost=plan_cost, replan_actions=defer_actions)

        plan_time += elapsed_time(plan_start_time)
        #wait_for_duration(elapsed_time(plan_start_time)) # Mocks the real planning time
        if plan is None:
            break
        #print('Preimage:', sorted(certificate.preimage_facts, key=lambda f: f[0]))
        if not plan:
            achieved_goal = True
            break
        print_separator(n=25)
        plan_prefix = extract_plan_prefix(plan, replan_actions=replan_actions)
        print('Prefix:', plan_prefix)
        # sequences = [plan_prefix]
        sequences = [[action] for action in plan_prefix]

        success = belief.check_consistent()
        for i, sequence in enumerate(sequences):
            if not success:
                break
            print(i, sequence)
            num_actions += len(sequence)
            commands = commands_from_plan(world, sequence)
            num_commands += len(commands)
            print('Commands:', commands)
            success &= transition_fn(belief, commands) and \
                       transition_belief_update(belief, sequence) and \
                       belief.check_consistent()
            total_cost += sum(command.cost for command in commands)
        num_successes += success

        # TODO: store history of stream evaluations
        if success and constrain:
            plan_postfix = get_plan_postfix(plan, plan_prefix)
            # TODO: exit if plan_postfix is empty?
            # TODO: make_exact_skeleton still has as bug in it
            previous_skeleton = make_wild_skeleton(world, plan_postfix)
            #previous_skeleton = make_exact_skeleton(world, plan_postfix)  # make_exact_skeleton | make_wild_skeleton
            previous_facts = reuse_facts(problem, certificate, previous_skeleton)  # []
        else:
            previous_skeleton = None
            previous_facts = []

    if achieved_goal:
        print('Success!')
    else:
        print('Failure!')
    # TODO: timed out flag
    # TODO: store current and peak memory usage
    data = {
        'achieved_goal': achieved_goal,
        'total_time': elapsed_time(total_start_time),
        'plan_time': plan_time,
        'num_iterations': num_iterations,
        'num_constrained': num_constrained,
        'num_unconstrained': num_unconstrained,
        'num_successes': num_successes,
        'num_actions': num_actions,
        'num_commands': num_commands,
        'peak_memory': get_peak_memory_in_kb(),
        'total_cost': total_cost,
    }
    print('Data:', str_from_object(data))
    return data
