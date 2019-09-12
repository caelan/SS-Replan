from __future__ import print_function

import time

from pybullet_tools.utils import wait_for_duration, wait_for_user, print_separator, INF, elapsed_time
from src.belief import create_observable_belief, transition_belief_update
from src.planner import solve_pddlstream, extract_plan_prefix, commands_from_plan
from src.problem import pdddlstream_from_problem
from src.replan import get_plan_postfix, make_exact_skeleton, reuse_facts, OBSERVATION_ACTIONS, \
    STOCHASTIC_ACTIONS


def run_policy(task, args, observation_fn, transition_fn, constrain=True, defer=True,
               max_time=5*60, max_planning_time=30):
    replan_actions = OBSERVATION_ACTIONS if args.deterministic else STOCHASTIC_ACTIONS
    defer_actions = replan_actions if defer else set()
    world = task.world
    if args.observable:
        # TODO: problematic if not observable
        belief = create_observable_belief(world)  # Fast
    else:
        belief = task.create_belief()
    belief.liquid.update(task.init_liquid)
    print('Prior:', belief)

    previous_facts = []
    previous_skeleton = None
    start_time = time.time()
    plan_duration = 0
    while elapsed_time(start_time) < max_time:
        print_separator(n=50)
        observation = observation_fn(belief) # TODO: could allow this to be an arbitrary belief transformation
        print('Observation:', observation)
        belief.update(observation)
        print('Belief:', belief)
        belief.draw()
        plan_time = time.time()
        #wait_for_user('Plan?')
        plan, cost = None, INF
        problem = pdddlstream_from_problem(belief, fixed_base=True, additional_init=previous_facts,
                                           collisions=not args.cfree, teleport=args.teleport)
        if previous_skeleton is not None:
            print_separator(n=25)
            print('Skeleton:', previous_skeleton)
            print('Reused facts:', sorted(previous_facts, key=lambda f: f[0]))
            # TODO: could compare to the previous plan cost
            planning_time = min(max_time - elapsed_time(start_time), max_planning_time, args.max_time)
            plan, cost, certificate = solve_pddlstream(belief, problem, args, max_time=planning_time,
                                                       skeleton=previous_skeleton, replan_actions=defer_actions)
            if plan is None:
                print('Failed to adhere to plan')
                #wait_for_user()

        # TODO: store history of stream evaluations
        if plan is None:
            problem = pdddlstream_from_problem(belief, fixed_base=not task.movable_base,# or not constrain,
                                               additional_init=previous_facts,
                                               collisions=not args.cfree, teleport=args.teleport)
            print_separator(n=25)
            planning_time = min(max_time - elapsed_time(start_time), args.max_time)
            plan, cost, certificate = solve_pddlstream(belief, problem, args, max_time=planning_time,
                                                       max_cost=cost, replan_actions=defer_actions)
        plan_duration += elapsed_time(plan_time)
        #wait_for_duration(elapsed_time(plan_time)) # Mocks the real planning time
        if plan is None:
            break
        #print('Preimage:', sorted(certificate.preimage_facts, key=lambda f: f[0]))
        if not plan:
            print('Success!')
            return True
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
            commands = commands_from_plan(world, sequence)
            print('Commands:', commands)
            success &= transition_fn(belief, commands) and \
                       transition_belief_update(belief, sequence) and \
                       belief.check_consistent()
        if success and constrain:
            plan_postfix = get_plan_postfix(plan, plan_prefix)
            # TODO: exit if plan_postfix is empty?
            previous_skeleton = make_exact_skeleton(world, plan_postfix)  # make_exact_skeleton | make_wild_skeleton
            previous_facts = reuse_facts(problem, certificate, previous_skeleton)  # []
        else:
            previous_facts = []
            previous_skeleton = None
    print('Failure!')
    return False
