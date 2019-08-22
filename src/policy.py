from __future__ import print_function

from pybullet_tools.utils import wait_for_user, print_separator, INF
from src.belief import create_observable_belief, transition_belief_update
from src.planner import solve_pddlstream, extract_plan_prefix, commands_from_plan
from src.problem import pdddlstream_from_problem
from src.replan import get_plan_postfix, make_exact_skeleton, reuse_facts, OBSERVATION_ACTIONS, \
    STOCHASTIC_ACTIONS


def run_policy(task, args, observation_fn, transition_fn):
    replan_actions = OBSERVATION_ACTIONS if args.deterministic else STOCHASTIC_ACTIONS
    world = task.world
    if args.observable:
        belief = create_observable_belief(world)  # Fast
    else:
        belief = task.create_belief()
    print('Prior:', belief)

    previous_facts = []
    previous_skeleton = None
    while True:
        print_separator(n=50)
        observation = observation_fn(belief) # TODO: could allow this to be an arbitrary belief transformation
        print('Observation:', observation)
        belief.update(observation)
        print('Belief:', belief)
        belief.draw()
        wait_for_user('Plan?')
        problem = pdddlstream_from_problem(belief, fixed_base=True, additional_init=previous_facts,
                                           collisions=not args.cfree, teleport=args.teleport)
        print_separator(n=25)
        plan, cost = None, INF
        if previous_skeleton is not None:
            print('Skeleton:', previous_skeleton)
            print('Reused facts:', sorted(previous_facts, key=lambda f: f[0]))
            # TODO: could compare to the previous plan cost
            plan, cost, certificate = solve_pddlstream(belief, problem, args, max_time=30, skeleton=previous_skeleton,
                                                       replan_actions=replan_actions)
            if plan is None:
                print('Failed to adhere to plan')
                wait_for_user()

        # TODO: store history of stream evaluations
        if plan is None:
            problem = pdddlstream_from_problem(belief, fixed_base=not task.movable_base,
                                               additional_init=previous_facts,
                                               collisions=not args.cfree, teleport=args.teleport)
            print_separator(n=25)
            plan, cost, certificate = solve_pddlstream(belief, problem, args, max_time=args.max_time,
                                                       max_cost=cost, replan_actions=replan_actions)
        if plan is None:
            print('Failure!')
            return False
        #print('Preimage:', sorted(certificate.preimage_facts, key=lambda f: f[0]))
        if not plan:
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
            commands = commands_from_plan(world, sequence)
            print('Commands:', commands)
            # if not video:  # Video doesn't include planning time
            #    wait_for_user() # TODO: move to the pybullet version?
            success &= transition_fn(belief, commands) and \
                       transition_belief_update(belief, sequence) and \
                       belief.check_consistent()
        if success:
            plan_postfix = get_plan_postfix(plan, plan_prefix)
            # TODO: exit if plan_postfix is empty
            previous_skeleton = make_exact_skeleton(world, plan_postfix)  # make_exact_skeleton | make_wild_skeleton
            previous_facts = reuse_facts(problem, certificate, previous_skeleton)  # []
        else:
            previous_facts = []
            previous_skeleton = None

    print('Success!')
    return True
