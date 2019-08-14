from __future__ import print_function

from pybullet_tools.utils import wait_for_user, VideoSaver, print_separator, INF
from src.belief import create_observable_belief, transition_belief_update
from src.planner import VIDEO_TEMPLATE, solve_pddlstream, extract_plan_prefix, commands_from_plan
from src.problem import pdddlstream_from_problem
from src.replan import get_plan_postfix, make_exact_skeleton, reuse_facts

def run_policy(task, args, observation_fn, transition_fn):
    world = task.world
    if args.observable:
        belief = create_observable_belief(world)  # Fast
    else:
        belief = task.create_belief()
    print('Prior:', belief)
    video = None
    if args.record:
        wait_for_user('Start?')
        video = VideoSaver(VIDEO_TEMPLATE.format(args.problem))

    previous_facts = []
    previous_skeleton = None
    while True:
        print_separator(n=50)
        observation = observation_fn()
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
            plan, cost, certificate = solve_pddlstream(
                problem, args, max_time=30, skeleton=previous_skeleton)
            if plan is None:
                wait_for_user('Failed to adhere to plan')

        # TODO: store history of stream evaluations
        if plan is None:
            problem = pdddlstream_from_problem(belief, fixed_base=not task.movable_base,
                                               additional_init=previous_facts,
                                               collisions=not args.cfree, teleport=args.teleport)
            print_separator(n=25)
            plan, cost, certificate = solve_pddlstream(problem, args, max_time=args.max_time, max_cost=cost)
        if plan is None:
            print('Failure!')
            return False
        # print('Preimage:', sorted(certificate.preimage_facts, key=lambda f: f[0]))
        if not plan:
            break
        print_separator(n=25)
        plan_prefix = extract_plan_prefix(plan)
        print('Prefix:', plan_prefix)
        commands = commands_from_plan(world, plan_prefix)
        print('Commands:', commands)
        if not video:  # Video doesn't include planning time
            wait_for_user()
        result = transition_fn(belief, commands)  # Break if none?
        transition_belief_update(belief, plan_prefix)
        plan_postfix = get_plan_postfix(plan, plan_prefix)
        previous_skeleton = make_exact_skeleton(plan_postfix)  # make_wild_skeleton
        previous_facts = reuse_facts(problem, certificate, previous_skeleton)  # []

    print('Success!')
    if video:
        video.restore()
    return True
