from pddlstream.algorithms.constraints import WILD
from pddlstream.language.constants import Action, EQ, get_prefix, get_args, is_cost, is_parameter
from pddlstream.language.object import OPT_PREFIX
from pddlstream.algorithms.downward import get_fluents
from pddlstream.algorithms.algorithm import parse_domain
from pddlstream.utils import INF, implies, hash_or_id
from src.problem import ACTION_COSTS


def make_wild_skeleton(plan):
    # Can always constrain grasps and selected poses
    # Could store previous values to suggest new ones
    # Recover skeleton
    # Save parameters that shouldn't change
    # And keep their evaluations
    # If all args the same
    skeleton = []
    for name, args in plan:
        if name == 'place':
            indices = [0, 2, 3, 4] # ?o1 ?g ?rp ?o2
        else:
            indices = []
        new_args = [arg if ((index in indices) or isinstance(arg, str)) and
                           not (isinstance(arg, str) and arg.startswith(OPT_PREFIX)) else WILD
                    for index, arg in enumerate(args)]
        skeleton.append(Action(name, new_args))
    return skeleton

def reuse_facts(problem, evaluations, skeleton):
    # TODO: extract facts in the preimage of the plan
    # TODO: repackage streams
    new_facts = []
    if skeleton is None:
        return new_facts
    obj_from_id = set()
    for action, args in skeleton:
        for arg in args:
            if (arg != WILD) and not is_parameter(arg):
                obj_from_id.add(hash_or_id(arg))
    domain = parse_domain(problem.domain_pddl)
    fluents = get_fluents(domain)
    for fact in evaluations:
        predicate = get_prefix(fact)
        if (predicate == EQ) or (predicate in fluents):
            # Could technically evaluate functions as well
            continue
        if all(map(obj_from_id.__contains__, get_args(fact))):
            new_facts.append(fact)
    return new_facts

def make_exact_skeleton(plan):
    skeleton = []
    arg_from_id = {}
    var_from_id = {}
    #var_from_opt = {}
    for name, args in plan:
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                if arg.startswith(OPT_PREFIX):
                    #new_arg = WILD
                    new_arg = '?{}'.format(arg[len(OPT_PREFIX):])
                else:
                    new_arg = arg
            else:
                if 'move_arm' in name and (i not in [0, 2]) and False:
                    new_arg = WILD
                else:
                    arg_from_id[id(arg)] = arg
                    new_arg = var_from_id.setdefault(id(arg), '?w{}'.format(len(var_from_id)))
            # TODO: not sure why this fails still
            #print(arg, new_arg)
            new_args.append(new_arg)
        skeleton.append(Action(name, new_args))
        print(skeleton[-1])
    for i, var in sorted(var_from_id.items(), key=lambda pair: pair[-1]):
        print(arg_from_id[i], var)
    raw_input()
    return skeleton


def compute_plan_cost(plan):
    if plan is None:
        return INF
    cost = 0
    for name, args in plan:
        cost += ACTION_COSTS[name]
    return cost


def get_plan_postfix(plan, plan_prefix):
    return [action for action in plan[len(plan_prefix):]
            if isinstance(action, Action)]