from itertools import count

from pddlstream.algorithms.constraints import WILD, ORDER_PREDICATE
from pddlstream.language.constants import Action, EQ, get_prefix, get_args, is_cost, is_parameter
from pddlstream.language.object import OPT_PREFIX
from pddlstream.algorithms.downward import get_fluents
from pddlstream.algorithms.algorithm import parse_domain
from pddlstream.utils import INF, implies, hash_or_id
from src.problem import ACTION_COSTS

REUSE_ARGUMENTS = {
    # This should really be done by type instead
    'pick': [0, 2],  # ?o1 ?g
    'place': [0, 2, 3, 4], # ?o1 ?g ?rp ?o2
    # The previous pick error I had was because it moved to the carry_aq and then detected
    # However, it was next constrained to move the base rather than the arm
}


# TODO: could keep around previous base plans as long as we don't reuse them

def is_optimistic(arg):
    return isinstance(arg, str) and arg.startswith(OPT_PREFIX)

def test_reusable(name, index, arg):
    indices = REUSE_ARGUMENTS.get(name, [])
    return ((index in indices) or isinstance(arg, str)) and not is_optimistic(arg)

def make_wild_skeleton(plan):
    # Can always constrain grasps and selected poses
    # Could store previous values to suggest new ones
    # Recover skeleton
    # Save parameters that shouldn't change
    # And keep their evaluations
    # If all args the same
    skeleton = []
    for name, args in plan:
        new_args = [arg if test_reusable(name, index, arg) else WILD for index, arg in enumerate(args)]
        skeleton.append(Action(name, new_args))
        #print(len(skeleton), skeleton[-1])
    return skeleton

def make_exact_skeleton(plan):
    # TODO: spend more effort on samples that were previously discovered
    # TODO: possibly reuse the kinematic solutions as seeds
    #arg_from_id = {}
    var_from_id = {}
    count_from_prefix = {}
    skeleton = []
    for name, args in plan:
        new_args = []
        for i, arg in enumerate(args):
            #arg_from_id[id(arg)] = arg
            if test_reusable(name, i, arg):
                new_args.append(arg)
            else:
                key = id(arg)
                if key not in var_from_id:
                    if is_optimistic(arg):
                        var_from_id[key] = '?{}'.format(arg[len(OPT_PREFIX):]) # WILD
                    else:
                        prefix = str(arg)[:2].lower() # 'w'
                        num = next(count_from_prefix.setdefault(prefix, count()))
                        var_from_id[key] = '?{}{}'.format(prefix, num)
                new_args.append(var_from_id[key])
        skeleton.append(Action(name, new_args))
        #print(skeleton[-1])
    #for i, var in sorted(var_from_id.items(), key=lambda pair: pair[-1]):
    #    print(arg_from_id[i], var)
    # TODO: could fall back on wild if this fails
    # TODO: this fails for placing (due to carry_conf / rest_conf disagreement)
    return skeleton

################################################################################

def reuse_facts(problem, certificate, skeleton):
    # TODO: repackage streams
    # TODO: recover the full axiom + action plan
    # TODO: recover the plan preimage annotated with use time
    # Some supporting args are quantified out and thus lack some facts
    new_facts = []
    if skeleton is None:
        return new_facts
    reuse_objs = set()
    for action, args in skeleton:
        for arg in args:
            if (arg != WILD) and not is_parameter(arg):
                reuse_objs.add(hash_or_id(arg))

    # The reuse relpose omission is due to the fact that the initial pose was selected
    # (which is populated in the initial state)
    order_predicate = ORDER_PREDICATE.format('')
    domain = parse_domain(problem.domain_pddl)
    fluents = get_fluents(domain)
    for fact in certificate.preimage_facts:
        predicate = get_prefix(fact)
        if (predicate in {order_predicate, EQ}) or (predicate in fluents):
            # Could technically evaluate functions as well
            continue
        if all(isinstance(arg, str) or (hash_or_id(arg) in reuse_objs)
               for arg in get_args(fact)):
            new_facts.append(fact)
    return new_facts

################################################################################

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