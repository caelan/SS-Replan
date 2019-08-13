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

def test_reusable(name, index, arg):
    indices = REUSE_ARGUMENTS.get(name, [])
    return ((index in indices) or isinstance(arg, str)) and \
           not (isinstance(arg, str) and arg.startswith(OPT_PREFIX))

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
    skeleton = []
    arg_from_id = {}
    var_from_id = {}
    count_from_prefix = {}
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
                #if 'move_arm' in name and (i not in [0, 2]) and False:
                #    new_arg = WILD
                #else:
                #prefix = 'w'
                prefix = str(arg)[:2].lower()
                num = count_from_prefix.get(prefix, 0)
                count_from_prefix[prefix] = count_from_prefix.get(prefix, 0) + 1
                var = '?{}{}'.format(prefix, num)
                arg_from_id[id(arg)] = arg
                new_arg = var_from_id.setdefault(id(arg), var)
            # TODO: not sure why this fails still
            #print(arg, new_arg)
            new_args.append(new_arg)
        skeleton.append(Action(name, new_args))
        print(skeleton[-1])
    for i, var in sorted(var_from_id.items(), key=lambda pair: pair[-1]):
        print(arg_from_id[i], var)
    #raw_input('Continue?')
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