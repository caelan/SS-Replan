from pddlstream.algorithms.constraints import WILD
from pddlstream.language.constants import Action
from pddlstream.language.object import OPT_PREFIX
from pybullet_tools.utils import INF
from src.problem import ACTION_COSTS


def make_wild_skeleton(plan):
    skeleton = []
    for name, args in plan:
        new_args = [arg if isinstance(arg, str) and not arg.startswith(OPT_PREFIX) else WILD
                    for arg in args]
        skeleton.append(Action(name, new_args))
    return skeleton


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
    return [action for action in plan[len(plan_prefix):] if isinstance(action, Action)]