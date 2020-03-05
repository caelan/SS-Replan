#!/usr/bin/env python2

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
#import matplotlib.pyplot as plt

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from itertools import islice

#from run_experiment import DIRECTORY, MAX_TIME
from pddlstream.utils import str_from_object, implies
from pybullet_tools.utils import read_json, SEPARATOR, INF

#from tabulate import tabulate

from run_experiment import TASK_NAMES, POLICIES, MAX_TIME, name_from_policy, ERROR_OUTCOME

# https://github.mit.edu/caelan/pddlstream-experiments/blob/master/analyze_experiment.py
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/learn_tools/analyze_experiment.py

MAX_TRIALS = INF

PRINT_ATTRIBUTES = [
    'achieved_goal',
    'total_time',
    'error',
    'plan_time',
    #'num_iterations',
    #'num_constrained',
    #'num_unconstrained',
    #'num_successes',
    #'num_actions',
    'peak_memory',
    'num_commands',
    #'total_cost',
]

ACHIEVED_GOAL = [
    'total_time',
    'plan_time',
    'num_actions',
    'total_cost',
]

POLICY_NAMES = [
    'Constain',
    'Defer',
    'Constrain+Defer',
]

################################################################################

def take(iterable, n=INF):
    if n == INF:
        n = None  # NOTE - islice takes None instead of INF
    return islice(iterable, n)

################################################################################

# https://github.mit.edu/caelan/ss/blob/master/openrave/serial_analyze.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+', help='Name of the experiment')
    args = parser.parse_args()

    outcomes_per_task = {}
    for path in args.experiments:
        for result in read_json(path):
            experiment = result['experiment']
            problem = experiment['problem']
            outcome = result['outcome']
            #policy = frozenset(experiment['policy'].items())
            policy = name_from_policy(experiment['policy'])
            outcomes_per_task.setdefault(problem['task'], {}).setdefault(policy, []).append(outcome)
    #outcomes_per_task['inspect_drawer']['constrain=0_defer=1'].append(ERROR_OUTCOME)
    #outcomes_per_task['detect_block']['constrain=1_defer=0'].append(ERROR_OUTCOME)

    # TODO: robust poses
    # TODO: intelligent IR for pour
    table = ''
    for task in TASK_NAMES:
        if task not in outcomes_per_task:
            continue
        print('\nTask: {}'.format(task))
        items = [task]
        for policy in POLICIES:
            policy = name_from_policy(policy)
            if policy not in outcomes_per_task[task]:
                continue
            outcomes = list(take(outcomes_per_task[task][policy], MAX_TRIALS))
            value_per_attribute = {}
            for outcome in outcomes:
                if outcome['error']:
                    outcome.update(ERROR_OUTCOME)
                if MAX_TIME < outcome.get('total_time', INF):
                    outcome['achieved_goal'] = False
                if not outcome['achieved_goal']:
                    outcome['total_time'] = MAX_TIME
                    outcome['plan_time'] = MAX_TIME
                for attribute, value in outcome.items():
                    if (attribute not in ['policy']) and (attribute in PRINT_ATTRIBUTES) and \
                            not isinstance(value, str) and implies(attribute in ACHIEVED_GOAL,
                                                                   outcome['achieved_goal']):
                        value_per_attribute.setdefault(attribute, []).append(float(value))

            statistics = {attribute: np.round(np.average(values), 3) # '{:.2f}'.format(
                          for attribute, values in value_per_attribute.items()} # median, min, max of solved?
            statistics['trials'] = len(outcomes)
            print('{}: {}'.format(policy, str_from_object(statistics)))
            items += [
                '{:.0f}'.format(100*statistics['achieved_goal']),
                '{:.0f}'.format(statistics['plan_time']),
            ]
        table += '{}\n\\\\ \hline\n'.format(' & '.join(items))
    print(SEPARATOR)
    print(POLICIES)
    print(table)


if __name__ == '__main__':
    main()