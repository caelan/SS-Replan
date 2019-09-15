#!/usr/bin/env python2

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
#import matplotlib.pyplot as plt

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])


#from run_experiment import DIRECTORY, MAX_TIME
from pddlstream.utils import str_from_object, implies
from pybullet_tools.utils import read_json

#from tabulate import tabulate

from run_experiment import TASK_NAMES, POLICIES, MAX_TIME, name_from_policy

# https://github.mit.edu/caelan/pddlstream-experiments/blob/master/analyze_experiment.py
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/learn_tools/analyze_experiment.py


PRINT_ATTRIBUTES = [
    'achieved_goal',
    'total_time',
    'error',
    #'plan_time',
    #'plan_start_time',
    #'num_iterations',
    #'num_constrained',
    #'num_unconstrained',
    #'num_successes',
    #'num_actions',
    'num_commands',
    #'total_cost',
]

ACHIEVED_GOAL = [
    'total_time',
    'plan_time',
    'plan_start_time',
    'num_actions',
    'total_cost',
]


ERROR_OUTCOME = {
    'achieved_goal': False,
    'total_time': MAX_TIME,
    'plan_time': 0,
    'num_iterations': 0,
    'num_constrained': 0,
    'num_unconstrained': 0,
    'num_successes': 0,
    'num_actions': 0,
    'num_commands': 0,
    'total_cost': 0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='Name of the experiment')
    args = parser.parse_args()

    outcomes_per_task = {}
    for result in read_json(args.experiment):
        experiment = result['experiment']
        problem = experiment['problem']
        outcome = result['outcome']
        #policy = frozenset(experiment['policy'].items())
        policy = name_from_policy(experiment['policy'])
        outcomes_per_task.setdefault(problem['task'], {}).setdefault(policy, []).append(outcome)

    for task in TASK_NAMES:
        if task not in outcomes_per_task:
            continue
        print('\nTask: {}'.format(task))
        for policy in POLICIES:
            policy = name_from_policy(policy)
            if policy not in outcomes_per_task[task]:
                continue
            value_per_attribute = {}
            for outcome in outcomes_per_task[task][policy]:
                if outcome['error']:
                    outcome.update(ERROR_OUTCOME)
                outcome['total_time'] = min(outcome['total_time'], MAX_TIME)
                for attribute, value in outcome.items():
                    if (attribute not in ['policy']) and (attribute in PRINT_ATTRIBUTES) and \
                            not isinstance(value, str) and implies(attribute in ACHIEVED_GOAL, outcome['achieved_goal']):
                        value_per_attribute.setdefault(attribute, []).append(value)
            statistics = {attribute: '{:.2f}'.format(np.mean(values)) for attribute, values in value_per_attribute.items()}
            statistics['trials'] = len(outcomes_per_task[task][policy])
            print('{}: {}'.format(policy, str_from_object(statistics)))
            # TODO: robust poses
            # TODO: intelligent IR for pour

if __name__ == '__main__':
    main()