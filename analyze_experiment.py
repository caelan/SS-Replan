#!/usr/bin/env python2

from __future__ import print_function

import argparse
import os
import sys
import math
import scipy.stats
import numpy as np
#import matplotlib.pyplot as plt

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])


#from run_experiment import DIRECTORY, MAX_TIME
from pddlstream.utils import read_pickle, str_from_object
from pybullet_tools.utils import read_json

from collections import OrderedDict, defaultdict
#from tabulate import tabulate

from run_experiment import TASK_NAMES, POLICIES, MAX_TIME

# https://github.mit.edu/caelan/pddlstream-experiments/blob/master/analyze_experiment.py
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/learn_tools/analyze_experiment.py


ERROR_OUTCOME = {
    'achieved_goal': False,
    'total_time': MAX_TIME,
    'plan_start_time': 0,
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
        task = result['experiment']['task']
        for outcome in result['outcomes']:
            policy = frozenset(outcome['policy'].items())
            outcomes_per_task.setdefault(task, {}).setdefault(policy, []).append(outcome)

    for task in TASK_NAMES:
        if task not in outcomes_per_task:
            continue
        print('\nTask: {}'.format(task))
        for policy in POLICIES:
            policy = frozenset(policy.items())
            if policy not in outcomes_per_task[task]:
                continue
            name = '_'.join('{}={:d}'.format(key, value) for key, value in sorted(policy))
            value_per_attribute = {}
            for outcome in outcomes_per_task[task][policy]:
                if outcome['error']:
                    outcome.update(ERROR_OUTCOME)
                outcome['total_time'] = min(outcome['total_time'], MAX_TIME)
                for attribute, value in outcome.items():
                    if (attribute not in ['policy']) and not isinstance(value, str):
                        value_per_attribute.setdefault(attribute, []).append(value)
            statistics = {attribute: 'mean={:.3f}'.format(np.mean(values)) for attribute, values in value_per_attribute.items()}
            statistics['trials'] = len(outcomes_per_task[task][policy])
            print('{}: {}'.format(name, str_from_object(statistics)))

if __name__ == '__main__':
    main()