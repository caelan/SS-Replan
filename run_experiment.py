#!/usr/bin/env python2

from __future__ import print_function

# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/learn_tools/collect_simulation.py
# https://github.mit.edu/caelan/pddlstream-experiments/blob/master/run_experiment.py

import argparse
import numpy as np
import time
import datetime
import math
import numpy
import random
import os
import sys
import traceback

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

import pddlstream.language.statistics
pddlstream.language.statistics.LOAD_STATISTICS = False
pddlstream.language.statistics.SAVE_STATISTICS = False

from pybullet_tools.utils import elapsed_time, user_input, ensure_dir, \
    write_json, SEPARATOR, WorldSaver
from pddlstream.utils import str_from_object, safe_rm_dir
from pddlstream.algorithms.algorithm import reset_globals

from src.command import create_state, iterate_commands
from src.observe import observe_pybullet
from src.world import World
from src.policy import run_policy
from src.task import cook_block, TASKS_FNS
from run_pybullet import create_parser

from multiprocessing import Pool, TimeoutError, cpu_count

DATA_DIRECTORY = 'data/'
TEMP_DIRECTORY = 'temp_parallel/'
MAX_TIME = 5*60
VERBOSE = False

N = 100

POLICIES = [
    #{'constrain': False, 'defer': False},
    {'constrain': True, 'defer': False},
    {'constrain': False, 'defer': True},
    {'constrain': True, 'defer': True},
]

TASK_NAMES = [
    'detect_block',
    'hold_block',
    'detect_drawers',
    'sugar_drawer',
    'cook_block',
    'cook_meal',
    'stow_block',
]

################################################################################

def map_parallel(fn, inputs, num_cores=None, timeout=None):
    # Processes rather than threads (shared memory)
    # TODO: with statement on Pool
    pool = Pool(processes=num_cores) #, initializer=mute)
    generator = pool.imap_unordered(fn, inputs, chunksize=1)
    # pool_result = pool.map_async(worker, args)
    #return generator
    while True:
        try:
            yield generator.next(timeout=timeout)
        except StopIteration:
            break
        except TimeoutError:
            print('Error! Timed out after {:.3f} seconds'.format(timeout))
            break
    if pool is not None:
        pool.close()
        pool.terminate()
        pool.join()
    #import psutil
    #if parallel:
    #    process = psutil.Process(os.getpid())
    #    print(process)
    #    print(process.get_memory_info())

################################################################################

def run_experiment(experiment):
    pid = os.getpid()
    if not VERBOSE:
       sys.stdout = open(os.devnull, 'w')
    current_wd = os.getcwd()
    trial_wd = os.path.join(current_wd, TEMP_DIRECTORY, '{}/'.format(pid))
    safe_rm_dir(trial_wd)
    ensure_dir(trial_wd)
    os.chdir(trial_wd)

    random.seed(hash((0, pid, time.time())))
    numpy.random.seed(hash((1, pid, time.time())) % (2**32))

    parser = create_parser()
    args = parser.parse_args()

    task_fn_from_name = {fn.__name__: fn for fn in TASKS_FNS}
    task_fn = task_fn_from_name[experiment['task']]
    world = World(use_gui=False)
    task = task_fn(world, fixed=args.fixed)
    world._update_initial()
    saver = WorldSaver()

    outcomes = []
    state1 = random.getstate()
    state2 = numpy.random.get_state()
    for policy in POLICIES:
        random.setstate(state1)
        numpy.random.set_state(state2)
        saver.restore()
        reset_globals()
        real_state = create_state(world)
        #start_time = time.time()
        try:
            observation_fn = lambda belief: observe_pybullet(world)
            transition_fn = lambda belief, commands: iterate_commands(real_state, commands, time_step=0)
            data = run_policy(task, args, observation_fn, transition_fn,
                              max_time=MAX_TIME, max_planning_time=30, **policy)
            data['error'] = False
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            traceback.print_exc()
            data = {'error': True}
        data['policy'] = policy
        outcomes.append(data)

    world.destroy()
    os.chdir(current_wd)
    safe_rm_dir(trial_wd)
    if not VERBOSE:
        sys.stdout.close()

    result = {
        'experiment': experiment,
        'outcomes': outcomes,
    }
    return result

################################################################################

def main():
    #parser = create_parser()
    #parser.add_argument('-problem', default=task_names[-1], choices=task_names,
    #                    help='The name of the problem to solve.')
    #parser.add_argument('-record', action='store_true',
    #                    help='When enabled, records and saves a video at {}'.format(
    #                        VIDEO_TEMPLATE.format('<problem>')))
    #args = parser.parse_args()

    # https://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
    # https://stackoverflow.com/questions/39884898/large-amount-of-multiprocessing-process-causing-deadlock
    num_cores = cpu_count()
    directory = os.path.realpath(DATA_DIRECTORY)
    date_name = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    json_path = os.path.join(directory, date_name)
    experiments = [{'task': task, 'trial': trial} for task in TASK_NAMES for trial in range(N)]


    print('Results:', json_path)
    print('Num Cores:', num_cores)
    print('Tasks: {} | {}'.format(len(TASK_NAMES), TASK_NAMES))
    print('Policies: {} | {}'.format(len(POLICIES), POLICIES))
    print('Num Trials:', N)
    print('Num Experiments:', len(experiments))

    #max_parallel = math.ceil(float(len(trials)) / num_cores)
    #time_per_trial = (MAX_TIME * len(ALGORITHMS)) / HOURS_TO_SECS
    #max_hours = max_parallel * time_per_trial
    #print('Max hours:', max_hours)
    user_input('Begin?')

    ensure_dir(directory)
    ensure_dir(TEMP_DIRECTORY)
    start_time = time.time()
    results = []
    try:
        for result in map_parallel(run_experiment, experiments, timeout=2*MAX_TIME):
            results.append(result)
            print('{}\nExperiments: {} / {} | Time: {:.3f}'.format(
                SEPARATOR, len(results), len(experiments), elapsed_time(start_time)))
            print('Experiment:', str_from_object(result['experiment']))
            print('Outcomes:', str_from_object(result['outcomes']))
            write_json(json_path, results)
    #except BaseException as e:
    #    traceback.print_exc() # e
    finally:
        safe_rm_dir(TEMP_DIRECTORY)
        if results:
            write_json(json_path, results)
        print(SEPARATOR)
        print('Results:', len(results))
        print('Hours: {:.3f}'.format(elapsed_time(start_time) / (60*60)))
    return results


if __name__ == '__main__':
    main()