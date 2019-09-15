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
import resource
import copy

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

import pddlstream.language.statistics
pddlstream.language.statistics.LOAD_STATISTICS = False
pddlstream.language.statistics.SAVE_STATISTICS = False

from pybullet_tools.utils import has_gui, elapsed_time, user_input, ensure_dir, \
    write_json, SEPARATOR, WorldSaver, \
    get_random_seed, get_numpy_seed, set_random_seed, set_numpy_seed, wait_for_user
from pddlstream.utils import str_from_object, safe_rm_dir, Verbose
from pddlstream.algorithms.algorithm import reset_globals

from src.command import create_state, iterate_commands
from src.observe import observe_pybullet
from src.world import World
from src.policy import run_policy
from src.task import cook_block, TASKS_FNS
from run_pybullet import create_parser

from multiprocessing import Pool, TimeoutError, cpu_count

EXPERIMENTS_DIRECTORY = 'experiments/'
TEMP_DIRECTORY = 'temp_parallel/'
MAX_TIME = 10*60
VERBOSE = False
SERIAL = False
SERIALIZE_TASK = True

TIME_PER_TRIAL = 150 # trial / sec
HOURS_TO_SECS = 60 * 60

N = 50
#MAX_RAM = 28 # Max of 31.1 Gigabytes
#BYTES_PER_KILOBYTE = math.pow(2, 10)
#BYTES_PER_GIGABYTE = math.pow(2, 30)

POLICIES = [
    #{'constrain': False, 'defer': False},
    {'constrain': True, 'defer': False},
    {'constrain': False, 'defer': True}, # Move actions grow immensely
    {'constrain': True, 'defer': True},
]

# Is it because the objects are large files?
# Switch to psutil

TASK_NAMES = [
    #'detect_block',
    #'hold_block',
    'swap_drawers',
    #'sugar_drawer',
    #'cook_block',
    #'cook_meal',
    #'stow_block',
]

# TODO: CPU usage at 300% due to TracIK or the visualizer?
# TODO: could check collisions only with real (non-observed) values

################################################################################

def map_parallel(fn, inputs, num_cores=None, timeout=None):
    # Processes rather than threads (shared memory)
    # TODO: with statement on Pool
    if SERIAL:
        for outputs in map(fn, inputs):
            yield outputs
        return
    pool = Pool(processes=num_cores) #, initializer=mute)
    generator = pool.imap_unordered(fn, inputs, chunksize=1)
    # pool_result = pool.map_async(worker, args)
    #return generator
    while True:
        # TODO: need to actually retrieve the info about which thread failed
        try:
            yield generator.next(timeout=timeout)
        except StopIteration:
            break
        except MemoryError: # as e:
            traceback.print_exc()
            continue
        except TimeoutError: # as e:
            traceback.print_exc()
            continue
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

def name_from_policy(policy):
    return '_'.join('{}={:d}'.format(key, value) for key, value in sorted(policy.items()))

def run_experiment(experiment):
    problem = experiment['problem']
    task_name = problem['task'].name if SERIALIZE_TASK else problem['task']
    trial = problem['trial']
    policy = experiment['policy']

    stdout = sys.stdout
    if not VERBOSE:
       sys.stdout = open(os.devnull, 'w')
    current_wd = os.getcwd()
    #trial_wd = os.path.join(current_wd, TEMP_DIRECTORY, '{}/'.format(os.getpid()))
    if not SERIAL:
        trial_wd = os.path.join(current_wd, TEMP_DIRECTORY, 't={}_n={}_{}/'.format(
            task_name, trial, name_from_policy(policy)))
        safe_rm_dir(trial_wd)
        ensure_dir(trial_wd)
        os.chdir(trial_wd)

    parser = create_parser()
    args = parser.parse_args()

    task_fn_from_name = {fn.__name__: fn for fn in TASKS_FNS}
    task_fn = task_fn_from_name[task_name]
    world = World(use_gui=SERIAL)
    if SERIALIZE_TASK:
        task_fn(world, fixed=args.fixed)
        task = problem['task']
        task.world = world
    else:
        # TODO: assumes task_fn is deterministic wrt task
        task_fn(world, fixed=args.fixed)
    problem['saver'].restore()
    world._update_initial()
    problem['task'] = task_name # for serialization
    del problem['saver']

    random.seed(hash((0, task_name, trial, time.time())))
    numpy.random.seed(hash((1, task_name, trial, time.time())) % (2**32))
    #seed1, seed2 = problem['seeds'] # No point unless you maintain the same random state per generator
    #set_random_seed(seed1)
    #set_random_seed(seed2)
    #random.setstate(state1)
    #numpy.random.set_state(state2)
    reset_globals()
    real_state = create_state(world)
    #start_time = time.time()
    if has_gui():
        wait_for_user()
    try:
        observation_fn = lambda belief: observe_pybullet(world)
        transition_fn = lambda belief, commands: iterate_commands(real_state, commands, time_step=0)
        outcome = run_policy(task, args, observation_fn, transition_fn,
                          max_time=MAX_TIME, **policy)
        outcome['error'] = False
    except KeyboardInterrupt:
        raise KeyboardInterrupt()
    except:
        traceback.print_exc()
        outcome = {'error': True}

    world.destroy()
    os.chdir(current_wd)
    if not SERIAL:
        safe_rm_dir(trial_wd)
    if not VERBOSE:
        sys.stdout.close()
    sys.stdout = stdout

    result = {
        'experiment': experiment,
        'outcome': outcome,
    }
    return result

def create_problems(args):
    task_fn_from_name = {fn.__name__: fn for fn in TASKS_FNS}
    problems = []
    for trial in range(N):
        print('\nTrial: {} / {}'.format(trial, N))
        for task_name in TASK_NAMES:
            random.seed(hash((0, task_name, trial, time.time())))
            numpy.random.seed(hash((1, task_name, trial, time.time())) % (2 ** 32))
            world = World(use_gui=SERIAL)
            task_fn = task_fn_from_name[task_name]
            task = task_fn(world, fixed=args.fixed)
            task.world = None
            if not SERIALIZE_TASK:
                task = task_name
            saver = WorldSaver()
            problems.append({
                'task': task,
                'trial': trial,
                'saver': saver,
                #'seeds': [get_random_seed(), get_numpy_seed()],
                #'seeds': [random.getstate(), numpy.random.get_state()],
            })
            #print(world.body_from_name) # TODO: does not remain the same
            #wait_for_user()
            #world.reset()
            #if has_gui():
            #    wait_for_user()
            world.destroy()
    return problems

################################################################################

#def set_soft_limit(name, limit):
#    # TODO: use FastDownward's memory strategy
#    # ulimit -a
#    soft, hard = resource.getrlimit(name) # resource.RLIM_INFINITY
#    soft = limit
#    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

def main():
    parser = create_parser()
    #parser.add_argument('-problem', default=task_names[-1], choices=task_names,
    #                   help='The name of the problem to solve.')
    #parser.add_argument('-record', action='store_true',
    #                   help='When enabled, records and saves a video at {}'.format(
    #                       VIDEO_TEMPLATE.format('<problem>')))
    args = parser.parse_args()

    # https://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
    # https://stackoverflow.com/questions/39884898/large-amount-of-multiprocessing-process-causing-deadlock
    # TODO: alternatively don't destroy the world
    num_cores = cpu_count() - 2
    directory = EXPERIMENTS_DIRECTORY
    date_name = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    json_path = os.path.abspath(os.path.join(directory, '{}.json'.format(date_name)))

    #memory_per_core = float(MAX_RAM) / num_cores # gigabytes
    #set_soft_limit(resource.RLIMIT_AS, int(BYTES_PER_GIGABYTE * memory_per_core)) # bytes
    #set_soft_limit(resource.RLIMIT_CPU, 2*MAX_TIME) # seconds
    # RLIMIT_MEMLOCK, RLIMIT_STACK, RLIMIT_DATA

    print('Results:', json_path)
    print('Num Cores:', num_cores)
    #print('Memory per Core: {:.2f}'.format(memory_per_core))
    print('Tasks: {} | {}'.format(len(TASK_NAMES), TASK_NAMES))
    print('Policies: {} | {}'.format(len(POLICIES), POLICIES))
    print('Num Trials:', N)
    num_experiments = len(TASK_NAMES)*len(POLICIES)*N
    print('Num Experiments:', num_experiments)
    max_parallel = math.ceil(float(num_experiments) / num_cores)
    print('Estimated duration: {:.2f} hours'.format(TIME_PER_TRIAL*max_parallel / HOURS_TO_SECS))
    user_input('Begin?')

    problem = create_problems(args)
    experiments = [{'problem': copy.deepcopy(task), 'policy': policy} #, 'args': args}
                   for task in problem for policy in POLICIES]

    ensure_dir(directory)
    safe_rm_dir(TEMP_DIRECTORY)
    ensure_dir(TEMP_DIRECTORY)
    start_time = time.time()
    results = []
    try:
        for result in map_parallel(run_experiment, experiments, num_cores=num_cores, timeout=2*MAX_TIME):
            results.append(result)
            print('{}\nExperiments: {} / {} | Time: {:.3f}'.format(
                SEPARATOR, len(results), len(experiments), elapsed_time(start_time)))
            print('Experiment:', str_from_object(result['experiment']))
            print('Outcome:', str_from_object(result['outcome']))
            write_json(json_path, results)
    #except BaseException as e:
    #    traceback.print_exc() # e
    finally:
        if results:
            write_json(json_path, results)
        print(SEPARATOR)
        print('Saved:', json_path)
        print('Results:', len(results))
        print('Duration / experiment: {:.3f}'.format(num_cores*len(experiments) / elapsed_time(start_time)))
        print('Duration: {:.2f} hours'.format(elapsed_time(start_time) / HOURS_TO_SECS))
        safe_rm_dir(TEMP_DIRECTORY)
    return results


if __name__ == '__main__':
    main()
