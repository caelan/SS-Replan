#!/usr/bin/env python2

from __future__ import print_function

import sys
import argparse
import os
import numpy as np

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, set_random_seed, set_numpy_seed, LockRenderer, \
    get_random_seed, get_numpy_seed, VideoSaver, set_camera, set_camera_pose, get_point, wait_for_duration
from src.command import create_state, iterate_commands, simulate_commands, DEFAULT_TIME_STEP
from src.visualization import add_markers
from src.observe import observe_pybullet
#from src.debug import test_observation
from src.planner import VIDEO_TEMPLATE
from src.world import World
from src.task import TASKS_FNS
from src.policy import run_policy
#from src.debug import dump_link_cross_sections, test_rays

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anytime', action='store_true',
                        help='Runs in an anytime mode')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    #parser.add_argument('-defer', action='store_true',
    #                    help='When enabled, defers evaluation of motion planning streams.')
    parser.add_argument('-deterministic', action='store_true',
                        help='Treats actions as fully deterministic')
    parser.add_argument('-fixed', action='store_true',
                        help="When enabled, fixes the robot_entity's base")
    parser.add_argument('-max_time', default=5*60, type=int,
                        help='The max computation time')
    parser.add_argument('-num', default=1, type=int,
                        help='The number of objects')
    parser.add_argument('-observable', action='store_true',
                        help='Treats the state as fully observable')
    #parser.add_argument('-seed', default=None,
    #                    help='The random seed to use.')
    parser.add_argument('-simulate', action='store_true',
                        help='When enabled, trajectories are simulated')
    parser.add_argument('-teleport', action='store_true',
                        help='When enabled, motion planning is skipped')
    parser.add_argument('-unit', action='store_true',
                        help='When enabled, uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes the planning world '
                             'rather than the simulated world (for debugging).')
    return parser
    # TODO: get rid of funky orientations by dropping them from some height

################################################################################

def main():
    task_names = [fn.__name__ for fn in TASKS_FNS]
    print('Tasks:', task_names)
    parser = create_parser()
    parser.add_argument('-problem', default=task_names[-1], choices=task_names,
                        help='The name of the problem to solve.')
    parser.add_argument('-record', action='store_true',
                        help='When enabled, records and saves a video at {}'.format(
                            VIDEO_TEMPLATE.format('<problem>')))
    args = parser.parse_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    #set_random_seed(0) # Doesn't ensure deterministic
    #set_numpy_seed(1)
    print('Random seed:', get_random_seed())
    print('Numpy seed:', get_numpy_seed())

    np.set_printoptions(precision=3, suppress=True)
    world = World(use_gui=True)
    task_fn_from_name = {fn.__name__: fn for fn in TASKS_FNS}
    task_fn = task_fn_from_name[args.problem]

    task = task_fn(world, num=args.num, fixed=args.fixed)
    wait_for_duration(0.1)
    world._update_initial()
    print('Objects:', task.objects)
    #target_point = get_point(world.get_body(task.objects[0]))
    #set_camera_pose(camera_point=target_point+np.array([-1, 0, 1]), target_point=target_point)

    #if not args.record:
    #    with LockRenderer():
    #        add_markers(task, inverse_place=False)
    #wait_for_user()
    # TODO: FD instantiation is slightly slow to a deepcopy
    # 4650801/25658    2.695    0.000    8.169    0.000 /home/caelan/Programs/srlstream/pddlstream/pddlstream/algorithms/skeleton.py:114(do_evaluate_helper)
    #test_observation(world, entity_name='big_red_block0')
    #return

    # TODO: mechanism that pickles the state of the world
    real_state = create_state(world)
    video = None
    if args.record:
        wait_for_user('Start?')
        video_path = VIDEO_TEMPLATE.format(args.problem)
        video = VideoSaver(video_path)
    time_step = None if args.teleport else DEFAULT_TIME_STEP

    def observation_fn(belief):
        return observe_pybullet(world)

    def transition_fn(belief, commands):
        # if not args.record:  # Video doesn't include planning time
        #    wait_for_user()
        # restore real_state just in case?
        # wait_for_user()
        if args.fixed: # args.simulate
            return simulate_commands(real_state, commands)
        return iterate_commands(real_state, commands, time_step=time_step, pause=False)

    run_policy(task, args, observation_fn, transition_fn)

    if video:
        print('Saved', video_path)
        video.restore()
    world.destroy()
    # TODO: make the sink extrude from the mesh

if __name__ == '__main__':
    main()
