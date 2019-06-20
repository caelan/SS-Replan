#!/usr/bin/env python2

import argparse
import datetime
import os
import random
import sys
import time
import numpy as np
from itertools import product


PDDLSTREAM_PATH = os.path.abspath(os.path.join(os.getcwd(), 'pddlstream'))
PYBULLET_PATH = os.path.join(PDDLSTREAM_PATH, 'examples/pybullet/utils')
sys.path.extend([PDDLSTREAM_PATH, PYBULLET_PATH])

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import wait_for_user, link_from_name, elapsed_time, multiply, \
    invert, get_link_pose, has_gui, write_json, get_body_name, get_link_name, draw_point, point_from_pose, \
    get_joint_name, joint_from_name
from utils import World, get_block_path, BLOCK_SIZES, BLOCK_COLORS, SURFACES, compute_custom_base_limits, GRASP_TYPES, \
    CARTER_BASE_LINK
from stream import get_pull_gen

from database import DATABASE_DIRECTORY, get_date

PULL_IR_FILENAME = '{joint_name}-pull.json'


################################################################################

def collect_pull(world, joint_name, args):
    date = get_date()
    #set_seed(args.seed)

    for joint in world.kitchen_joints:
        world.close_door(joint)
    world.open_gripper()

    base_link = link_from_name(world.robot, CARTER_BASE_LINK)
    custom_limits = compute_custom_base_limits(world)
    # dump_body(world.robot)

    joint = joint_from_name(world.kitchen, joint_name)
    closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
    open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])

    pull_gen = get_pull_gen(world, custom_limits=custom_limits,
                              collisions=not args.cfree, teleport=args.teleport)
                              #learned=False, max_attempts=args.attempts, max_successes=1, max_failures=0)

    print('Joint name: {}'.format(joint_name))
    tool_from_base_list = []

    start_time = time.time()
    while len(tool_from_base_list) < args.num_samples:
        result = next(pull_gen(joint_name, closed_conf, open_conf), None)
        if result is None:
            continue
        bq, aq, at = result
        bq.assign()
        aq.assign()
        base_pose = get_link_pose(world.robot, base_link)


        tool_from_base = multiply(invert(tool_pose), base_pose)
        tool_from_base_list.append(tool_from_base)

        print('{} / {} [{:.3f}]'.format(
            len(tool_from_base_list), args.num_samples, elapsed_time(start_time)))
        if has_gui():
            wait_for_user()
    #visualize_database(tool_from_base_list)

    # Assuming the kitchen is fixed but the objects might be open world
    # TODO: could store per data point
    data = {
        'date': date,
        'robot_name': get_body_name(world.robot), # get_name | get_body_name | get_base_name | world.robot_name
        'base_link': get_link_name(world.robot, base_link),
        'tool_link': get_link_name(world.robot, world.tool_link),
        'kitchen_name': get_body_name(world.robot),
        'joint_name': joint_name,
        'tool_from_base_list': tool_from_base_list,
    }

    filename = PULL_IR_FILENAME.format(joint_name=joint_name)
    path = os.path.join(DATABASE_DIRECTORY, filename)
    write_json(path, data)
    return data

################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-attempts', default=100, type=int,
                        help='The number of attempts')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-num_samples', default=1000, type=int,
                        help='The number of samples')
    parser.add_argument('-seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-teleport', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()
    # TODO: could record the full trajectories here

    world = World(use_gui=args.visualize)
    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        collect_pull(world, joint_name, args)
    world.destroy()

if __name__ == '__main__':
    main()

