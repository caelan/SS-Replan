#!/usr/bin/env python2

import argparse
import os
import sys
import time

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import wait_for_user, elapsed_time, multiply, \
    invert, get_link_pose, has_gui, write_json, get_body_name, get_link_name, \
    get_joint_name, joint_from_name, get_date, SEPARATOR
from src.world import World
from src.stream import get_pull_gen_fn
from src.database import DATABASE_DIRECTORY, get_joint_reference_pose, PULL_IR_FILENAME


def collect_pull(world, joint_name, args):
    date = get_date()
    #set_seed(args.seed)

    parent_pose = get_joint_reference_pose(world.kitchen, joint_name)

    joint = joint_from_name(world.kitchen, joint_name)
    closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
    open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])

    pull_gen = get_pull_gen_fn(world, collisions=not args.cfree, teleport=args.teleport, learned=False)
                            #learned=False, max_attempts=args.attempts, max_successes=1, max_failures=0)
    #handle_link, handle_grasp, _ = get_handle_grasp(world, joint)

    robot_name = get_body_name(world.robot)
    print(SEPARATOR)
    print('Robot name {} | Joint name: {}'.format(robot_name, joint_name))

    joint_from_base_list = []
    failures = 0
    start_time = time.time()
    while (len(joint_from_base_list) < args.num_samples) and \
            (elapsed_time(start_time) < args.max_time):
        result = next(pull_gen(joint_name, open_conf, closed_conf), None)
        if result is None:
            print('Failure! | {} / {} [{:.3f}]'.format(
                len(joint_from_base_list), args.num_samples, elapsed_time(start_time)))
            failures += 1
            continue
        open_conf.assign()
        bq, aq, at = result
        bq.assign()
        aq.assign()
        #next(at.commands[2].iterate(None, None))
        base_pose = get_link_pose(world.robot, world.base_link)
        #handle_pose = get_link_pose(world.robot, base_link)
        joint_from_base = multiply(invert(parent_pose), base_pose)
        joint_from_base_list.append(joint_from_base)
        print('Success! | {} / {} [{:.3f}]'.format(
            len(joint_from_base_list), args.num_samples, elapsed_time(start_time)))
        if has_gui():
            wait_for_user()
    if not joint_from_base_list:
        return None
    #visualize_database(joint_from_base_list)

    # Assuming the kitchen is fixed but the objects might be open world
    # TODO: could store per data point
    data = {
        'date': date,
        'robot_name': robot_name, # get_name | get_body_name | get_base_name | world.robot_name
        'base_link': get_link_name(world.robot, world.base_link),
        'tool_link': get_link_name(world.robot, world.tool_link),
        'kitchen_name': get_body_name(world.kitchen),
        'joint_name': joint_name,
        'joint_from_base_list': joint_from_base_list,
    }

    filename = PULL_IR_FILENAME.format(robot_name=robot_name, joint_name=joint_name)
    path = os.path.join(DATABASE_DIRECTORY, filename)
    write_json(path, data)
    print('Saved', path)
    return data

################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-attempts', default=100, type=int,
                        help='The number of attempts')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-max_time', default=10 * 60, type=float,
                        help='The maximum runtime')
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
        world.close_door(joint)
    world.open_gripper()

    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        collect_pull(world, joint_name, args)
    world.destroy()

if __name__ == '__main__':
    main()

