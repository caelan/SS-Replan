#!/usr/bin/env python3

#from __future__ import printfunction, divide

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


from pybullet_tools.utils import wait_for_user, link_from_name, elapsed_time, multiply, \
    invert, get_link_pose, has_gui, write_json, get_body_name, get_link_name, draw_point, \
    point_from_pose, read_json, RED, BLUE, dump_body, LockRenderer
from utils import World, get_block_path, BLOCK_SIZES, BLOCK_COLORS, SURFACES, compute_custom_base_limits, \
    GRASP_TYPES, CABINET_JOINTS, DRAWER_JOINTS, get_kitchen_parent, TOP_GRASP, SIDE_GRASP
from stream import get_pick_gen, get_stable_gen, get_grasp_gen

CARTER_BASE_LINK = 'carter_base_link'

DATABASE_DIRECTORY = os.path.join(os.getcwd(), 'databases/')
IR_FILENAME = '{surface_name}-{grasp_type}-place.json'

def get_random_seed():
    # random.getstate()[1][0]
    return np.random.get_state()[1][0]

def set_seed(seed):
    # These generators are different and independent
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    print('Seed:', seed)

def get_date():
    return datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

################################################################################

def visualize_database(tool_from_base_list):
    handles = []
    if not has_gui():
        return handles
    for gripper_from_base in tool_from_base_list:
        # TODO: move away from the environment
        handles.extend(draw_point(point_from_pose(gripper_from_base), color=RED))
    wait_for_user()
    return handles

def draw_picks(world, surface_name, grasp_type, **kwargs):
    # quantify out

    filename = IR_FILENAME.format(surface_name=surface_name, grasp_type=grasp_type)
    path = os.path.join(DATABASE_DIRECTORY, filename)
    data = read_json(path)
    #tool_from_base_list
    surface_link = link_from_name(world.kitchen, surface_name)
    surface_pose = get_link_pose(world.kitchen, surface_link)
    handles = []
    for surface_from_object in data['surface_from_object_list']:
        object_pose = multiply(surface_pose, surface_from_object)
        handles.extend(draw_point(point_from_pose(object_pose), **kwargs))
    return handles

def collect_place(world, object_name, surface_name, grasp_type, args):
    date = get_date()
    #set_seed(args.seed)

    base_link = link_from_name(world.robot, CARTER_BASE_LINK)
    custom_limits = compute_custom_base_limits(world)
    #dump_body(world.robot)
    parent_name = get_kitchen_parent(surface_name)
    parent_link = link_from_name(world.kitchen, parent_name)
    parent_pose = get_link_pose(world.kitchen, parent_link)

    stable_gen_fn = get_stable_gen(world, collisions=not args.cfree)
    grasp_gen_fn = get_grasp_gen(world, grasp_types=[grasp_type])
    ik_ir_gen = get_pick_gen(world, custom_limits=custom_limits,
                             collisions=not args.cfree, teleport=args.teleport,
                             learned=False, max_attempts=args.attempts,
                             max_successes=1, max_failures=0)

    stable_gen = stable_gen_fn(object_name, surface_name)
    grasps = list(grasp_gen_fn(object_name))
    tool_from_base_list = []
    surface_from_object_list = []

    print('\n' + 50*'-' + '\n')
    print('Object name: {} | Surface name: {} | Grasp type: {}'.format
          (object_name, surface_name, grasp_type))
    start_time = time.time()
    failures = 0
    while len(tool_from_base_list) < args.num_samples:
        (pose,) = next(stable_gen)
        if pose is None:
            break
        (grasp,) = random.choice(grasps)
        with LockRenderer():
            result = next(ik_ir_gen(object_name, pose, grasp), None)
        if result is None:
            print('Failure! | {} / {} [{:.3f}]'.format(
                len(tool_from_base_list), args.num_samples, elapsed_time(start_time)))
            failures += 1
            continue
        bq, aq, at = result
        pose.assign()
        bq.assign()
        aq.assign()
        base_pose = get_link_pose(world.robot, base_link)
        tool_pose = multiply(pose.value, invert(grasp.grasp_pose))
        tool_from_base = multiply(invert(tool_pose), base_pose)
        tool_from_base_list.append(tool_from_base)
        surface_from_object = multiply(invert(parent_pose), pose.value)
        surface_from_object_list.append(surface_from_object)
        print('Success! | {} / {} [{:.3f}]'.format(
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
        'surface_name': surface_name,
        'object_name': object_name,
        'grasp_type': grasp_type,
        'tool_from_base_list': tool_from_base_list,
        'surface_from_object_list': surface_from_object_list,
    }

    filename = IR_FILENAME.format(surface_name=surface_name, grasp_type=grasp_type)
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
    #parser.add_argument('-grasp_type', default=GRASP_TYPES[0],
    #                    help='Specifies the type of grasp.')
    #parser.add_argument('-problem', default='test_block',
    #                    help='The name of the problem to solve.')
    parser.add_argument('-num_samples', default=1000, type=int,
                        help='The number of samples')
    parser.add_argument('-seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-teleport', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()
    # TODO: open any cabinet doors

    # TODO: sample from set of objects?
    object_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    surface_names = SURFACES
    surface_names = DRAWER_JOINTS + CABINET_JOINTS

    world = World(use_gui=args.visualize)
    for joint in world.kitchen_joints:
        world.open_door(joint) # open_door | close_door
    world.open_gripper()
    world.add_body(object_name, get_block_path(object_name))

    grasp_colors = {
        TOP_GRASP: RED,
        SIDE_GRASP: BLUE,
    }
    for surface_name in surface_names:
        if surface_name in CABINET_JOINTS:
            grasp_types = [SIDE_GRASP]
        elif surface_name in DRAWER_JOINTS:
            grasp_types = [TOP_GRASP]
        else:
            grasp_types = GRASP_TYPES
        for grasp_type in grasp_types:
            #draw_picks(world, surface_name, grasp_type, color=grasp_colors[grasp_type])
            collect_place(world, object_name, surface_name, grasp_type, args)
    wait_for_user()
    world.destroy()

if __name__ == '__main__':
    main()

