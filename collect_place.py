#!/usr/bin/env python2

from __future__ import print_function

import argparse
import os
import random
import sys
import time

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from itertools import product

from pybullet_tools.utils import wait_for_user, elapsed_time, multiply, \
    invert, get_link_pose, has_gui, write_json, get_body_name, get_link_name, \
    RED, BLUE, LockRenderer, child_link_from_joint, get_date, SEPARATOR, dump_body, safe_remove
from src.utils import get_block_path, BLOCK_SIZES, BLOCK_COLORS, GRASP_TYPES, TOP_GRASP, \
    SIDE_GRASP, BASE_JOINTS, joint_from_name, ALL_SURFACES, FRANKA_CARTER, EVE, DRAWERS, \
    OPEN_SURFACES, ENV_SURFACES, CABINETS, ZED_LEFT_SURFACES
from src.world import World
from src.stream import get_stable_gen, get_grasp_gen, Z_EPSILON
from src.streams.pick import get_pick_gen_fn
from src.database import DATABASE_DIRECTORY, PLACE_IR_FILENAME, get_surface_reference_pose, get_place_path

# TODO: condition on the object type (but allow a default object)
# TODO: generalize to any manipulation with a movable entity
# TODO: extend to pouring

def collect_place(world, object_name, surface_name, grasp_type, args):
    date = get_date()
    #set_seed(args.seed)

    #dump_body(world.robot)
    surface_pose = get_surface_reference_pose(world.kitchen, surface_name) # TODO: assumes the drawer is open
    stable_gen_fn = get_stable_gen(world, z_offset=Z_EPSILON, visibility=False,
                                   learned=False, collisions=not args.cfree)
    grasp_gen_fn = get_grasp_gen(world)
    ik_ir_gen = get_pick_gen_fn(world, learned=False, collisions=not args.cfree, teleport=args.teleport)

    stable_gen = stable_gen_fn(object_name, surface_name)
    grasps = list(grasp_gen_fn(object_name, grasp_type))

    robot_name = get_body_name(world.robot)
    path = get_place_path(robot_name, surface_name, grasp_type)
    print(SEPARATOR)
    print('Robot name: {} | Object name: {} | Surface name: {} | Grasp type: {} | Filename: {}'.format(
        robot_name, object_name, surface_name, grasp_type, path))

    entries = []
    start_time = time.time()
    failures = 0
    while (len(entries) < args.num_samples) and \
            (elapsed_time(start_time) < args.max_time): #and (failures <= max_failures):
        (rel_pose,) = next(stable_gen)
        if rel_pose is None:
            break
        (grasp,) = random.choice(grasps)
        with LockRenderer(lock=True):
            result = next(ik_ir_gen(object_name, rel_pose, grasp), None)
        if result is None:
            print('Failure! | {} / {} [{:.3f}]'.format(
                len(entries), args.num_samples, elapsed_time(start_time)))
            failures += 1
            continue
        # TODO: ensure an arm motion exists
        bq, aq, at = result
        rel_pose.assign()
        bq.assign()
        aq.assign()
        base_pose = get_link_pose(world.robot, world.base_link)
        object_pose = rel_pose.get_world_from_body()
        tool_pose = multiply(object_pose, invert(grasp.grasp_pose))
        entries.append({
            'tool_from_base': multiply(invert(tool_pose), base_pose),
            'surface_from_object': multiply(invert(surface_pose), object_pose),
            'base_from_object': multiply(invert(base_pose), object_pose),
        })
        print('Success! | {} / {} [{:.3f}]'.format(
            len(entries), args.num_samples, elapsed_time(start_time)))
        if has_gui():
            wait_for_user()
    #visualize_database(tool_from_base_list)
    if not entries:
        safe_remove(path)
        return None

    # Assuming the kitchen is fixed but the objects might be open world
    data = {
        'date': date,
        'robot_name': robot_name, # get_name | get_body_name | get_base_name | world.robot_name
        'base_link': get_link_name(world.robot, world.base_link),
        'tool_link': get_link_name(world.robot, world.tool_link),
        'kitchen_name': get_body_name(world.kitchen),
        'surface_name': surface_name,
        'object_name': object_name,
        'grasp_type': grasp_type,
        'entries': entries,
        'failures': failures,
        'successes': len(entries),
    }

    write_json(path, data)
    print('Saved', path)
    return data

################################################################################

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-attempts', default=100, type=int,
    #                    help='The number of attempts')
    parser.add_argument('-cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    #parser.add_argument('-grasp_type', default=GRASP_TYPES[0],
    #                    help='Specifies the type of grasp.')
    #parser.add_argument('-problem', default='test_block',
    #                    help='The name of the problem to solve.')
    parser.add_argument('-max_time', default=10*60, type=float,
                        help='The maximum runtime')
    parser.add_argument('-num_samples', default=1000, type=int,
                        help='The number of samples')
    parser.add_argument('-robot', default=FRANKA_CARTER, choices=[FRANKA_CARTER, EVE],
                        help='The robot to use.')
    parser.add_argument('-seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-teleport', action='store_true',
                        help='Uses unit costs')
    parser.add_argument('-visualize', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()

    world = World(use_gui=args.visualize, robot_name=args.robot)
    #dump_body(world.robot)
    for joint in world.kitchen_joints:
        world.open_door(joint) # open_door | close_door
    world.open_gripper()
    # TODO: sample from set of objects?
    object_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    world.add_body(object_name)
    # TODO: could constrain Eve to be within a torso cone

    grasp_colors = {
        TOP_GRASP: RED,
        SIDE_GRASP: BLUE,
    }
    #combinations = list(product(OPEN_SURFACES, GRASP_TYPES)) \
    #               + [(surface_name, TOP_GRASP) for surface_name in DRAWERS] \
    #               + [(surface_name, SIDE_GRASP) for surface_name in CABINETS] # ENV_SURFACES
    combinations = []
    for surface_name in ZED_LEFT_SURFACES:
        if surface_name in (OPEN_SURFACES + DRAWERS):
            combinations.append((surface_name, TOP_GRASP))
        if surface_name in (OPEN_SURFACES + CABINETS):
            combinations.append((surface_name, SIDE_GRASP))

    # TODO: parallelize
    print('Combinations:', combinations)
    wait_for_user('Start?')
    for surface_name, grasp_type in combinations:
        #draw_picks(world, object_name, surface_name, grasp_type, color=grasp_colors[grasp_type])
        collect_place(world, object_name, surface_name, grasp_type, args)
    world.destroy()

if __name__ == '__main__':
    main()

