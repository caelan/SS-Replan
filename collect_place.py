#!/usr/bin/env python2

import argparse
import os
import random
import sys
import time

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import wait_for_user, elapsed_time, multiply, \
    invert, get_link_pose, has_gui, write_json, get_body_name, get_link_name, RED, BLUE, LockRenderer, child_link_from_joint
from src.utils import get_block_path, BLOCK_SIZES, BLOCK_COLORS, GRASP_TYPES, TOP_GRASP, \
    SIDE_GRASP, BASE_JOINTS, joint_from_name, ALL_SURFACES
from src.world import World
from src.stream import get_pick_gen_fn, get_stable_gen, get_grasp_gen
from src.database import DATABASE_DIRECTORY, PLACE_IR_FILENAME, get_date, get_surface_reference_pose, SEPARATOR


def collect_place(world, object_name, surface_name, grasp_type, args):
    date = get_date()
    #set_seed(args.seed)

    base_link = child_link_from_joint(joint_from_name(world.robot, BASE_JOINTS[-1]))
    #dump_body(world.robot)
    parent_pose = get_surface_reference_pose(world.kitchen, surface_name)

    stable_gen_fn = get_stable_gen(world, collisions=not args.cfree)
    grasp_gen_fn = get_grasp_gen(world)
    ik_ir_gen = get_pick_gen_fn(world, collisions=not args.cfree, teleport=args.teleport,
                                learned=False, max_attempts=args.attempts,
                                max_successes=1, max_failures=0)

    stable_gen = stable_gen_fn(object_name, surface_name)
    grasps = list(grasp_gen_fn(object_name, grasp_type))

    robot_name = get_body_name(world.robot)
    print(SEPARATOR)
    print('Robot name: {} | Object name: {} | Surface name: {} | Grasp type: {}'.format(
        robot_name, object_name, surface_name, grasp_type))

    tool_from_base_list = []
    surface_from_object_list = []
    start_time = time.time()
    failures = 0
    while (len(tool_from_base_list) < args.num_samples) and \
            (elapsed_time(start_time) < args.max_time): #and (failures <= max_failures):
        (rel_pose,) = next(stable_gen)
        if rel_pose is None:
            break
        (grasp,) = random.choice(grasps)
        with LockRenderer(lock=True):
            result = next(ik_ir_gen(object_name, rel_pose, grasp), None)
        if result is None:
            print('Failure! | {} / {} [{:.3f}]'.format(
                len(tool_from_base_list), args.num_samples, elapsed_time(start_time)))
            failures += 1
            continue
        bq, at = result
        rel_pose.assign()
        bq.assign()
        world.carry_conf.assign()
        base_pose = get_link_pose(world.robot, base_link)
        tool_pose = multiply(rel_pose.get_world_from_body(), invert(grasp.grasp_pose))
        tool_from_base = multiply(invert(tool_pose), base_pose)
        tool_from_base_list.append(tool_from_base)
        surface_from_object = multiply(invert(parent_pose), rel_pose.get_world_from_body())
        surface_from_object_list.append(surface_from_object)
        print('Success! | {} / {} [{:.3f}]'.format(
            len(tool_from_base_list), args.num_samples, elapsed_time(start_time)))
        if has_gui():
            wait_for_user()
    #visualize_database(tool_from_base_list)
    if not tool_from_base_list:
        return None

    # Assuming the kitchen is fixed but the objects might be open world
    # TODO: could store per data point
    data = {
        'date': date,
        'robot_name': robot_name, # get_name | get_body_name | get_base_name | world.robot_name
        'base_link': get_link_name(world.robot, base_link),
        'tool_link': get_link_name(world.robot, world.tool_link),
        'kitchen_name': get_body_name(world.kitchen),
        'surface_name': surface_name,
        'object_name': object_name,
        'grasp_type': grasp_type,
        'tool_from_base_list': tool_from_base_list,
        'surface_from_object_list': surface_from_object_list,
    }

    filename = PLACE_IR_FILENAME.format(robot_name=robot_name, surface_name=surface_name,
                                        grasp_type=grasp_type)
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
    #parser.add_argument('-grasp_type', default=GRASP_TYPES[0],
    #                    help='Specifies the type of grasp.')
    #parser.add_argument('-problem', default='test_block',
    #                    help='The name of the problem to solve.')
    parser.add_argument('-max_time', default=10*60, type=float,
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
    # TODO: open any cabinet doors

    # TODO: sample from set of objects?
    object_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    surface_names = ALL_SURFACES
    #surface_names = SURFACES + CABINET_JOINTS + DRAWER_JOINTS
    #surface_names = CABINET_JOINTS + SURFACES + DRAWER_JOINTS

    world = World(use_gui=args.visualize)
    for joint in world.kitchen_joints:
        world.open_door(joint) # open_door | close_door
    world.open_gripper()
    world.add_body(object_name, get_block_path(object_name))
    # TODO: could constrain eve to be within a torso cone

    grasp_colors = {
        TOP_GRASP: RED,
        SIDE_GRASP: BLUE,
    }
    combinations = [(surface_name, grasp_type) for surface_name in surface_names
                    for grasp_type in GRASP_TYPES]
    print('Combinations:', combinations)
    for surface_name, grasp_type in combinations:
        #draw_picks(world, object_name, surface_name, grasp_type, color=grasp_colors[grasp_type])
        collect_place(world, object_name, surface_name, grasp_type, args)
    world.destroy()

if __name__ == '__main__':
    main()

