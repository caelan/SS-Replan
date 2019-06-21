import datetime
import os
import random

import numpy as np

from pybullet_tools.utils import read_json, link_from_name, get_link_pose, multiply, \
    euler_from_quat, draw_point, wait_for_user, set_joint_positions, joints_from_names
from utils import GRASP_TYPES, get_kitchen_parent, BASE_JOINTS

DATABASE_DIRECTORY = os.path.join(os.getcwd(), 'databases/')
IR_FILENAME = '{robot_name}-{surface_name}-{grasp_type}-place.json'

SEPARATOR = '\n' + 50*'-' + '\n'

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

def get_surface_reference_pose(kitchen, surface_name):
    parent_name = get_kitchen_parent(surface_name)
    parent_link = link_from_name(kitchen, parent_name)
    return get_link_pose(kitchen, parent_link)


def load_place_database(robot_name, surface_name, grasp_type, field):
    filename = IR_FILENAME.format(robot_name=robot_name, surface_name=surface_name,
                                  grasp_type=grasp_type)
    path = os.path.join(DATABASE_DIRECTORY, filename)
    if not os.path.exists(path):
        return []
    data = read_json(path)
    return data[field]


def load_placements(world, surface_name, grasp_types=GRASP_TYPES):
    # TODO: could also annotate which grasp came with which placement
    placements = []
    for grasp_type in grasp_types:
        placements.extend(load_place_database(world.robot_name, surface_name, grasp_type,
                                              field='surface_from_object_list'))
    random.shuffle(placements)
    return placements

def load_base_poses(world, tool_pose, surface_name, grasp_type):
    # TODO: should I not actually use surface?
    # TODO: Gaussian perturbation
    gripper_from_base_list = load_place_database(world.robot_name, surface_name, grasp_type,
                                                 field='tool_from_base_list')
    random.shuffle(gripper_from_base_list)
    handles = []
    for gripper_from_base in gripper_from_base_list:
        base_point, base_quat = multiply(tool_pose, gripper_from_base)
        x, y, _ = base_point
        _, _, theta = euler_from_quat(base_quat)
        base_values = (x, y, theta)
        #set_joint_positions(world.robot, joints_from_names(world.robot, BASE_JOINTS), base_values)
        #handles.extend(draw_point(np.array([x, y, -0.1]), color=(1, 0, 0), size=0.05))
        yield base_values
    #wait_for_user()
    #return None
