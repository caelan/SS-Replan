import datetime
import os
import random

import numpy as np

from pybullet_tools.utils import read_json, link_from_name, get_link_pose
from utils import GRASP_TYPES, get_kitchen_parent

DATABASE_DIRECTORY = os.path.join(os.getcwd(), 'databases/')
IR_FILENAME = '{robot_name}-{surface_name}-{grasp_type}-place.json'


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
    return placements
