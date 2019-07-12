import os
import random

from pybullet_tools.utils import read_json, link_from_name, get_link_pose, multiply, \
    euler_from_quat, draw_point, wait_for_user, set_joint_positions, joints_from_names, parent_link_from_joint, has_gui, \
    point_from_pose, RED, child_link_from_joint, get_pose, get_point, invert, base_values_from_pose
from src.utils import GRASP_TYPES, get_surface, BASE_JOINTS, joint_from_name, unit_pose

DATABASE_DIRECTORY = os.path.join(os.getcwd(), 'databases/')
PLACE_IR_FILENAME = '{robot_name}-{surface_name}-{grasp_type}-place.json'
PULL_IR_FILENAME = '{robot_name}-{joint_name}-pull.json'

# TODO: which frame should the place motion be in?
# Do I trust the robot base or the kitchen for the floor plane?


def get_surface_reference_pose(kitchen, surface_name):
    surface = get_surface(surface_name)
    link = link_from_name(kitchen, surface.link)
    return get_link_pose(kitchen, link)

def has_place_database(robot_name, surface_name, grasp_type):
    path = os.path.join(DATABASE_DIRECTORY, PLACE_IR_FILENAME.format(
        robot_name=robot_name, surface_name=surface_name, grasp_type=grasp_type))
    return os.path.exists(path)

def load_place_database(robot_name, surface_name, grasp_type, field):
    if not has_place_database(robot_name, surface_name, grasp_type):
        return []
    path = os.path.join(DATABASE_DIRECTORY, PLACE_IR_FILENAME.format(
        robot_name=robot_name, surface_name=surface_name, grasp_type=grasp_type))
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

def project_base_pose(base_pose):
    #return base_values_from_pose(base_pose)
    base_point, base_quat = base_pose
    x, y, _ = base_point
    _, _, theta = euler_from_quat(base_quat)
    base_values = (x, y, theta)
    return base_values

def load_place_base_poses(world, tool_pose, surface_name, grasp_type):
    # TODO: Gaussian perturbation
    gripper_from_base_list = load_place_database(world.robot_name, surface_name, grasp_type,
                                                 field='tool_from_base_list')
    random.shuffle(gripper_from_base_list)
    handles = []
    for gripper_from_base in gripper_from_base_list:
        #world_from_model = get_pose(world.robot)
        world_from_model = unit_pose()
        base_values = project_base_pose(multiply(invert(world_from_model), tool_pose, gripper_from_base))
        #x, y, _ = base_values
        #_, _, z = get_point(world.floor)
        #set_joint_positions(world.robot, joints_from_names(world.robot, BASE_JOINTS), base_values)
        #handles.extend(draw_point(np.array([x, y, z + 0.01]), color=(1, 0, 0), size=0.05))
        #wait_for_user()
        yield base_values

################################################################################

def get_joint_reference_pose(kitchen, surface_name):
    joint = joint_from_name(kitchen, surface_name)
    link = parent_link_from_joint(kitchen, joint)
    return get_link_pose(kitchen, link)

def load_pull_database(robot_name, joint_name):
    filename = PULL_IR_FILENAME.format(robot_name=robot_name, joint_name=joint_name)
    path = os.path.join(DATABASE_DIRECTORY, filename)
    if not os.path.exists(path):
        return []
    data = read_json(path)
    return data['joint_from_base_list']

def load_pull_base_poses(world, joint_name):
    joint_from_base_list = load_pull_database(world.robot_name, joint_name)
    parent_pose = get_joint_reference_pose(world.kitchen, joint_name)
    random.shuffle(joint_from_base_list)
    handles = []
    for joint_from_base in joint_from_base_list:
        #world_from_model = get_pose(world.robot)
        world_from_model = unit_pose()
        base_values = project_base_pose(multiply(invert(world_from_model), parent_pose, joint_from_base))
        #set_joint_positions(world.robot, joints_from_names(world.robot, BASE_JOINTS), base_values)
        #x, y, _ = base_values
        #handles.extend(draw_point(np.array([x, y, -0.1]), color=(1, 0, 0), size=0.05))
        yield base_values
    #wait_for_user()

################################################################################

def visualize_database(tool_from_base_list):
    #tool_from_base_list
    handles = []
    if not has_gui():
        return handles
    for gripper_from_base in tool_from_base_list:
        # TODO: move away from the environment
        handles.extend(draw_point(point_from_pose(gripper_from_base), color=RED))
    wait_for_user()
    return handles
