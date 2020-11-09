from __future__ import print_function

import os
import numpy as np
import string
import math
import random

from itertools import cycle
from collections import namedtuple

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.pr2_utils import get_top_grasps, get_side_grasps, close_until_collision
from pybullet_tools.utils import joints_from_names, joint_from_name, Attachment, link_from_name, get_unit_vector, \
    unit_pose, BodySaver, multiply, Pose, \
    get_link_subtree, clone_body, get_all_links, invert, get_link_pose, set_pose, interpolate_poses, set_color, LockRenderer, get_body_name, randomize, unit_point, create_obj, BASE_LINK, get_link_descendants, \
    get_aabb, get_collision_data, point_from_pose, get_data_pose, get_data_extents, AABB, \
    apply_affine, get_aabb_vertices, aabb_from_points, read_obj, tform_mesh, create_attachment, draw_point, \
    child_link_from_joint, is_placed_on_aabb, pairwise_collision, flatten_links, has_link, get_difference_fn, Euler, approximate_as_prism, \
    get_joint_positions, implies, unit_from_theta

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'models/')

VISUAL = True
if VISUAL:
    FRANKA_CARTER_PATH = os.path.join(MODELS_PATH, 'panda_arm_hand_on_carter_visual.urdf')
else:
    FRANKA_CARTER_PATH = os.path.join(MODELS_PATH, 'panda_arm_hand_on_carter_collision.urdf')

DEBUG = True

BASE_JOINTS = ['x', 'y', 'theta']
WHEEL_JOINTS = ['left_wheel', 'right_wheel']

FRANKA_CARTER = 'franka_carter'
FRANKA_TOOL_LINK = 'right_gripper'  # right_gripper | panda_wrist_end_pt | panda_forearm_end_pt
TOOL_POSE = unit_pose()

# +z: pointing, +y: left finger
FINGER_EXTENT = np.array([0.02, 0.01, 0.02]) # 2cm x 1cm x 2cm
FRANKA_GRIPPER_LINK = 'panda_link7' # panda_link7 | panda_link8 | panda_hand

################################################################################

TOP_GRASP = 'top'
SIDE_GRASP = 'side' # TODO: allow normal side grasps for cabinets?
UNDER_GRASP = 'under' # TODO: for franka_carter
GRASP_TYPES = [
    TOP_GRASP,
    SIDE_GRASP,
]
APPROACH_DISTANCE = 0.075 # 0.075 | 0.1

################################################################################

NAME_TEMPLATE = '{}{}'
BLOCK_TEMPLATE = '{}_{}_block'

BLOCK_SIZES = ['small', 'big']
BLOCK_COLORS = ['red', 'green', 'blue', 'yellow']
YCB_PATH = os.path.join(MODELS_PATH, 'ycb/')
# TODO: ycb obj files have 6 vertex coordinates?

CHEEZIT = 'cracker_box'
SUGAR = 'sugar_box'
TOMATO_SOUP = 'tomato_soup_can'
PUDDING = 'pudding_box'
JELLO = 'gelatin_box'
SPAM = 'potted_meat_can'
MUSTARD = 'mustard_bottle'
BOWL = 'bowl'
#BANANA = 'banana'
YCB_OBJECTS = [SPAM, MUSTARD, TOMATO_SOUP, SUGAR, CHEEZIT] # + [BOWL, PUDDING, BANANA]
BLOCK = 'block'

ECHO_COUNTER = 'echo'
INDIGO_COUNTER = 'indigo_tmp'
TOP_DRAWER = 'indigo_drawer_top'
BOTTOM_DRAWER = 'indigo_drawer_bottom'
LEFT_DOOR = 'dagger_door_left'

BOWLS = [BOWL]
POURABLE = [TOMATO_SOUP, MUSTARD] # SUGAR
COOKABLE = [BOWL]

################################################################################

#KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen_stl.urdf')
#KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen_obj.urdf')
KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen_convex.urdf')

SURFACE_BOTTOM = 'bottom'
SURFACE_TOP = 'top'

LEFT_CABINETS = ['baker', 'chewie_door_left', 'chewie_door_right']
RIGHT_CABINETS = ['dagger_door_left'] #, 'dagger_door_right', 'indigo_tmp_bottom']
CABINETS = LEFT_CABINETS + RIGHT_CABINETS

DRAWERS = [
    #'hitman_drawer_top', #'hitman_drawer_bottom',
    'indigo_drawer_top', 'indigo_drawer_bottom',
]

STOVE_LOCATIONS = [
    #'back_left', 'back_right',
    #'front_left',
    'front_right',
]

STOVE_TEMPLATE = '{}_stove'
STOVES = [STOVE_TEMPLATE.format(location) for location in STOVE_LOCATIONS]

KNOB_TEMPLATE = '{}_knob'
KNOBS = [KNOB_TEMPLATE.format(location) for location in STOVE_LOCATIONS]

ENV_SURFACES = [
    'echo',  # fox is covered by echo
    'golf',
    #'range',
    'table',
]

COUNTERS = ['hitman_tmp', 'indigo_tmp', 'range']
OPEN_SURFACES = COUNTERS + STOVES

Surface = namedtuple('Surface', ['link', 'shape', 'joints'])

SURFACE_FROM_NAME = {
    # TODO: could infer joints from the name
    'baker': Surface('sektion', 'Cube.bottom.004_Cube.028', ['baker_joint']),
    'chewie_door_left': Surface('sektion', 'Cube.bottom.002_Cube.020', ['chewie_door_left_joint']),
    'chewie_door_right': Surface('sektion', 'Cube.bottom_Cube.000', ['chewie_door_right_joint']),

    'dagger_door_left': Surface('dagger', 'Cube.bottom.008_Cube.044', ['dagger_door_left_joint']),
    'dagger_door_right': Surface('dagger', 'Cube.bottom.012_Cube.060', ['dagger_door_right_joint']),

    'hitman_drawer_top': Surface('hitman_drawer_top', 'Cube_Cube.001', ['hitman_drawer_top_joint']),
    'hitman_drawer_bottom': Surface('hitman_drawer_bottom', 'Cube_Cube.001', ['hitman_drawer_bottom_joint']),

    #'indigo_tmp_bottom': Surface('indigo_tmp', SURFACE_BOTTOM, ['indigo_door_left_joint', 'indigo_door_right_joint']),
    #'indigo_drawer_top': Surface('indigo_drawer_top', 'Cube_Cube.001', ['indigo_drawer_top_joint']),
    #'indigo_drawer_bottom': Surface('indigo_drawer_bottom', 'Cube_Cube.001', ['indigo_drawer_bottom_joint']),
    'indigo_drawer_top': Surface('indigo_drawer_top', SURFACE_BOTTOM, ['indigo_drawer_top_joint']),
    'indigo_drawer_bottom': Surface('indigo_drawer_bottom', SURFACE_BOTTOM, ['indigo_drawer_bottom_joint']),
}

ZED_LEFT_SURFACES = [
    'dagger_door_left',
    'indigo_tmp',
    #'range',
    'indigo_drawer_top', 'indigo_drawer_bottom',
] + STOVES

ALL_SURFACES = ZED_LEFT_SURFACES
#ALL_SURFACES = [] + OPEN_SURFACES + DRAWERS + CABINETS

KITCHEN_LEFT_PATH = os.path.join(MODELS_PATH, 'kitchen_left')

################################################################################

CAMERA_TEMPLATE = 'zed_{}'
LEFT_CAMERA = CAMERA_TEMPLATE.format('left')
RIGHT_CAMERA = CAMERA_TEMPLATE.format('right')
CAMERAS = [LEFT_CAMERA, RIGHT_CAMERA]

KINECT_DEPTH = 5.0
CAMERA_MATRIX = np.array(
    [[ 532.569,    0.,     320.,   ],
     [   0.,     532.569,  240.,   ],
     [   0.,       0.,       1.,   ]])

KITCHEN_FROM_ZED_LEFT = (
    (1.0600011348724365, 1.529999017715454, 0.5699998736381531),
    (-0.10374931246042252, 0.9274755120277405, -0.19101102650165558, -0.30420398712158203))
KITCHEN_FROM_ZED_RIGHT = (
    (1.0300002098083496, -1.1000001430511475, 0.3399999439716339),
    (0.7792050242424011, 0.45354312658309937, -0.20711229741573334, -0.3797929286956787))
CAMERA_POSES = {
    LEFT_CAMERA: KITCHEN_FROM_ZED_LEFT,
    RIGHT_CAMERA: KITCHEN_FROM_ZED_RIGHT,
}

# https://gitlab-master.nvidia.com/search?utf8=%E2%9C%93&snippets=&scope=&search=zed_right&project_id=7659

################################################################################

JOINT_TEMPLATE = '{}_joint'

CABINET_JOINTS = [JOINT_TEMPLATE.format(name) for name in CABINETS]
DRAWER_JOINTS = [JOINT_TEMPLATE.format(name) for name in DRAWERS]

#LEFT_VISIBLE = ['chewie_door_left_joint', # chewie isn't in the viewcone though
#                'dagger_door_left_joint', 'dagger_door_right_joint']

# ZED_LEFT_JOINTS = [
#    'indigo_drawer_top_joint', 'indigo_drawer_bottom_joint',
# ]
ZED_LEFT_JOINTS = [JOINT_TEMPLATE.format(name) for name in ZED_LEFT_SURFACES
                   if JOINT_TEMPLATE.format(name) in (CABINET_JOINTS + DRAWER_JOINTS)]

ALL_JOINTS = ZED_LEFT_JOINTS
#ALL_JOINTS = [] + DRAWER_JOINTS + CABINET_JOINTS

################################################################################

def ycb_type_from_file(path):
    return path.split('_', 1)[-1]

def get_ycb_types():
    return sorted(map(ycb_type_from_file, os.listdir(YCB_PATH)))

def get_ycb_obj_path(ycb_type):
    # TODO: simplify geometry (although pybullet does this automatically)
    ycb_path = YCB_PATH
    path_from_type = {ycb_type_from_file(path): path for path in os.listdir(ycb_path)}
    if ycb_type not in path_from_type:
        return None
    # texture_map.png textured.mtl textured.obj textured_simple.obj textured_simple.obj.mtl
    #return os.path.join(ycb_path, path_from_type[ycb_type], 'textured_simple.obj')
    return os.path.join(ycb_path, path_from_type[ycb_type], 'textured.obj')

def load_ycb(ycb_name, **kwargs):
    # TODO: simplify geometry
    ycb_type = type_from_name(ycb_name)
    ycb_obj_path = get_ycb_obj_path(ycb_type)
    assert ycb_obj_path is not None
    # TODO: set color (as average) or texture
    return create_obj(ycb_obj_path, color=None, **kwargs)

def name_from_type(obj_type, suffix=''):
    return NAME_TEMPLATE.format(obj_type, suffix)

def type_from_name(name):
    return name.strip(string.digits)

def get_block_path(block_name):
    raise NotImplementedError(block_name)
    block_type = type_from_name(block_name)
    size, color, block = block_type.split('_')
    assert block == 'block'

def get_obj_path(obj_type):
    if 'block' in obj_type:
        return get_block_path(obj_type)
    return get_ycb_obj_path(obj_type)

################################################################################

def get_tool_from_root(robot):
    root_link = link_from_name(robot, get_gripper_link(robot))
    tool_link = link_from_name(robot, get_tool_link(robot))
    return multiply(invert(get_link_pose(robot, tool_link)),
                    get_link_pose(robot, root_link))

def set_tool_pose(world, tool_pose):
    #root_from_urdf = multiply(invert(get_pose(world.gripper, BASE_LINK)), # Previously 0?
    #                          get_pose(world.gripper))
    root_from_urdf = unit_pose()
    tool_from_root = get_tool_from_root(world.robot)
    set_pose(world.gripper, multiply(tool_pose, tool_from_root, root_from_urdf))

def iterate_approach_path(world, pose, grasp, body=None):
    world_from_body = pose.get_world_from_body()
    grasp_pose = multiply(world_from_body, invert(grasp.grasp_pose))
    approach_pose = multiply(world_from_body, invert(grasp.pregrasp_pose))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_tool_pose(world, tool_pose)
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.grasp_pose))
        yield

def get_gripper_link(robot):
    robot_name = get_body_name(robot)
    if robot_name == FRANKA_CARTER:
        return FRANKA_GRIPPER_LINK
    #elif robot_name == EVE:
    #    #return EVE_GRIPPER_LINK.format(a='l') # TODO: issue copying *.dae
    #    return EVE_GRIPPER_LINK.format(arm=DEFAULT_ARM)
    raise ValueError(robot_name)

def get_tool_link(robot):
    robot_name = get_body_name(robot)
    if robot_name == FRANKA_CARTER:
        return FRANKA_TOOL_LINK
    #elif robot_name == EVE:
    #    return EVE_TOOL_LINK.format(arm=DEFAULT_ARM)
    raise ValueError(robot_name)

def create_gripper(robot, visual=False):
    gripper_link = link_from_name(robot, get_gripper_link(robot))
    links = get_link_descendants(robot, gripper_link) # get_link_descendants | get_link_subtree
    with LockRenderer():
        # Actually uses the parent of the first link
        gripper = clone_body(robot, links=links, visual=False, collision=True)  # TODO: joint limits
        if not visual:
            for link in get_all_links(gripper):
                set_color(gripper, np.zeros(4), link)
    #dump_body(robot)
    #dump_body(gripper)
    #user_input()
    return gripper

class FConf(Conf):
    def __repr__(self):
        if len(self.joints) == 2:
            prefix = 'dq'
        elif len(self.joints) == 2:
            prefix = 'gq'
        elif len(self.joints) == 3:
            prefix = 'bq'
        elif len(self.joints) == 7:
            prefix = 'aq'
        else:
            prefix = 'q'
        return '{}{}'.format(prefix, id(self) % 1000)

################################################################################

def surface_from_name(surface_name):
    return SURFACE_FROM_NAME.get(surface_name, Surface(surface_name, SURFACE_TOP, []))

def surface_from_joint(joint_name):
    for name, surface in SURFACE_FROM_NAME.items():
        if joint_name in surface.joints:
            return name
    raise ValueError(joint_name)

def create_surface_attachment(world, obj_name, surface_name):
    body = world.get_body(obj_name)
    if surface_name in ENV_SURFACES:
        surface_body = world.environment_bodies[surface_name]
        surface_link = BASE_LINK
    else:
        surface = surface_from_name(surface_name)
        surface_body = world.kitchen
        surface_link = link_from_name(surface_body, surface.link)
    return create_attachment(surface_body, surface_link, body)

def pose_from_attachment(attachment, **kwargs):
    return RelPose(attachment.child, reference_body=attachment.parent,
                   reference_link=attachment.parent_link, confs=[attachment], **kwargs)

def create_relative_pose(world, name, surface, **kwargs):
    attachment = create_surface_attachment(world, name, surface)
    return pose_from_attachment(attachment, support=surface, **kwargs)

def create_world_pose(world, name, **kwargs):
    attachment = create_attachment(world.kitchen, BASE_LINK, world.get_body(name))
    return pose_from_attachment(attachment, support=None, **kwargs)

class RelPose(object):
#class RelPose(Pose):
    def __init__(self, body, #link=BASE_LINK,
                 reference_body=None, reference_link=BASE_LINK,
                 confs=[], support=None, init=False):
        self.body = body
        #self.link = link
        self.reference_body = reference_body
        self.reference_link = reference_link
        # Could also perform recursively
        self.confs = tuple(confs) # Attachment is treated as a conf
        self.support = support
        self.init = init
        self.observations = 0
        # TODO: method for automatically composing these
    @property
    def bodies(self):
        bodies = set() # (self.body, None)
        #if self.reference_body is not None:
        #    bodies.update({self.reference_body, frozenset(get_link_subtree(self.body, self.reference_link))})
        for conf in self.confs:
            bodies.update(conf.bodies)
        return bodies
    def assign(self):
        for conf in self.confs: # Assumed to be totally ordered
            conf.assign()
    def get_world_from_reference(self):
        if self.reference_body is None:
            return unit_pose()
        self.assign()
        return get_link_pose(self.reference_body, self.reference_link)
    def get_world_from_body(self):
        self.assign()
        return get_link_pose(self.body, BASE_LINK)
    def get_reference_from_body(self):
        return multiply(invert(self.get_world_from_reference()),
                        self.get_world_from_body())
    def draw(self, **kwargs):
        point_reference = point_from_pose(self.get_reference_from_body())
        if self.reference_body is None:
            return draw_point(point_reference, **kwargs)
        return draw_point(point_reference, parent=self.reference_body,
                          parent_link=self.reference_link, **kwargs)
    def __repr__(self):
        if self.reference_body is None:
            return 'wp{}'.format(id(self) % 1000)
        return 'rp{}'.format(id(self) % 1000)

def compute_surface_aabb(world, surface_name):
    if surface_name in ENV_SURFACES: # TODO: clean this up
        # TODO: the aabb for golf is off the table
        surface_body = world.environment_bodies[surface_name]
        return get_aabb(surface_body)
    surface_body = world.kitchen
    surface_name, shape_name, _ = surface_from_name(surface_name)
    surface_link = link_from_name(surface_body, surface_name)
    surface_pose = get_link_pose(surface_body, surface_link)
    if shape_name == SURFACE_TOP:
        surface_aabb = get_aabb(surface_body, surface_link)
    elif shape_name == SURFACE_BOTTOM:
        data = sorted(get_collision_data(surface_body, surface_link),
                      key=lambda d: point_from_pose(get_data_pose(d))[2])[0]
        extent = np.array(get_data_extents(data))
        aabb = AABB(-extent/2., +extent/2.)
        vertices = apply_affine(multiply(surface_pose, get_data_pose(data)), get_aabb_vertices(aabb))
        surface_aabb = aabb_from_points(vertices)
    else:
        [data] = filter(lambda d: d.filename != '',
                        get_collision_data(surface_body, surface_link))
        meshes = read_obj(data.filename)
        #colors = spaced_colors(len(meshes))
        #set_color(surface_body, link=surface_link, color=np.zeros(4))
        mesh = meshes[shape_name]
        #for i, (name, mesh) in enumerate(meshes.items()):
        mesh = tform_mesh(multiply(surface_pose, get_data_pose(data)), mesh=mesh)
        surface_aabb = aabb_from_points(mesh.vertices)
        #add_text(surface_name, position=surface_aabb[1])
        #draw_mesh(mesh, color=colors[i])
        #wait_for_user()
    #draw_aabb(surface_aabb)
    #wait_for_user()
    return surface_aabb

################################################################################

INVALID_GRASPS = {
    TOP_GRASP: [MUSTARD, BOWL], #, TOMATO_SOUP],
    SIDE_GRASP: [BOWL, SPAM, BLOCK],
}

MID_SIDE_GRASPS = [MUSTARD, SUGAR, CHEEZIT]

REFERENCE_POSE = {
    MUSTARD: Pose(euler=Euler(yaw=math.radians(24))),
}

CYLINDERS = [TOMATO_SOUP]

def is_valid_grasp_type(obj_name, grasp_type):
    obj_type = type_from_name(obj_name)
    return obj_type not in INVALID_GRASPS.get(grasp_type, [])

class Grasp(object):
    def __init__(self, world, body_name, grasp_type, index, grasp_pose, pregrasp_pose,
                 grasp_width=None):
        self.world = world
        self.body_name = body_name
        self.grasp_type = grasp_type
        self.index = index
        self.grasp_pose = grasp_pose
        self.pregrasp_pose = pregrasp_pose
        self.grasp_width = grasp_width
    def assign(self):
        attachment = self.get_attachment()
        attachment.assign()
    def get_attachment(self):
        return Attachment(self.world.robot, self.world.tool_link,
                          self.grasp_pose, self.world.get_body(self.body_name))
    def get_gripper_conf(self):
        conf = [self.grasp_width] * len(self.world.gripper_joints)
        return FConf(self.world.robot, self.world.gripper_joints, conf)
    def set_gripper(self):
        return self.get_gripper_conf().assign()
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.grasp_type, self.index)

def get_grasps(world, name, grasp_types=GRASP_TYPES, pre_distance=APPROACH_DISTANCE, **kwargs):
    use_width = world.robot_name == FRANKA_CARTER
    body = world.get_body(name)
    #fraction = 0.25
    obj_type = type_from_name(name)
    body_pose = REFERENCE_POSE.get(obj_type, unit_pose())
    center, extent = approximate_as_prism(body, body_pose)

    for grasp_type in grasp_types:
        if not implies(world.is_real(), is_valid_grasp_type(name, grasp_type)):
            continue
        #assert is_valid_grasp_type(name, grasp_type)
        if grasp_type == TOP_GRASP:
            grasp_length = 1.5 * FINGER_EXTENT[2]  # fraction = 0.5
            pre_direction = pre_distance * get_unit_vector([0, 0, 1])
            post_direction = unit_point()
            generator = get_top_grasps(body, under=True, tool_pose=TOOL_POSE, body_pose=body_pose,
                                       grasp_length=grasp_length, max_width=np.inf, **kwargs)
        elif grasp_type == SIDE_GRASP:
            # Take max of height and something
            grasp_length = 1.75 * FINGER_EXTENT[2]  # No problem if pushing a little
            x, z = pre_distance * get_unit_vector([3, -1])
            pre_direction = [0, 0, x]
            post_direction = [0, 0, z]
            top_offset = extent[2] / 2 if obj_type in MID_SIDE_GRASPS else 1.0*FINGER_EXTENT[0]
            # Under grasps are actually easier for this robot
            # TODO: bug in under in that it grasps at the bottom
            generator = get_side_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=body_pose,
                                        grasp_length=grasp_length, top_offset=top_offset, max_width=np.inf, **kwargs)
            # if world.robot_name == FRANKA_CARTER else unit_pose()
            generator = (multiply(Pose(euler=Euler(yaw=yaw)), grasp)
                         for grasp in generator for yaw in [0, np.pi])
        else:
            raise ValueError(grasp_type)
        grasp_poses = randomize(list(generator))
        if obj_type in CYLINDERS:
            # TODO: filter first
            grasp_poses = (multiply(grasp_pose, Pose(euler=Euler(
                yaw=random.uniform(-math.pi, math.pi)))) for grasp_pose in cycle(grasp_poses))
        for i, grasp_pose in enumerate(grasp_poses):
            pregrasp_pose = multiply(Pose(point=pre_direction), grasp_pose,
                                     Pose(point=post_direction))
            grasp = Grasp(world, name, grasp_type, i, grasp_pose, pregrasp_pose)
            with BodySaver(body):
                grasp.get_attachment().assign()
                with BodySaver(world.robot):
                    grasp.grasp_width = close_until_collision(
                        world.robot, world.gripper_joints, bodies=[body])
            #print(get_joint_positions(world.robot, world.arm_joints)[-1])
            #draw_pose(unit_pose(), parent=world.robot, parent_link=world.tool_link)
            #grasp.get_attachment().assign()
            #wait_for_user()
            ##for value in get_joint_limits(world.robot, world.arm_joints[-1]):
            #for value in [-1.8973, 0, +1.8973]:
            #    set_joint_position(world.robot, world.arm_joints[-1], value)
            #    grasp.get_attachment().assign()
            #    wait_for_user()
            if use_width and (grasp.grasp_width is None):
                continue
            yield grasp

################################################################################

def custom_limits_from_base_limits(robot, base_limits, yaw_limit=None):
    x_limits, y_limits = zip(*base_limits)
    custom_limits = {
        joint_from_name(robot, 'x'): x_limits,
        joint_from_name(robot, 'y'): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update({
            joint_from_name(robot, 'theta'): yaw_limit,
        })
    return custom_limits

def get_descendant_obstacles(body, link=BASE_LINK):
    # TODO: deprecate?
    return {(body, frozenset([link]))
            for link in get_link_subtree(body, link)}

################################################################################

Z_EPSILON = 2.5e-3


def open_surface_joints(world, surface_name, joint_names=ALL_JOINTS):
    surface = surface_from_name(surface_name)
    for joint_name in surface.joints:
        joint = joint_from_name(world.kitchen, joint_name)
        if joint_name in joint_names:
            # TODO: remove this mechanic in the future
            world.open_door(joint)

def get_surface_obstacles(world, surface_name):
    surface = surface_from_name(surface_name)
    obstacles = set()
    for joint_name in surface.joints:
        link = child_link_from_joint(joint_from_name(world.kitchen, joint_name))
        obstacles.update(get_descendant_obstacles(world.kitchen, link))
    # Be careful to call this before each check
    open_surface_joints(world, surface_name, joint_names=CABINET_JOINTS)
    return obstacles

def test_supported(world, body, surface_name, collisions=True):
    # TODO: is_center_on_aabb or is_placed_on_aabb
    surface_aabb = compute_surface_aabb(world, surface_name)
    # TODO: epsilon thresholds?
    if not is_placed_on_aabb(body, surface_aabb):  # , above_epsilon=z_offset+1e-3):
        return False
    obstacles = world.static_obstacles | get_surface_obstacles(world, surface_name)
    if not collisions:
        obstacles = set()
    #print([get_link_name(obst[0], list(obst[1])[0]) for obst in obstacles
    #       if pairwise_collision(body, obst)])
    return not any(pairwise_collision(body, obst) for obst in obstacles)


def get_link_obstacles(world, link_name):
    if link_name in world.movable:
        return flatten_links(world.get_body(link_name))
    elif has_link(world.kitchen, link_name):
        link = link_from_name(world.kitchen, link_name)
        return flatten_links(world.kitchen, get_link_subtree(world.kitchen, link)) # subtree?
    assert link_name in SURFACE_FROM_NAME
    return set()

################################################################################

def are_confs_close(conf1, conf2, tol=1e-8):
    assert (conf1.body == conf2.body) and (conf1.joints == conf2.joints)
    difference_fn = get_difference_fn(conf1.body, conf1.joints)
    difference = difference_fn(conf1.values, conf2.values)
    return np.allclose(difference, np.zeros(len(conf1.joints)), rtol=0., atol=tol)


def translate_linearly(world, distance):
    # TODO: could just apply in the base frame
    x, y, theta = get_joint_positions(world.robot, world.base_joints)
    pos = np.array([x, y])
    goal_pos = pos + distance * unit_from_theta(theta)
    goal_pose = np.append(goal_pos, [theta])
    return goal_pose