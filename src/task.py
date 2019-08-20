from __future__ import print_function

import numpy as np
import random
import time

from pybullet_tools.utils import set_pose, Pose, Point, Euler, multiply, get_pose, \
    create_box, set_all_static, COLOR_FROM_NAME, \
    stable_z_on_aabb, pairwise_collision, elapsed_time
from src.stream import get_stable_gen, MAX_COST
from src.utils import JOINT_TEMPLATE, BLOCK_SIZES, BLOCK_COLORS, COUNTERS, \
    ALL_JOINTS, LEFT_CAMERA, CAMERA_MATRIX, CAMERA_POSES, CAMERAS, compute_surface_aabb, \
    BLOCK_TEMPLATE, name_from_type, GRASP_TYPES, SIDE_GRASP, joint_from_name
from examples.discrete_belief.dist import UniformDist, DeltaDist
#from examples.pybullet.pr2_belief.problems import BeliefState, BeliefTask, OTHER
from src.belief import create_surface_belief

class Task(object):
    def __init__(self, world, prior={}, skeletons=[],
                 movable_base=True, noisy_base=True, grasp_types=GRASP_TYPES,
                 return_init_bq=False, return_init_aq=False,
                 goal_hand_empty=False, goal_holding=[], goal_detected=[],
                 goal_on={}, goal_open=[], goal_closed=[], goal_cooked=[],
                 init=[], goal=[], max_cost=MAX_COST):
        self.world = world
        world.task = self
        self.prior = dict(prior) # DiscreteDist over
        self.skeletons = list(skeletons)
        self.movable_base = movable_base
        self.noisy_base = noisy_base
        self.grasp_types = tuple(grasp_types)
        self.return_init_bq = return_init_bq
        self.return_init_aq = return_init_aq
        self.goal_hand_empty = goal_hand_empty
        self.goal_holding = set(goal_holding)
        self.goal_on = dict(goal_on)
        self.goal_detected = set(goal_detected)
        self.goal_open = set(goal_open)
        self.goal_closed = set(goal_closed)
        self.goal_cooked = set(goal_cooked)
        self.init = init
        self.goal = goal
        self.max_cost = max_cost # TODO: use instead of the default
    @property
    def objects(self):
        return set(self.prior.keys())
    def create_belief(self):
        t0 = time.time()
        print('Creating initial belief')
        belief = create_surface_belief(self.world, self.prior)
        belief.task = self
        print('Took {:2f} seconds'.format(elapsed_time(t0)))
        return belief
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, {
            key: value for key, value in self.__dict__.items() if value not in [self.world]})

################################################################################

# (x, y, yaw)
UNIT_POSE2D = (0., 0., 0.)
BOX_POSE2D = (0.1, 1.15, 0.)
SPAM_POSE2D = (0.125, 1.175, -np.pi / 4)
CRACKER_POSE2D = (0.2, 1.2, np.pi/4)

def pose2d_on_surface(world, entity_name, surface_name, pose2d=UNIT_POSE2D):
    x, y, yaw = pose2d
    body = world.get_body(entity_name)
    surface_aabb = compute_surface_aabb(world, surface_name)
    z = stable_z_on_aabb(body, surface_aabb)
    pose = Pose(Point(x, y, z), Euler(yaw=yaw))
    set_pose(body, pose)
    return pose

def add_block(world, idx=0, **kwargs):
    # TODO: automatically produce a unique name
    block_type = BLOCK_TEMPLATE.format(BLOCK_SIZES[-1], BLOCK_COLORS[0])
    #block_type = 'potted_meat_can'
    name = name_from_type(block_type, idx)
    world.add_body(name)
    pose2d_on_surface(world, name, COUNTERS[0], **kwargs)
    return name


def add_cracker_box(world, idx=0, **kwargs):
    ycb_type = 'cracker_box'
    name = name_from_type(ycb_type, idx)
    world.add_body(name, color=np.ones(4))
    pose2d_on_surface(world, name, COUNTERS[0], **kwargs)
    return name


def add_cylinder(world, color_name, idx=0, **kwargs):
    height = 0.14
    width = 0.07
    name = name_from_type(color_name, idx)
    # TODO: geometry type
    body = create_box(w=width, l=width, h=height, color=COLOR_FROM_NAME[color_name])
    # body = create_cylinder(radius=width/2, height=height, color=COLOR_FROM_NAME[color_name])
    world.add(name, body)
    # pose2d_on_surface(world, name, COUNTERS[0], **kwargs)
    return name

def add_kinect(world, camera_name=LEFT_CAMERA):
    # TODO: could intersect convex with half plane
    world_from_camera = multiply(get_pose(world.kitchen), CAMERA_POSES[camera_name])
    world.add_camera(camera_name, world_from_camera, CAMERA_MATRIX)

################################################################################

def sample_placement(world, entity_name, surface_name, **kwargs):
    entity_body = world.get_body(entity_name)
    placement_gen = get_stable_gen(world, pos_scale=1e-3, rot_scale=1e-2, **kwargs)
    for pose, in placement_gen(entity_name, surface_name):
        pose.assign()
        if not any(pairwise_collision(entity_body, obst_body) for obst_body in
                   world.body_from_name.values() if entity_body != obst_body):
            return pose
    raise RuntimeError('Unable to find a pose for object {} on surface {}'.format(entity_name, surface_name))

def close_all_doors(world):
    for joint in world.kitchen_joints:
        world.close_door(joint)

def open_all_doors(world):
    for joint in world.kitchen_joints:
        world.open_door(joint)

################################################################################

def detect_block(world, **kwargs):
    entity_name = add_block(world, idx=0, pose2d=BOX_POSE2D)
    obstruction_name = add_cracker_box(world, idx=0, pose2d=CRACKER_POSE2D)
    #other_name = add_box(world, idx=1)
    set_all_static()
    for side in CAMERAS[:1]:
        add_kinect(world, side)
    goal_surface = 'indigo_drawer_top'
    initial_distribution = UniformDist([goal_surface]) # indigo_tmp
    initial_surface = initial_distribution.sample()
    if random.random() < 0.:
        # TODO: sometimes base/arm failure causes the planner to freeze
        # Freezing is because the planner is struggling to find new samples
        sample_placement(world, entity_name, initial_surface, learned=True)
    #sample_placement(world, other_name, 'hitman_tmp', learned=True)

    prior = {
        entity_name: UniformDist(['indigo_tmp']),  # 'indigo_tmp', 'indigo_drawer_top'
        obstruction_name: DeltaDist('indigo_tmp'),
    }
    return Task(world, prior=prior, movable_base=True,
                #return_init_bq=True, return_init_aq=True,
                #goal_detected=[entity_name],
                #goal_holding=[entity_name],
                goal_on={entity_name: goal_surface},
                goal_closed=ALL_JOINTS,
                **kwargs)

################################################################################

def hold_block(world, num=5, **kwargs):
    # TODO: compare with the NN grasp prediction in clutter
    # TODO: consider a task where most directions are blocked except for one
    initial_surface = 'indigo_tmp'
    # initial_surface = 'dagger_door_left'
    # joint_name = JOINT_TEMPLATE.format(initial_surface)
    #world.open_door(joint_from_name(world.kitchen, joint_name))
    #open_all_doors(world)

    prior = {}
    # green_name = add_block(world, idx=0, pose2d=BOX_POSE2D)
    green_name = add_cylinder(world, 'green', idx=0)
    prior[green_name] = DeltaDist(initial_surface)
    sample_placement(world, green_name, initial_surface, learned=True)
    for idx in range(num):
        red_name = add_cylinder(world, 'red', idx=idx)
        prior[red_name] = DeltaDist(initial_surface)
        sample_placement(world, red_name, initial_surface, learned=True)

    set_all_static()
    add_kinect(world)

    return Task(world, prior=prior, movable_base=True,
                # grasp_types=GRASP_TYPES,
                grasp_types=[SIDE_GRASP],
                return_init_bq=True, return_init_aq=True,
                goal_holding=[green_name],
                #goal_closed=ALL_JOINTS,
                **kwargs)


################################################################################

def cracker_drawer(world, **kwargs):
    initial_surface = 'indigo_drawer_top'
    # initial_surface = 'indigo_drawer_bottom'
    joint_name = JOINT_TEMPLATE.format(initial_surface)
    world.open_door(joint_from_name(world.kitchen, joint_name))
    # open_all_doors(world)

    obj_name = add_cracker_box(world, idx=0)
    prior = {obj_name: DeltaDist(initial_surface)}
    sample_placement(world, obj_name, initial_surface, learned=True)

    set_all_static()
    add_kinect(world)

    return Task(world, prior=prior, movable_base=True,
                return_init_bq=True, return_init_aq=True,
                # goal_open=[JOINT_TEMPLATE.format('indigo_drawer_top')],
                goal_closed=ALL_JOINTS,
                **kwargs)

################################################################################

BASE_POSE2D = (0.73, 0.80, -np.pi)


def fixed_stow(world, **kwargs):
    # set_base_values
    entity_name = add_block(world, idx=0, pose2d=BOX_POSE2D)
    set_all_static()
    add_kinect(world)

    # set_base_values(world.robot, BASE_POSE2D)
    world.set_base_conf(BASE_POSE2D)

    #initial_surface, goal_surface = 'indigo_tmp', 'indigo_drawer_top'
    initial_surface, goal_surface = 'indigo_drawer_top', 'indigo_drawer_top'
    if initial_surface == 'indigo_drawer_top':
        sample_placement(world, entity_name, initial_surface, learned=True)
    # joint_name = JOINT_TEMPLATE.format(goal_surface)
    #world.open_door(joint_from_name(world.kitchen, joint_name))

    # TODO: declare success if already believe it's in the drawer or require detection?
    prior = {
        entity_name: UniformDist([initial_surface]),
    }
    return Task(world, prior=prior, movable_base=False,
                #goal_detected=[entity_name],
                goal_on={entity_name: goal_surface},
                return_init_bq=True, return_init_aq=True,
                #goal_open=[joint_name],
                goal_closed=ALL_JOINTS,
                **kwargs)

################################################################################

def stow_block(world, num=2, **kwargs):
    #world.open_gq.assign()
    # dump_link_cross_sections(world, link_name='indigo_drawer_top')
    # wait_for_user()

    # initial_surface = random.choice(DRAWERS) # COUNTERS | DRAWERS | SURFACES | CABINETS
    initial_surface = 'indigo_tmp'  # hitman_tmp | indigo_tmp | range
    # initial_surface = 'indigo_drawer_top'
    goal_surface = 'indigo_drawer_top'  # baker | hitman_drawer_top | indigo_drawer_top | hitman_tmp | indigo_tmp
    joint_name = 'indigo_drawer_top_joint'
    print('Initial surface: | Goal surface: ', initial_surface, initial_surface)

    prior = {}
    goal_on = {}
    for idx in range(num):
        entity_name = add_block(world, idx=idx, pose2d=SPAM_POSE2D)
        prior[entity_name] = DeltaDist(initial_surface)
        goal_on[entity_name] = goal_surface
        sample_placement(world, entity_name, initial_surface, learned=True)
    #obstruction_name = add_box(world, idx=0)
    #sample_placement(world, obstruction_name, 'hitman_tmp')
    set_all_static()
    add_kinect(world)  # TODO: this needs to be after set_all_static

    #world.open_door(joint_from_name(world.kitchen, joint_name))

    #initial_surface = 'golf' # range | table | golf
    #surface_body = world.environment_bodies[initial_surface]
    #draw_aabb(get_aabb(surface_body))
    #while True:
    #    sample_placement(world, entity_name, surface_name=initial_surface, learned=False)
    #    wait_for_user()

    return Task(world, prior=prior, movable_base=True,
                #goal_holding=[entity_name],
                goal_on=goal_on,
                return_init_bq=True, return_init_aq=True,
                #goal_open=[joint_name],
                goal_closed=ALL_JOINTS,
                **kwargs)

################################################################################

TASKS = [
    detect_block,
    hold_block,
    fixed_stow,
    stow_block,
    cracker_drawer,
]
