import numpy as np

from pybullet_tools.pr2_utils import get_viewcone
from pybullet_tools.utils import stable_z, link_from_name, set_pose, Pose, Point, Euler, multiply, get_pose, \
    apply_alpha, RED, step_simulation, joint_from_name, set_all_static, WorldSaver
from src.observation import KITCHEN_FROM_ZED_LEFT, DEPTH, CAMERA_MATRIX
from src.stream import get_stable_gen
from src.utils import BLOCK_SIZES, BLOCK_COLORS, get_block_path, COUNTERS, get_ycb_obj_path, DRAWER_JOINTS, ALL_JOINTS


class Task(object):
    def __init__(self, world,  skeletons=[],
                 movable_base=True, fixed_base=False, noisy_base=False,
                 goal_init_conf=False, goal_hand_empty=False, goal_holding=[],
                 goal_on={}, goal_closed=[], goal_cooked=[]):
        self.world = world
        self.skeletons = list(skeletons)
        self.movable_base = movable_base
        self.fixed_base = fixed_base
        self.noisy_base = noisy_base
        self.goal_init_conf = goal_init_conf
        self.goal_hand_empty = goal_hand_empty
        self.goal_holding = set(goal_holding)
        self.goal_on = dict(goal_on)
        self.goal_closed = set(goal_closed)
        self.goal_cooked = set(goal_cooked)
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, {key: value for key, value in self.__dict__.items()
                                                       if value not in [self.world]})

################################################################################

def add_block(world):
    entity_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], 0)
    entity_path = get_block_path(entity_name)
    #entity_name = 'potted_meat_can'
    #entity_path = get_ycb_obj_path(entity_name)
    world.add_body(entity_name, entity_path)
    entity_body = world.get_body(entity_name)
    z = stable_z(entity_body, world.kitchen, link_from_name(world.kitchen, COUNTERS[0]))
    set_pose(entity_body, Pose(Point(0.1, 1.15, z), Euler()))
    return entity_name

def add_box(world):
    obstruction_name = 'cracker_box'
    obstruction_path = get_ycb_obj_path(obstruction_name)
    world.add_body(obstruction_name, obstruction_path, color=np.ones(4))
    obstruction_body = world.get_body(obstruction_name)
    z = stable_z(obstruction_body, world.kitchen, link_from_name(world.kitchen, COUNTERS[0]))
    set_pose(obstruction_body, Pose(Point(0.2, 1.2, z), Euler(yaw=np.pi/4)))
    return obstruction_name

def add_kinect(world):
    # TODO: could intersect convex with half plane
    world_from_zed_left = multiply(get_pose(world.kitchen), KITCHEN_FROM_ZED_LEFT)
    cone_body = get_viewcone(depth=DEPTH, camera_matrix=CAMERA_MATRIX, color=apply_alpha(RED, 0.1))
    set_pose(cone_body, world_from_zed_left)
    step_simulation()
    return cone_body

################################################################################

def stow_block(world):
    # for joint in world.kitchen_joints:
    # for name in LEFT_VISIBLE:
    for name in DRAWER_JOINTS[1:2]:
        joint = joint_from_name(world.kitchen, name)
        # world.open_door(joint)
        # world.close_door(joint)
    # dump_link_cross_sections(world, link_name='indigo_drawer_top')
    # wait_for_user()

    entity_name = add_block(world)
    # obstruction_name = add_box(world)
    # test_grasps(world, entity_name)
    set_all_static()
    add_kinect(world)  # TODO: this needs to be after set_all_static

    #surface_name = random.choice(DRAWERS)
    surface_name = COUNTERS[0] # COUNTERS | DRAWERS | SURFACES | CABINETS
    #surface_name = 'indigo_tmp' # hitman_drawer_top_joint | hitman_tmp | indigo_tmp
    print('Initial surface:', surface_name)
    with WorldSaver():
        placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)
        pose, = next(placement_gen(entity_name, surface_name), (None,))
    assert pose is not None
    pose.assign()
    #wait_for_user()

    return Task(world, goal_init_conf=True, goal_hand_empty=True,
                goal_on={entity_name: 'indigo_drawer_top'},
                goal_closed=ALL_JOINTS)

TASKS = [
    stow_block,
]