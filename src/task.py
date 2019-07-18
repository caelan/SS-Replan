import numpy as np

from pybullet_tools.pr2_utils import get_viewcone
from pybullet_tools.utils import stable_z, link_from_name, set_pose, Pose, Point, Euler, multiply, get_pose, \
    apply_alpha, RED, step_simulation, joint_from_name, set_all_static, WorldSaver
from src.observation import KITCHEN_FROM_ZED_LEFT, DEPTH, CAMERA_MATRIX
from src.stream import get_stable_gen
from src.utils import BLOCK_SIZES, BLOCK_COLORS, get_block_path, COUNTERS, get_ycb_obj_path, DRAWER_JOINTS, ALL_JOINTS


class Task(object):
    def __init__(self, world, skeletons=[],
                 movable_base=True, noisy_base=True,
                 return_init_bq=True, return_init_aq=True,
                 goal_hand_empty=False, goal_holding=[],
                 goal_on={}, goal_closed=[], goal_cooked=[]):
        self.world = world
        world.task = self
        self.skeletons = list(skeletons)
        self.movable_base = movable_base
        self.noisy_base = noisy_base
        self.return_init_bq = return_init_bq
        self.return_init_aq = return_init_aq
        self.goal_hand_empty = goal_hand_empty
        self.goal_holding = set(goal_holding)
        self.goal_on = dict(goal_on)
        self.goal_closed = set(goal_closed)
        self.goal_cooked = set(goal_cooked)
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, {key: value for key, value in self.__dict__.items()
                                                       if value not in [self.world]})

################################################################################

def add_block(world, x=0.1, y=1.15, yaw=0, idx=0):
    # TODO: automatically produce a unique name
    entity_name = '{}_{}_block{}'.format(BLOCK_SIZES[-1], BLOCK_COLORS[0], idx)
    entity_path = get_block_path(entity_name)
    #entity_name = 'potted_meat_can'
    #entity_path = get_ycb_obj_path(entity_name)
    world.add_body(entity_name, entity_path)
    entity_body = world.get_body(entity_name)
    z = stable_z(entity_body, world.kitchen, link_from_name(world.kitchen, COUNTERS[0]))
    set_pose(entity_body, Pose(Point(x, y, z), Euler(yaw=yaw)))
    return entity_name

def add_box(world, x=0.2, y=1.2, yaw=np.pi/4, idx=0):
    obstruction_name = 'cracker_box{}'.format(idx)
    obstruction_path = get_ycb_obj_path(obstruction_name)
    world.add_body(obstruction_name, obstruction_path, color=np.ones(4))
    obstruction_body = world.get_body(obstruction_name)
    z = stable_z(obstruction_body, world.kitchen, link_from_name(world.kitchen, COUNTERS[0]))
    set_pose(obstruction_body, Pose(Point(x, y, z), Euler(yaw=yaw)))
    return obstruction_name

def add_kinect(world):
    # TODO: could intersect convex with half plane
    world_from_zed_left = multiply(get_pose(world.kitchen), KITCHEN_FROM_ZED_LEFT)
    cone_body = get_viewcone(depth=DEPTH, camera_matrix=CAMERA_MATRIX, color=apply_alpha(RED, 0.1))
    set_pose(cone_body, world_from_zed_left)
    step_simulation()
    return cone_body

################################################################################

def sample_placement(world, entity_name, surface_name, **kwargs):
    # TODO: check for collisions
    with WorldSaver():
        placement_gen = get_stable_gen(world, pos_scale=1e-3, rot_scale=1e-2, **kwargs)
        pose, = next(placement_gen(entity_name, surface_name), (None,))
    assert pose is not None
    pose.assign()

def close_all_doors(world):
    for joint in world.kitchen_joints:
        world.close_door(joint)

def open_all_doors(world):
    for joint in world.kitchen_joints:
        world.open_door(joint)

################################################################################

def relocate_block(world, **kwargs):
    #open_all_doors(world)
    entity_name = add_block(world, idx=0)
    initial_surface = 'hitman_tmp'
    goal_surface = 'indigo_tmp'
    set_all_static()
    add_kinect(world)
    sample_placement(world, entity_name, initial_surface, learned=True)

    return Task(world, movable_base=True,
                return_init_bq=True, # return_init_aq=False,
                goal_holding=[entity_name],
                #goal_on={entity_name: goal_surface},
                #goal_closed=ALL_JOINTS,
                **kwargs)

################################################################################

# skeleton = [
#     ('calibrate', [WILD, WILD, WILD]),
#     ('move_base', [WILD, WILD, WILD]),
#     ('pull', ['indigo_drawer_top_joint', WILD, WILD,
#               'indigo_drawer_top', WILD, WILD, WILD, WILD, WILD  ]),
#     ('move_base', [WILD, WILD, WILD]),
#     ('pick', ['big_red_block0', WILD, WILD, WILD,
#               'indigo_drawer_top', WILD, WILD, WILD, WILD]),
#     ('move_base', [WILD, WILD, WILD]),
# ]

def stow_block(world, **kwargs):
    #world.open_gq.assign()
    # dump_link_cross_sections(world, link_name='indigo_drawer_top')
    # wait_for_user()

    entity_name = add_block(world, idx=0)
    #entity_name = add_block(world, x=0.2, y=1.15, idx=1) # Will be randomized anyways
    # obstruction_name = add_box(world)
    # test_grasps(world, entity_name)
    set_all_static()
    add_kinect(world)  # TODO: this needs to be after set_all_static

    #initial_surface = random.choice(DRAWERS) # COUNTERS | DRAWERS | SURFACES | CABINETS
    initial_surface = COUNTERS[0]
    #initial_surface = 'indigo_drawer_top'
    goal_surface = 'indigo_drawer_top' # baker | hitman_drawer_top | indigo_drawer_top | hitman_tmp | indigo_tmp
    print('Initial surface: | Goal surface: ', initial_surface, initial_surface)
    sample_placement(world, entity_name, initial_surface)

    return Task(world, movable_base=True,
                goal_hand_empty=False,
                #goal_holding=[entity_name],
                goal_on={entity_name: goal_surface},
                goal_closed=ALL_JOINTS, **kwargs)

TASKS = [
    relocate_block,
    stow_block,
]