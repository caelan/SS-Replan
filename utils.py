import os

import numpy as np
import yaml

from pybullet_tools.pr2_utils import get_top_grasps, get_side_grasps, close_until_collision
from pybullet_tools.utils import connect, HideOutput, load_pybullet, dump_body, set_point, Point, add_data_path, \
    joints_from_names, joint_from_name, set_joint_positions, set_joint_position, get_min_limit, get_max_limit, \
    get_joint_name, Attachment, link_from_name, get_unit_vector, unit_pose, BodySaver, multiply, Pose, disconnect, \
    get_link_descendants, get_link_subtree, get_link_name, get_links

SRL_PATH = '/home/caelan/Programs/srl_system'
MODELS_PATH = './models'

YUMI_PATH = os.path.join(SRL_PATH, 'packages/external/abb_yumi_model/yumi_description/urdf/yumi_lula.urdf')
INTEL_PATH = os.path.join(SRL_PATH, 'packages/external/intel_robot_model/robots/intel_robot_model.urdf')
KUKA_ALLEGRO = os.path.join(SRL_PATH, 'packages/external/lula_kuka_allegro/urdf/lula_kuka_allegro.urdf')
BAXTER_PATH = os.path.join(SRL_PATH, 'packages/third_party/baxter_common/baxter_description/urdf/baxter.urdf')
ALLEGRO = os.path.join(SRL_PATH, 'packages/third_party/allegro_driver/description/allegro.urdf')
ABB_PATH = os.path.join(SRL_PATH, 'packages/third_party/abb_irb120_support/urdf/irb120_3_58.urdf') # irb120t_3_58.urdf
LBR4_PATH = os.path.join(SRL_PATH, 'packages/third_party/ll4ma_robots_description/urdf/lbr4/lbr4_kdl.urdf') # right_push_stick_rig.urdf |
FRANKA_PATH = os.path.join(MODELS_PATH, 'lula_franka_gen.urdf')
#FRANKA_PATH = os.path.join(SRL_PATH, 'packages/external/lula_franka/urdf/lula_franka_gen.urdf')
# ./packages/third_party/franka_ros/franka_description
CARTER_PATH = os.path.join(SRL_PATH, 'packages/third_party/carter_description/urdf/carter.urdf')
CARTER_FRANKA_PATH = os.path.join(MODELS_PATH, 'carter_description/urdf/carter_franka.urdf')

FRANKA_CARTER_PATH = os.path.join(MODELS_PATH, 'franka_carter.urdf')
#CARTER_FRANKA_PATH = os.path.join(SRL_PATH, 'packages/third_party/carter_description/urdf/carter_franka.urdf')
FRANKA_YAML = os.path.join(SRL_PATH, 'packages/external/lula_franka/config/robot_descriptor.yaml')

BLOCK_SIZES = ['small', 'big']
BLOCK_COLORS = ['red', 'green', 'blue', 'yellow']
BLOCK_PATH = os.path.join(SRL_PATH, 'packages/isaac_bridge/urdf/blocks/{}_block_{}.urdf')
YCB_PATH = os.path.join(SRL_PATH, 'packages/kitchen_demo_visualization/ycb/')

KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen.urdf')
#KITCHEN_PATH = '/home/caelan/Programs/robot_kitchen/model/robot_kitchen.sdf'
#KITCHEN_PATH = os.path.join(SRL_PATH, 'packages/kitchen_description/urdf/kitchen_part_right_gen.urdf')
KITCHEN_YAML = os.path.join(SRL_PATH, 'packages/kitchen_description/config/robot_descriptor.yaml')

BASE_JOINTS = ['x', 'y', 'theta']
FRANKA_TOOL_LINK = 'right_gripper'  # right_gripper | panda_wrist_end_pt | panda_forearm_end_pt
# +z: pointing, +y: left finger
FINGER_EXTENT = np.array([0.02, 0.01, 0.02]) # 2cm x 1cm x 2cm

KITCHEN = 'kitchen'
STOVES = ['range']
SURFACES = ['hitman_tmp', 'indigo_tmp'] + STOVES

#CABINET_PATH = os.path.join(SRL_PATH, 'packages/sektion_cabinet_model/urdf/sektion_cabinet.urdf')
# Could recursively find all *.urdf | *.sdf
# packages/posecnn_pytorch/ycb_render/
# packages/deepim_pytorch/ycb_render/

# find . -name "*.yaml"
# find . -name "*.yaml" | grep "franka"
# ./packages/kitchen_description/config/robot_descriptor.yaml
# ./packages/external/lula_franka/config/robot_descriptor.yaml
# ./packages/third_party/carter_description/launch/carter_franka_description.launch
# ./packages/internal/ros/lula_opt_ros/lula/opt/tracking_priors.cpp

################################################################################

def load_yaml(path):
    # grep -r --include="*.py" "yaml\." *
    # yaml.dump()
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

def get_block_path(name):
    size, color, block = name.split('_')
    assert block.startswith('block')
    return BLOCK_PATH.format(size, color)

################################################################################

class World(object):
    def __init__(self, robot_name='carter_franka', use_gui=True):
        self.client = connect(use_gui=use_gui)
        self.robot_name = robot_name
        if self.robot_name == 'carter_franka':
            urdf_path, yaml_path = FRANKA_CARTER_PATH, FRANKA_YAML
        else:
            raise ValueError(self.robot_name)
        with HideOutput():
            self.robot = load_pybullet(urdf_path)
        #dump_body(self.robot)
        self.robot_yaml = load_yaml(yaml_path)
        print(self.robot_yaml)
        self.set_initial_conf()

        with HideOutput():
            self.kitchen = load_pybullet(KITCHEN_PATH, fixed_base=True)
        dump_body(self.kitchen)
        self.kitchen_yaml = load_yaml(KITCHEN_YAML)
        print(self.kitchen_yaml)
        set_point(self.kitchen, Point(z=1.35))

        add_data_path()
        self.floor = load_pybullet('plane.urdf', fixed_base=True)
        self.body_from_name = {}
    @property
    def base_joints(self):
        return joints_from_names(self.robot, BASE_JOINTS)
    @property
    def arm_joints(self):
        return joints_from_names(self.robot, self.robot_yaml['cspace'])
    @property
    def gripper_joints(self):
        return [joint_from_name(self.robot, rule['name']) for rule in self.robot_yaml['cspace_to_urdf_rules']]
    @property
    def kitchen_joints(self):
        return joints_from_names(self.kitchen, self.kitchen_yaml['cspace'])
    @property
    def tool_link(self):
        return link_from_name(self.robot, FRANKA_TOOL_LINK)
    @property
    def door_links(self):
        door_links = set()
        for joint in self.kitchen_joints:
            door_links.update(get_link_subtree(self.kitchen, joint))
            #print(get_joint_name(self.kitchen, joint), [get_link_name(self.kitchen, link) for link in links])
        return door_links
    @property
    def static_obstacles(self):
        # link=None is fine
        return [(self.kitchen, set(get_links(self.kitchen)) - self.door_links)]
    @property
    def movable(self):
        return set(self.body_from_name)
    @property
    def initial_conf(self):
        return self.robot_yaml['default_q']
    def set_initial_conf(self):
        set_joint_positions(self.robot, self.base_joints, [2.0, 0, np.pi])
        for rule in self.robot_yaml['cspace_to_urdf_rules']:  # max is open
            joint = joint_from_name(self.robot, rule['name'])
            set_joint_position(self.robot, joint, rule['value'])
        set_joint_positions(self.robot, self.arm_joints, self.initial_conf)  # active_task_spaces
    def close_gripper(self):
        for joint in self.gripper_joints:
            set_joint_position(self.robot, joint, get_min_limit(self.robot, joint))
    def open_gripper(self):
        for joint in self.gripper_joints:
            set_joint_position(self.robot, joint, get_max_limit(self.robot, joint))
    def close_door(self, joint):
        close_fn = get_max_limit if 'left' in get_joint_name(self.kitchen, joint) else get_min_limit
        set_joint_position(self.kitchen, joint, close_fn(self.kitchen, joint))
    def open_door(self, joint):
        close_fn = get_min_limit if 'left' in get_joint_name(self.kitchen, joint) else get_max_limit
        set_joint_position(self.kitchen, joint, close_fn(self.kitchen, joint))
    def add_body(self, name, path, **kwargs):
        # TODO: support obj case
        assert name not in self.body_from_name
        self.body_from_name[name] = load_pybullet(path, **kwargs)
        return name
    def get_body(self, name):
        return self.body_from_name[name]
    def remove_body(self, name):
        raise NotImplementedError()
    def get_name(self, name):
        inverse = {v: k for k, v in self.body_from_name.items()}
        return inverse.get(name, None)
    def destroy(self):
        disconnect()

################################################################################

class Grasp(object):
    def __init__(self, world, body_name, grasp_type, index, grasp_pose, pregrasp_pose, grasp_width):
        self.world = world
        self.body_name = body_name
        self.grasp_type = grasp_type
        self.index = index
        self.grasp_pose = grasp_pose
        self.pregrasp_pose = pregrasp_pose
        self.grasp_width = grasp_width
    def get_attachment(self):
        return Attachment(self.world.robot, self.world.tool_link,
                          self.grasp_pose, self.world.get_body(self.body_name))
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.grasp_type, self.index)


def get_grasps(world, name, grasp_types=['top', 'side']):
    pre_distance = 0.1
    body = world.get_body(name)
    for grasp_type in grasp_types:
        if grasp_type == 'top':
            pre_direction = pre_distance * get_unit_vector([0, 0, 1])
            generator = get_top_grasps(body, under=False, tool_pose=unit_pose(),
                                       grasp_length=FINGER_EXTENT[2] / 2, max_width=np.inf)
        elif grasp_type == 'side':
            pre_direction = pre_distance * get_unit_vector([1, 0, 3])
            generator = get_side_grasps(body, under=False, tool_pose=unit_pose(),
                                        grasp_length=FINGER_EXTENT[2] / 2, max_width=np.inf,
                                        top_offset=FINGER_EXTENT[0] / 2)
        else:
            raise ValueError(grasp_type)
        for i, grasp_pose in enumerate(generator):
            with BodySaver(world.robot):
                grasp_width = close_until_collision(world.robot, world.gripper_joints, bodies=[body])
            pregrasp_pose = multiply(Pose(point=pre_direction), grasp_pose)
            grasp = Grasp(world, name, grasp_type, i, grasp_pose, pregrasp_pose, grasp_width)
            yield grasp
