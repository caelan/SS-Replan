import os

import numpy as np
import yaml
import rospy

from pybullet_tools.pr2_utils import get_top_grasps, get_side_grasps, close_until_collision
from pybullet_tools.utils import connect, HideOutput, load_pybullet, dump_body, set_point, Point, add_data_path, \
    joints_from_names, joint_from_name, set_joint_positions, set_joint_position, get_min_limit, get_max_limit, \
    get_joint_name, Attachment, link_from_name, get_unit_vector, unit_pose, BodySaver, multiply, Pose, disconnect, \
    get_link_descendants, get_link_subtree, get_link_name, get_links, aabb_union, get_aabb, \
    get_bodies, draw_base_limits, wait_for_user, draw_pose, get_link_parent, clone_body, \
    set_color, get_all_links, invert, get_link_pose, set_pose, interpolate_poses, get_pose, \
    LockRenderer, get_sample_fn, get_movable_joints, get_body_name, stable_z, draw_aabb, \
    get_joint_limits, read, sub_inverse_kinematics, child_link_from_joint, parent_link_from_joint, \
    get_configuration, get_joint_positions, randomize, unit_point

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

#KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen.urdf')
#KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen_concave.urdf')
KITCHEN_PATH = os.path.join(MODELS_PATH, 'kitchen_description/urdf/kitchen_part_right_gen_obj.urdf')
#KITCHEN_PATH = '/home/caelan/Programs/robot_kitchen/model/robot_kitchen.sdf'
#KITCHEN_PATH = os.path.join(SRL_PATH, 'packages/kitchen_description/urdf/kitchen_part_right_gen.urdf')
KITCHEN_YAML = os.path.join(SRL_PATH, 'packages/kitchen_description/config/robot_descriptor.yaml')

BASE_JOINTS = ['x', 'y', 'theta']
FRANKA_TOOL_LINK = 'right_gripper'  # right_gripper | panda_wrist_end_pt | panda_forearm_end_pt
# +z: pointing, +y: left finger
FINGER_EXTENT = np.array([0.02, 0.01, 0.02]) # 2cm x 1cm x 2cm

FRANKA_GRIPPER_LINK = 'panda_link7' # panda_link7 | panda_link8 | panda_hand

EVE_PATH = os.path.join(MODELS_PATH, 'eve-model-master/eve/urdf/eve_7dof_arms.urdf')

#CARTER_BASE_LINK = 'carter_base_link'

KITCHEN = 'kitchen'
STOVES = ['range']
COUNTERS = ['hitman_tmp', 'indigo_tmp']
SURFACES = COUNTERS + STOVES

LINK_SHAPE_FROM_JOINT = {
    'baker_joint': ('sektion', 'Cube.bottom.004_Cube.028'),
    'chewie_door_left_joint': ('sektion', 'Cube.bottom.002_Cube.020'),
    'chewie_door_right_joint': ('sektion', 'Cube.bottom_Cube.000'),

    'dagger_door_left_joint': ('dagger', 'Cube.bottom.008_Cube.044'),
    'dagger_door_right_joint': ('dagger', 'Cube.bottom.012_Cube.060'),

    'hitman_drawer_top_joint': ('hitman_drawer_top', 'Cube_Cube.001'),
    'hitman_drawer_bottom_joint': ('hitman_drawer_bottom', 'Cube_Cube.001'),

    #'indigo_door_left_joint': ('indigo_tmp', 'Sektion'), # TODO: extract out just the base
    #'indigo_door_right_joint': ('indigo_tmp', 'Sektion'),
    'indigo_drawer_top_joint': ('indigo_drawer_top', 'Cube_Cube.001'),
    'indigo_drawer_bottom_joint': ('indigo_drawer_bottom', 'Cube_Cube.001'),
}

CABINET_JOINTS = [
    'baker_joint', 'chewie_door_left_joint', 'chewie_door_right_joint',
    'dagger_door_left_joint', 'dagger_door_right_joint',
    #'indigo_door_left_joint', 'indigo_door_right_joint',
] # door
# TODO: cannot actually do any drawers yet due to nonconvex hitman and indigo
DRAWER_JOINTS = [
    #'hitman_drawer_top_joint', #'hitman_drawer_bottom_joint',
    #'indigo_drawer_top_joint', 'indigo_drawer_bottom_joint',
] # drawer

ALL_SURFACES = SURFACES + CABINET_JOINTS + DRAWER_JOINTS

USE_TRACK_IK = True

def get_kitchen_parent(link_name):
    if link_name in LINK_SHAPE_FROM_JOINT:
        return LINK_SHAPE_FROM_JOINT[link_name][0]
    return link_name

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

def get_tool_from_root(robot):
    root_link = link_from_name(robot, get_gripper_link(robot))
    tool_link = link_from_name(robot, get_tool_link(robot))
    return multiply(invert(get_link_pose(robot, tool_link)),
                    get_link_pose(robot, root_link))

def iterate_approach_path(robot, gripper, pose, grasp, body=None):
    root_from_urdf = multiply(invert(get_link_pose(gripper, 0)), get_pose(gripper))
    tool_from_root = get_tool_from_root(robot)
    grasp_pose = multiply(pose.value, invert(grasp.grasp_pose))
    approach_pose = multiply(pose.value, invert(grasp.pregrasp_pose))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root, root_from_urdf))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.grasp_pose))
        yield

def get_gripper_link(robot):
    robot_name = get_body_name(robot)
    if robot_name == FRANKA_CARTER:
        return FRANKA_GRIPPER_LINK
    elif robot_name == EVE:
        #return EVE_GRIPPER_LINK.format(a='l') # TODO: issue copying *.dae
        return EVE_GRIPPER_LINK.format(arm=DEFAULT_ARM)
    raise ValueError(robot_name)

def get_tool_link(robot):
    robot_name = get_body_name(robot)
    if robot_name == FRANKA_CARTER:
        return FRANKA_TOOL_LINK
    elif robot_name == EVE:
        return EVE_TOOL_LINK.format(arm=DEFAULT_ARM)
    raise ValueError(robot_name)

def create_gripper(robot, visual=False):
    #dump_body(robot)
    links = get_link_subtree(robot, link_from_name(robot, get_gripper_link(robot)))
    with LockRenderer():
        gripper = clone_body(robot, links=links, visual=False, collision=True)  # TODO: joint limits
        if not visual:
            for link in get_all_links(gripper):
                set_color(gripper, np.zeros(4), link)
    return gripper

################################################################################

EVE_GRIPPER_LINK = 'qbhand_{arm}_base_link' # qbhand_{arm}_base_link
#EVE_GRIPPER_LINK = '{a}_palm' # Technically the start

#EVE_TOOL_LINK = 'qbhand_{arm}_palm_link'
EVE_TOOL_LINK = 'qbhand_{arm}_tendon_virtual_link'

EVE_HIP_JOINTS = ['j_hip_z', 'j_hip_x', 'j_hip_y']
EVE_ANKLE_JOINTS = ['j_knee_y', 'j_ankle_y', 'j_ankle_x']
EVE_WHEEL_JOINTS = ['j_{a}_wheel_y', 'j_{a}_wheel_y']
EVE_ARM_JOINTS = ['j_{a}_shoulder_y', 'j_{a}_shoulder_x', 'j_{a}_shoulder_z',
                  'j_{a}_elbow_y', 'j_{a}_elbow_z', 'j_{a}_wrist_y', 'j_{a}_wrist_x'] # j_neck_y

FRANKA_CARTER = 'franka_carter'
EVE = 'Eve'
ARMS = ['left', 'right']
DEFAULT_ARM = ARMS[0]

def get_eve_arm_joints(robot, arm):
    name = [j.format(a=arm[0]) for j in EVE_ARM_JOINTS]
    return joints_from_names(robot, name)

class World(object):
    def __init__(self, robot_name=FRANKA_CARTER, use_gui=True):
        self.client = connect(use_gui=use_gui)
        add_data_path()
        self.floor = load_pybullet('plane.urdf', fixed_base=True)
        self.robot_name = robot_name
        if self.robot_name == FRANKA_CARTER:
            urdf_path, yaml_path = FRANKA_CARTER_PATH, FRANKA_YAML
        elif self.robot_name == EVE:
            urdf_path, yaml_path = EVE_PATH, None
        else:
            raise ValueError(self.robot_name)
        with HideOutput(enable=True):
            self.robot = load_pybullet(urdf_path)
        #dump_body(self.robot)
        set_point(self.robot, Point(z=stable_z(self.robot, self.floor)))
        #draw_aabb(get_aabb(self.robot))

        self.robot_yaml = yaml_path if yaml_path is None else load_yaml(yaml_path)
        #print(self.robot_yaml)
        self.set_initial_conf()
        self.gripper = create_gripper(self.robot)
        #dump_body(self.gripper)

        with HideOutput(enable=True):
            self.kitchen = load_pybullet(KITCHEN_PATH, fixed_base=True)
        draw_pose(Pose(point=Point(z=1e-2)), length=3)
        #dump_body(self.kitchen)
        self.kitchen_yaml = load_yaml(KITCHEN_YAML)
        #print(self.kitchen_yaml)
        set_point(self.kitchen, Point(z=1.35))

        if USE_TRACK_IK:
            #import roslaunch
            #import rospy
            #self.ros_core = roslaunch.parent.ROSLaunchParent(
            #    run_id='master',  roslaunch_files=[], is_core=True)
            #self.ros_core.start()
            #rospy.set_param('/robot_description', read(urdf_path))
            from trac_ik_python.trac_ik import IK # killall -9 rosmaster
            base_link = get_link_name(self.robot, parent_link_from_joint(self.robot, self.arm_joints[0]))
            tip_link = get_link_name(self.robot, child_link_from_joint(self.arm_joints[-1]))
            #dump_body(self.robot)
            #print(base_link, tip_link)
            # limit effort and velocities are required
            self.ik_solver = IK(base_link=str(base_link), tip_link=str(tip_link),
                                timeout=0.005, epsilon=1e-5, solve_type="Speed",
                                urdf_string=read(urdf_path))
            #print(self.ik_solver.joint_names, self.ik_solver.link_names)
            # https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/
            #self.ik_solver.set_joint_limits([0.0] * self.ik_solver.number_of_joints, upper_bound)
        else:
            self.ik_solver = None
            self.ros_core = None
        self.body_from_name = {}
        self.path_from_name = {}
        self.custom_limits = compute_custom_base_limits(self)
    @property
    def base_joints(self):
        return joints_from_names(self.robot, BASE_JOINTS)
    @property
    def arm_joints(self):
        if self.robot_yaml is None:
            return get_eve_arm_joints(self.robot, arm=DEFAULT_ARM)
        return joints_from_names(self.robot, self.robot_yaml['cspace'])
    @property
    def gripper_joints(self):
        if self.robot_yaml is None:
            return []
        return [joint_from_name(self.robot, rule['name'])
                for rule in self.robot_yaml['cspace_to_urdf_rules']]
    @property
    def kitchen_joints(self):
        return joints_from_names(self.kitchen, self.kitchen_yaml['cspace'])
    @property
    def tool_link(self):
        return link_from_name(self.robot, get_tool_link(self.robot))
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
        # TODO: decompose obstacles
        #return [(self.kitchen, frozenset(get_links(self.kitchen)) - self.door_links)]
        return {(self.kitchen, frozenset([link])) for link in set(get_links(self.kitchen)) - self.door_links}
    @property
    def movable(self):
        return set(self.body_from_name) # frozenset?
    @property
    def initial_conf(self):
        if self.robot_yaml is None:
            # Eve starts outside of joint limits
            conf = [np.average(get_joint_limits(self.robot, joint)) for joint in self.arm_joints]
            #conf = np.zeros(len(self.arm_joints))
            #conf[3] -= np.pi / 2
            return conf
        conf = np.array(self.robot_yaml['default_q'])
        conf[1] += np.pi / 4
        #conf[3] -= np.pi / 4
        return conf
    def solve_inverse_kinematics(self, world_from_tool, use_track_ik=USE_TRACK_IK, **kwargs):
        if use_track_ik:
            assert self.ik_solver is not None
            base_link = link_from_name(self.robot, self.ik_solver.base_link)
            world_from_base = get_link_pose(self.robot, base_link)
            tip_link = link_from_name(self.robot, self.ik_solver.tip_link)
            tool_from_tip = multiply(invert(get_link_pose(self.robot, self.tool_link)),
                                      get_link_pose(self.robot, tip_link))
            world_from_tip = multiply(world_from_tool, tool_from_tip)
            base_from_tip = multiply(invert(world_from_base), world_from_tip)

            joints = joints_from_names(self.robot, self.ik_solver.joint_names)
            seed_state = get_joint_positions(self.robot, joints)
            #seed_state = [0.0] * self.ik_solver.number_of_joints
            (x, y, z), (rx, ry, rz, rw) = base_from_tip
            # TODO: can also adjust tolerances
            conf = self.ik_solver.get_ik(seed_state, x, y, z, rx, ry, rz, rw)
            if conf is None:
                return conf
            set_joint_positions(self.robot, joints, conf)
            return get_configuration(self.robot)
        return sub_inverse_kinematics(self.robot, self.arm_joints[0], self.tool_link, world_from_tool,
                                     custom_limits=self.custom_limits, **kwargs)
    def set_initial_conf(self):
        set_joint_positions(self.robot, self.base_joints, [2.0, 0, np.pi])
        #for rule in self.robot_yaml['cspace_to_urdf_rules']:  # gripper: max is open
        #    joint = joint_from_name(self.robot, rule['name'])
        #    set_joint_position(self.robot, joint, rule['value'])
        set_joint_positions(self.robot, self.arm_joints, self.initial_conf)  # active_task_spaces
        if self.robot_name == EVE:
            for arm in ARMS:
                joints = get_eve_arm_joints(self.robot, arm)[2:4]
                set_joint_positions(self.robot, joints, -0.2*np.ones(len(joints)))
    def close_gripper(self):
        for joint in self.gripper_joints:
            set_joint_position(self.robot, joint, get_min_limit(self.robot, joint))
    def open_gripper(self):
        for joint in self.gripper_joints:
            set_joint_position(self.robot, joint, get_max_limit(self.robot, joint))
    def closed_conf(self, joint):
        if 'left' in get_joint_name(self.kitchen, joint):
            return get_max_limit(self.kitchen, joint)
        return get_min_limit(self.kitchen, joint)
    def open_conf(self, joint):
        if 'left' in get_joint_name(self.kitchen, joint):
            return get_min_limit(self.kitchen, joint)
        return get_max_limit(self.kitchen, joint)
    def close_door(self, joint):
        set_joint_position(self.kitchen, joint, self.closed_conf(joint))
    def open_door(self, joint):
        set_joint_position(self.kitchen, joint, self.open_conf(joint))
    def add_body(self, name, path, **kwargs):
        # TODO: support obj case
        assert name not in self.body_from_name
        self.path_from_name[name] = path
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
        #if self.ros_core is not None:
        #    self.ros_core.shutdown()
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

TOP_GRASP = 'top'
SIDE_GRASP = 'side'
UNDER_GRASP = 'under' # TODO: for franka_carter
GRASP_TYPES = [TOP_GRASP, SIDE_GRASP]

def get_grasps(world, name, grasp_types=GRASP_TYPES, pre_distance=0.1, **kwargs):
    body = world.get_body(name)
    for grasp_type in grasp_types:
        if grasp_type == TOP_GRASP:
            pre_direction = pre_distance * get_unit_vector([0, 0, 1])
            post_direction = unit_point()

            generator = get_top_grasps(body, under=False, tool_pose=unit_pose(),
                                       grasp_length=FINGER_EXTENT[2] / 2, max_width=np.inf, **kwargs)
        elif grasp_type == SIDE_GRASP:
            x, z = pre_distance * get_unit_vector([3, -1])
            pre_direction = [0, 0, x]
            post_direction = [0, 0, z]
            # Under grasps are actually easier for this robot
            generator = get_side_grasps(body, under=False, tool_pose=unit_pose(),
                                        grasp_length=FINGER_EXTENT[2] / 2, max_width=np.inf,
                                        top_offset=FINGER_EXTENT[0] / 2, **kwargs)
            #generator = grasps[4:]
            rotate_z = Pose(euler=[0, 0, np.pi]) if world.robot_name == FRANKA_CARTER else unit_pose()
            generator = (multiply(rotate_z, grasp) for grasp in generator)
        else:
            raise ValueError(grasp_type)


        for i, grasp_pose in enumerate(randomize(list(generator))):
            with BodySaver(world.robot):
                grasp_width = close_until_collision(world.robot, world.gripper_joints, bodies=[body])
            #if grasp_width is None:
            #    continue
            pregrasp_pose = multiply(Pose(point=pre_direction), grasp_pose, Pose(point=post_direction))
            grasp = Grasp(world, name, grasp_type, i, grasp_pose, pregrasp_pose, grasp_width)
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


def compute_custom_base_limits(world):
    full_aabb = aabb_union(get_aabb(body) for body in get_bodies() if body != world.floor)
    #draw_aabb(full_aabb)
    lower, upper = full_aabb
    base_limits = (lower[:2], upper[:2])
    draw_base_limits(base_limits)
    #wait_for_user()
    return custom_limits_from_base_limits(world.robot, base_limits)
