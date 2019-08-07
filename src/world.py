import numpy as np
import os
from collections import namedtuple

from ikfast.ik import sample_tool_ik
from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.pr2_utils import get_viewcone
from pybullet_tools.utils import connect, add_data_path, load_pybullet, HideOutput, set_point, Point, stable_z, \
    draw_pose, Pose, get_link_name, parent_link_from_joint, child_link_from_joint, read, joints_from_names, \
    joint_from_name, link_from_name, get_link_subtree, get_links, get_joint_limits, aabb_union, get_aabb, get_point, \
    remove_debug, draw_base_limits, get_link_pose, multiply, invert, get_joint_positions, \
    step_simulation, apply_alpha, RED, \
    set_joint_positions, get_configuration, set_joint_position, get_min_limit, get_max_limit, \
    get_joint_name, remove_body, disconnect, get_min_limits, get_max_limits, add_body_name, WorldSaver, \
    is_placed_on_aabb, is_center_on_aabb, Euler, euler_from_quat, quat_from_pose, point_from_pose, get_pose, set_pose, stable_z_on_aabb, \
    set_quat, quat_from_euler, INF, get_difference
from src.issac import load_calibrate_conf
from src.command import State
from src.utils import FRANKA_CARTER, FRANKA_CARTER_PATH, FRANKA_YAML, EVE, EVE_PATH, load_yaml, create_gripper, \
    KITCHEN_PATH, KITCHEN_YAML, USE_TRACK_IK, BASE_JOINTS, get_eve_arm_joints, DEFAULT_ARM, ALL_JOINTS, \
    get_tool_link, custom_limits_from_base_limits, ARMS, CABINET_JOINTS, DRAWER_JOINTS, \
    ALL_SURFACES, compute_surface_aabb, create_surface_attachment, KINECT_DEPTH

DISABLED_FRANKA_COLLISIONS = {
    ('panda_link1', 'chassis_link'),
}

DEFAULT_ARM_CONF = [0.01200158428400755, -0.5697816014289856, 5.6801487517077476e-05,
                    -2.8105969429016113, -0.00025768374325707555, 3.0363450050354004, 0.7410701513290405]

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py#L59
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L52

CABINET_OPEN_ANGLE = 4 * np.pi / 9 # out of np.pi / 2
DRAWER_OPEN_FRACTION = 0.75

Camera = namedtuple('Camera', ['body', 'matrix', 'depth'])

################################################################################

class World(object):
    def __init__(self, robot_name=FRANKA_CARTER, use_gui=True):
        self.task = None
        self.client = connect(use_gui=use_gui)
        add_data_path()
        self.floor = load_pybullet('plane.urdf', fixed_base=True)
        draw_pose(Pose(point=Point(z=1e-2)), length=3)

        self.robot_name = robot_name
        if self.robot_name == FRANKA_CARTER:
            urdf_path, yaml_path = FRANKA_CARTER_PATH, FRANKA_YAML
        elif self.robot_name == EVE:
            urdf_path, yaml_path = EVE_PATH, None
        else:
            raise ValueError(self.robot_name)
        self.robot_yaml = yaml_path if yaml_path is None else load_yaml(yaml_path)
        with HideOutput(enable=True):
            self.robot = load_pybullet(urdf_path)
        #dump_body(self.robot)
        #chassis_pose = get_link_pose(self.robot, link_from_name(self.robot, 'chassis_link'))
        #wheel_pose = get_link_pose(self.robot, link_from_name(self.robot, 'left_wheel_link'))
        #wait_for_user()

        set_point(self.robot, Point(z=stable_z(self.robot, self.floor)))
        self.set_initial_conf()
        self.gripper = create_gripper(self.robot)

        self.kitchen_yaml = load_yaml(KITCHEN_YAML)
        with HideOutput(enable=True):
            self.kitchen = load_pybullet(KITCHEN_PATH, fixed_base=True)
        set_point(self.kitchen, Point(z=stable_z(self.kitchen, self.floor)))

        if USE_TRACK_IK:
            from trac_ik_python.trac_ik import IK # killall -9 rosmaster
            base_link = get_link_name(self.robot, parent_link_from_joint(self.robot, self.arm_joints[0]))
            tip_link = get_link_name(self.robot, child_link_from_joint(self.arm_joints[-1]))
            # limit effort and velocities are required
            # solve_type: Speed, Distance, Manipulation1, Manipulation2
            # TODO: fast solver and slow solver
            self.ik_solver = IK(base_link=str(base_link), tip_link=str(tip_link),
                                timeout=0.01, epsilon=1e-5, solve_type="Speed",
                                urdf_string=read(urdf_path))
            lower, upper = self.ik_solver.get_joint_limits()
            buffer = 0.1*np.ones(len(self.ik_solver.joint_names))
            self.ik_solver.set_joint_limits(lower + buffer, upper - buffer)
        else:
            self.ik_solver = None
        self.body_from_name = {}
        self.path_from_name = {}
        self.custom_limits = {}
        self.base_limits_handles = []
        self.cameras = {}

        self.disabled_collisions = set()
        if self.robot_name == FRANKA_CARTER:
            self.disabled_collisions.update(tuple(link_from_name(self.robot, link) for link in pair)
                                            for pair in DISABLED_FRANKA_COLLISIONS)

        self.carry_conf = Conf(self.robot, self.arm_joints, self.default_conf)
        #self.calibrate_conf = Conf(self.robot, self.arm_joints, load_calibrate_conf(side='left'))
        self.calibrate_conf = Conf(self.robot, self.arm_joints, self.default_conf) # Must differ from carry_conf
        self.special_confs = [self.carry_conf, self.calibrate_conf]
        self.open_gq = Conf(self.robot, self.gripper_joints,
                            get_max_limits(self.robot, self.gripper_joints))
        self.closed_gq = Conf(self.robot, self.gripper_joints,
                              get_min_limits(self.robot, self.gripper_joints))
        self.update_initial()
    def update_initial(self):
        # TODO: store initial poses as well?
        self.update_floor()
        self.update_custom_limits()
        self.initial_saver = WorldSaver()
    def get_initial_state(self):
        # TODO: would be better to explicitly keep the state around
        initial_attachments = []
        self.initial_saver.restore()
        for obj_name in self.movable:
            surface_name = self.get_supporting(obj_name)
            if surface_name is not None:
                initial_attachments.append(create_surface_attachment(self, obj_name, surface_name))
        return State(self, savers=[self.initial_saver],
                     attachments=initial_attachments)

    #########################

    @property
    def base_joints(self):
        return joints_from_names(self.robot, BASE_JOINTS)
    @property
    def arm_joints(self):
        if self.robot_name == EVE:
            return get_eve_arm_joints(self.robot, arm=DEFAULT_ARM)
        return joints_from_names(self.robot, self.robot_yaml['cspace'])
    @property
    def gripper_joints(self):
        if self.robot_yaml is None:
            return []
        return tuple(joint_from_name(self.robot, rule['name'])
                for rule in self.robot_yaml['cspace_to_urdf_rules'])
    @property
    def kitchen_joints(self):
        #return joints_from_names(self.kitchen, self.kitchen_yaml['cspace'])
        return joints_from_names(self.kitchen, filter(
            ALL_JOINTS.__contains__, self.kitchen_yaml['cspace']))
    @property
    def base_link(self):
        return child_link_from_joint(self.base_joints[-1])
    @property
    def tool_link(self):
        return link_from_name(self.robot, get_tool_link(self.robot))
    @property
    def door_links(self):
        door_links = set()
        for joint in self.kitchen_joints:
            door_links.update(get_link_subtree(self.kitchen, joint))
        return door_links
    @property
    def static_obstacles(self):
        # link=None is fine
        # TODO: decompose obstacles
        #return [(self.kitchen, frozenset(get_links(self.kitchen)) - self.door_links)]
        return {(self.kitchen, frozenset([link])) for link in
                set(get_links(self.kitchen)) - self.door_links}
    @property
    def movable(self):
        return set(self.body_from_name) # frozenset?
    @property
    def all_bodies(self):
        return set(self.body_from_name.values()) | {self.robot, self.kitchen}
    @property
    def default_conf(self):
        if self.robot_name == EVE:
            # Eve starts outside of joint limits
            conf = [np.average(get_joint_limits(self.robot, joint)) for joint in self.arm_joints]
            #conf = np.zeros(len(self.arm_joints))
            #conf[3] -= np.pi / 2
            return conf
        return DEFAULT_ARM_CONF
        #conf = np.array(self.robot_yaml['default_q'])
        ##conf[1] += np.pi / 4
        ##conf[3] -= np.pi / 4
        #return conf

    #########################

    # TODO: could perform base motion planning without free joints
    def get_base_conf(self):
        return get_joint_positions(self.robot, self.base_joints)
    def set_base_conf(self, conf):
        set_joint_positions(self.robot, self.base_joints, conf)
    def get_world_aabb(self):
        return aabb_union(get_aabb(body) for body in self.all_bodies)
    def update_floor(self):
        z = stable_z(self.kitchen, self.floor) - get_point(self.floor)[2]
        point = np.array(get_point(self.kitchen)) - np.array([0, 0, z])
        set_point(self.floor, point)
    def update_custom_limits(self, min_extent=0.0):
        #robot_extent = get_aabb_extent(get_aabb(self.robot))
        # Scaling by 0.5 to prevent getting caught in corners
        #min_extent = 0.5 * min(robot_extent[:2]) * np.ones(2) / 2
        full_lower, full_upper = self.get_world_aabb()
        base_limits = (full_lower[:2] - min_extent, full_upper[:2] + min_extent)
        for handle in self.base_limits_handles:
            remove_debug(handle)
        z = get_point(self.floor)[2] + 1e-2
        self.base_limits_handles.extend(draw_base_limits(base_limits, z=z))
        self.custom_limits = custom_limits_from_base_limits(self.robot, base_limits)
        return self.custom_limits
    def solve_inverse_kinematics(self, world_from_tool, use_track_ik=True,
                                 nearby_tolerance=INF, **kwargs):
        if use_track_ik:
            assert self.ik_solver is not None
            init_lower, init_upper = self.ik_solver.get_joint_limits()
            base_link = link_from_name(self.robot, self.ik_solver.base_link)
            world_from_base = get_link_pose(self.robot, base_link)
            tip_link = link_from_name(self.robot, self.ik_solver.tip_link)
            tool_from_tip = multiply(invert(get_link_pose(self.robot, self.tool_link)),
                                     get_link_pose(self.robot, tip_link))
            world_from_tip = multiply(world_from_tool, tool_from_tip)
            base_from_tip = multiply(invert(world_from_base), world_from_tip)
            joints = joints_from_names(self.robot, self.ik_solver.joint_names) # self.ik_solver.link_names
            seed_state = get_joint_positions(self.robot, joints)
            #seed_state = [0.0] * self.ik_solver.number_of_joints
            # TODO: adjust the joint limits here instead of the URDF

            lower, upper = init_lower, init_upper
            if nearby_tolerance < INF:
                tolerance = nearby_tolerance*np.ones(len(joints))
                lower = np.maximum(lower, seed_state - tolerance)
                upper = np.minimum(upper, seed_state + tolerance)
            self.ik_solver.set_joint_limits(lower, upper)

            (x, y, z), (rx, ry, rz, rw) = base_from_tip
            # TODO: can also adjust tolerances
            conf = self.ik_solver.get_ik(seed_state, x, y, z, rx, ry, rz, rw)
            self.ik_solver.set_joint_limits(init_lower, init_upper)
            if conf is None:
                return conf
            #if nearby_tolerance < INF:
            #    print(lower.round(3))
            #    print(upper.round(3))
            #    print(conf)
            #    print(get_difference(seed_state, conf).round(3))
            set_joint_positions(self.robot, joints, conf)
            return get_configuration(self.robot)

        conf = sample_tool_ik(self.robot, world_from_tool, max_attempts=100)
        if conf is None:
            return conf
        set_joint_positions(self.robot, self.arm_joints, conf)
        #wait_for_user()
        return get_configuration(self.robot)
        #return sub_inverse_kinematics(self.robot, self.arm_joints[0], self.tool_link, world_from_tool,
        #                             custom_limits=self.custom_limits, **kwargs)

    #########################

    def set_initial_conf(self):
        set_joint_positions(self.robot, self.base_joints, [2.0, 0, np.pi])
        #for rule in self.robot_yaml['cspace_to_urdf_rules']:  # gripper: max is open
        #    joint = joint_from_name(self.robot, rule['name'])
        #    set_joint_position(self.robot, joint, rule['value'])
        set_joint_positions(self.robot, self.arm_joints, self.default_conf)  # active_task_spaces
        if self.robot_name == EVE:
            for arm in ARMS:
                joints = get_eve_arm_joints(self.robot, arm)[2:4]
                set_joint_positions(self.robot, joints, -0.2*np.ones(len(joints)))
    def set_gripper(self, value):
        positions = value*np.ones(len(self.gripper_joints))
        set_joint_positions(self.robot, self.gripper_joints, positions)
    def close_gripper(self):
        self.closed_gq.assign()
    def open_gripper(self):
        self.open_gq.assign()

    #########################

    def get_door_sign(self, joint):
        return -1 if 'left' in get_joint_name(self.kitchen, joint) else +1
    def closed_conf(self, joint):
        if 'left' in get_joint_name(self.kitchen, joint):
            return get_max_limit(self.kitchen, joint)
        return get_min_limit(self.kitchen, joint)
    def open_conf(self, joint):
        joint_name = get_joint_name(self.kitchen, joint)
        if 'left' in joint_name:
            open_position = get_min_limit(self.kitchen, joint)
        else:
            open_position = get_max_limit(self.kitchen, joint)
        #print(get_joint_name(self.kitchen, joint), get_min_limit(self.kitchen, joint), get_max_limit(self.kitchen, joint))
        # drawers: [0.0, 0.4]
        # left doors: [-1.57, 0.0]
        # right doors: [0.0, 1.57]
        if joint_name in CABINET_JOINTS:
            # TODO: could make fraction of max
            return CABINET_OPEN_ANGLE * open_position / abs(open_position)
        if joint_name in DRAWER_JOINTS:
            return DRAWER_OPEN_FRACTION * open_position
        return open_position
    def close_door(self, joint):
        set_joint_position(self.kitchen, joint, self.closed_conf(joint))
    def open_door(self, joint):
        set_joint_position(self.kitchen, joint, self.open_conf(joint))

    #########################

    def add_camera(self, name, pose, camera_matrix, max_depth=KINECT_DEPTH):
        body = get_viewcone(depth=max_depth, camera_matrix=camera_matrix,
                                          color=apply_alpha(RED, 0.1))
        self.cameras[name] = Camera(body, camera_matrix, max_depth)
        set_pose(body, pose)
        step_simulation()
        return name
    def fix_pose(self, name, pose=None):
        body = self.get_body(name)
        if pose is None:
            pose = get_pose(body)
        # TODO: can also drop in simulation
        x, y, z = point_from_pose(pose)
        roll, pitch, yaw = euler_from_quat(quat_from_pose(pose))
        quat = quat_from_euler(Euler(yaw=yaw))
        set_quat(body, quat)
        surface_name = self.get_supporting(name)
        if surface_name is None:
            return None
        surface_aabb = compute_surface_aabb(self, surface_name)
        new_z = stable_z_on_aabb(body, surface_aabb)
        point = Point(x, y, new_z)
        set_point(body, point)
        # TODO: rotate objects that are symmetrical about xy 180 to ensure the start upright
        print('{}: roll={:.3f}, pitch={:.3f}, z-delta: {:.3f}'.format(
            name, roll, pitch, new_z - z))
        return (point, quat)
    def fix_geometry(self):
        for name in self.movable:
            fixed_pose = self.fix_pose(name)
            if fixed_pose is not None:
                set_pose(self.get_body(name), fixed_pose)
    def get_supporting(self, obj_name):
        # is_placed_on_aabb | is_center_on_aabb
        # Only want to generate stable placements, but can operate on initially unstable ones
        body = self.get_body(obj_name)
        supporting = [surface for surface in ALL_SURFACES if is_center_on_aabb(
            body, compute_surface_aabb(self, surface),
            above_epsilon=5e-2, below_epsilon=5e-2)]
        if len(supporting) != 1:
            print('{} is not supported by a single surface ({})!'.format(obj_name, supporting))
            return None
        [surface_name] = supporting
        return surface_name
    def add_body(self, name, path, **kwargs):
        assert name not in self.body_from_name
        self.path_from_name[name] = path
        self.body_from_name[name] = load_pybullet(path, **kwargs)
        assert self.body_from_name[name] is not None
        add_body_name(self.body_from_name[name], name)
        return name
    def get_body(self, name):
        return self.body_from_name[name]
    def get_body_path(self, name):
        return self.path_from_name[name]
    def get_body_type(self, name):
        filename, _ = os.path.splitext(os.path.basename(self.get_body_path(name)))
        return filename
    def get_name(self, name):
        inverse = {v: k for k, v in self.body_from_name.items()}
        return inverse.get(name, None)
    def remove_body(self, name):
        body = self.get_body(name)
        remove_body(body)
        del self.body_from_name[name]
    def reset(self):
        #remove_all_debug()
        for name in list(self.body_from_name):
            self.remove_body(name)
    def destroy(self):
        disconnect()
