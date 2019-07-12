import numpy as np

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import connect, add_data_path, load_pybullet, HideOutput, set_point, Point, stable_z, \
    draw_pose, Pose, get_link_name, parent_link_from_joint, child_link_from_joint, read, joints_from_names, \
    joint_from_name, link_from_name, get_link_subtree, get_links, get_joint_limits, aabb_union, get_aabb, get_bodies, \
    get_point, get_aabb_extent, remove_debug, draw_base_limits, get_link_pose, multiply, invert, get_joint_positions, \
    set_joint_positions, get_configuration, sub_inverse_kinematics, set_joint_position, get_min_limit, get_max_limit, \
    get_joint_name, remove_body, disconnect, wait_for_user, get_all_links, get_center_extent, get_pose, \
    read_obj, aabb_from_points, get_aabb_center, get_aabb_extent, dump_body, get_min_limits, \
    get_max_limits, add_body_name
from src.utils import FRANKA_CARTER, FRANKA_CARTER_PATH, FRANKA_YAML, EVE, EVE_PATH, load_yaml, create_gripper, \
    KITCHEN_PATH, KITCHEN_YAML, USE_TRACK_IK, BASE_JOINTS, get_eve_arm_joints, DEFAULT_ARM, ALL_JOINTS, \
    get_tool_link, custom_limits_from_base_limits, ARMS, CABINET_JOINTS, DRAWER_JOINTS

#KITCHEN1 = 'models/kitchen_description/urdf/kitchen_part_right_gen_stl.urdf'
#KITCHEN2 = 'models/kitchen_description/urdf/kitchen_part_right_gen_obj.urdf'
#mesh1 = read_obj('models/kitchen_description/meshes/obj/drawer_blender.obj', decompose=False)
#aabb1 = aabb_from_points(mesh1.vertices)
#mesh2 = read_obj('models/kitchen_description/meshes/obj/drawer_stl.obj', decompose=False)
#aabb2 = aabb_from_points(mesh2.vertices)
#print(get_aabb_center(aabb2) - get_aabb_center(aabb1), get_aabb_extent(aabb2) - get_aabb_extent(aabb1))

DISABLED_COLLISIONS = {
    ('panda_link1', 'chassis_link'),
}

class World(object):
    def __init__(self, robot_name=FRANKA_CARTER, use_gui=True):
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
        #wheel_pose = get_link_pose(self.robot,  link_from_name(self.robot, 'left_wheel_link'))
        #wait_for_user()

        set_point(self.robot, Point(z=stable_z(self.robot, self.floor)))
        #draw_aabb(get_aabb(self.robot))
        #print(self.robot_yaml)
        self.set_initial_conf()
        self.gripper = create_gripper(self.robot)
        #dump_body(self.gripper)

        self.kitchen_yaml = load_yaml(KITCHEN_YAML)
        with HideOutput(enable=True):
            self.kitchen = load_pybullet(KITCHEN_PATH, fixed_base=True)
        # kitchen2 = load_pybullet(KITCHEN2, fixed_base=True)
        # for joint in self.kitchen_joints:
        #     set_joint_position(self.kitchen, joint, self.open_conf(joint))
        #     set_joint_position(kitchen2, joint, self.open_conf(joint))
        # wait_for_user()
        # for link in get_all_links(self.kitchen):
        #     center1, extent1 = get_center_extent(self.kitchen, link=link)
        #     center2, extent2 = get_center_extent(kitchen2, link=link)
        #     print(get_link_name(self.kitchen, link), center2 - center1, extent2 - extent1)
        # wait_for_user()

        #dump_body(self.kitchen)
        #print(self.kitchen_yaml)
        set_point(self.kitchen, Point(z=stable_z(self.kitchen, self.floor)))
        #draw_pose(get_pose(self.kitchen), length=1)

        if USE_TRACK_IK:
            from trac_ik_python.trac_ik import IK # killall -9 rosmaster
            base_link = get_link_name(self.robot, parent_link_from_joint(self.robot, self.arm_joints[0]))
            tip_link = get_link_name(self.robot, child_link_from_joint(self.arm_joints[-1]))
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
        self.custom_limits = {}
        self.base_limits_handles = []
        self.update_custom_limits()
        self.initial_attachments = {}
        self.disabled_collisions = {tuple(link_from_name(self.robot, link) for link in pair)
                                    for pair in DISABLED_COLLISIONS}

        self.carry_conf = Conf(self.robot, self.arm_joints, self.default_conf)
        self.open_gq = Conf(self.robot, self.gripper_joints,
                            get_max_limits(self.robot, self.gripper_joints))
        self.closed_gq = Conf(self.robot, self.gripper_joints,
                              get_min_limits(self.robot, self.gripper_joints))
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
    def default_conf(self):
        if self.robot_yaml is None:
            # Eve starts outside of joint limits
            conf = [np.average(get_joint_limits(self.robot, joint)) for joint in self.arm_joints]
            #conf = np.zeros(len(self.arm_joints))
            #conf[3] -= np.pi / 2
            return conf
        conf = np.array(self.robot_yaml['default_q'])
        #conf[1] += np.pi / 4
        #conf[3] -= np.pi / 4
        return conf
    # TODO: could perform base motion planning without free joints
    def get_base_conf(self):
        return get_joint_positions(self.robot, self.base_joints)
    def set_base_conf(self, conf):
        set_joint_positions(self.robot, self.base_joints, conf)
    def all_bodies(self):
        return self.movable | {self.robot, self.kitchen, self.floor}
    def get_world_aabb(self):
        return aabb_union(get_aabb(body) for body in get_bodies() if body != self.floor)
    def update_floor(self):
        z = stable_z(self.kitchen, self.floor)
        set_point(self.floor, np.array(get_point(self.floor)) - np.array([0, 0, z]))
    def update_custom_limits(self):
        robot_extent = get_aabb_extent(get_aabb(self.robot))
        # Scaling by 0.5 to prevent getting caught in corners
        #min_extent = 0.5 * min(robot_extent[:2]) * np.ones(2) / 2
        min_extent = 0.0
        full_lower, full_upper = self.get_world_aabb()
        base_limits = (full_lower[:2] - min_extent, full_upper[:2] + min_extent)
        for handle in self.base_limits_handles:
            remove_debug(handle)
        z = get_point(self.floor)[2] + 1e-2
        self.base_limits_handles.extend(draw_base_limits(base_limits, z=z))
        self.custom_limits = custom_limits_from_base_limits(self.robot, base_limits)
        return self.custom_limits
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
    def closed_conf(self, joint):
        if 'left' in get_joint_name(self.kitchen, joint):
            return get_max_limit(self.kitchen, joint)
        return get_min_limit(self.kitchen, joint)
    def open_conf(self, joint):
        joint_name = get_joint_name(self.kitchen, joint)
        if 'left' in joint_name:
            #print(get_joint_name(self.kitchen, joint), get_max_limit(self.kitchen, joint), get_min_limit(self.kitchen, joint))
            open_position = get_min_limit(self.kitchen, joint)
        else:
            open_position = get_max_limit(self.kitchen, joint)
        #print(get_joint_name(self.kitchen, joint), get_min_limit(self.kitchen, joint), get_max_limit(self.kitchen, joint))
        # drawers: [0.0, 0.4]
        # doors: [0.0, 1.57]
        if joint_name in CABINET_JOINTS:
            # TODO: could make fraction of max
            return (4*np.pi / 9) * open_position / abs(open_position)
        if joint_name in DRAWER_JOINTS:
            return 0.75 * open_position
        return open_position
    def close_door(self, joint):
        set_joint_position(self.kitchen, joint, self.closed_conf(joint))
    def open_door(self, joint):
        set_joint_position(self.kitchen, joint, self.open_conf(joint))
    def add_body(self, name, path, **kwargs):
        assert name not in self.body_from_name
        self.path_from_name[name] = path
        self.body_from_name[name] = load_pybullet(path, **kwargs)
        assert self.body_from_name[name] is not None
        add_body_name(self.body_from_name[name], name)
        return name
    def get_body(self, name):
        return self.body_from_name[name]
    def remove_body(self, name):
        body = self.get_body(name)
        remove_body(body)
        del self.body_from_name[name]
    def get_name(self, name):
        inverse = {v: k for k, v in self.body_from_name.items()}
        return inverse.get(name, None)
    def reset(self):
        #remove_all_debug()
        for name in list(self.body_from_name):
            self.remove_body(name)
    def destroy(self):
        #if self.ros_core is not None:
        #    self.ros_core.shutdown()
        disconnect()