import numpy as np
import os
from collections import namedtuple

from ikfast.ik import sample_tool_ik
from pybullet_tools.pr2_utils import get_viewcone
from pybullet_tools.utils import connect, add_data_path, load_pybullet, HideOutput, set_point, Point, stable_z, \
    draw_pose, Pose, get_link_name, parent_link_from_joint, child_link_from_joint, read, joints_from_names, \
    joint_from_name, link_from_name, get_link_subtree, get_links, get_joint_limits, aabb_union, get_aabb, get_point, \
    remove_debug, draw_base_limits, get_link_pose, multiply, invert, get_joint_positions, \
    step_simulation, apply_alpha, approximate_as_prism, BASE_LINK, RED, \
    set_joint_positions, get_configuration, set_joint_position, get_min_limit, get_max_limit, \
    get_joint_name, remove_body, disconnect, get_min_limits, get_max_limits, add_body_name, WorldSaver, \
    is_center_on_aabb, Euler, euler_from_quat, quat_from_pose, point_from_pose, get_pose, set_pose, stable_z_on_aabb, \
    set_quat, quat_from_euler, INF, read_json, set_camera_pose, draw_aabb, \
    disable_gravity
from src.utils import FRANKA_CARTER, FRANKA_CARTER_PATH, FRANKA_YAML, EVE, EVE_PATH, load_yaml, create_gripper, \
    KITCHEN_PATH, KITCHEN_YAML, USE_TRACK_IK, BASE_JOINTS, get_eve_arm_joints, DEFAULT_ARM, ALL_JOINTS, \
    get_tool_link, custom_limits_from_base_limits, ARMS, CABINET_JOINTS, DRAWER_JOINTS, \
    ALL_SURFACES, compute_surface_aabb, KINECT_DEPTH, IKEA_PATH, FConf, are_confs_close
from log_poses import POSES_PATH

DISABLED_FRANKA_COLLISIONS = {
    ('panda_link1', 'chassis_link'),
}

DEFAULT_ARM_CONF = [0.01200158428400755, -0.5697816014289856, 5.6801487517077476e-05,
                    -2.8105969429016113, -0.00025768374325707555, 3.0363450050354004, 0.7410701513290405]

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py#L59
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L52

CABINET_OPEN_ANGLE = 4 * np.pi / 9 # out of np.pi / 2
DRAWER_OPEN_FRACTION = 0.75

#JOINT_LIMITS_BUFFER = 0.0
JOINT_LIMITS_BUFFER = 0.1
# TODO: make sure to regenerate databases upon adjusting

Camera = namedtuple('Camera', ['body', 'matrix', 'depth'])

################################################################################

# table distance +x: 116cm, -y: 353cm (rotated 90 degrees)
# From kitchen world coordinates (chewie bottom right)
TABLE_NAME = 'table'
TABLE_X = 1.16 # meters
TABLE_Y = 3.53 # meters

# +x distance to computer tables: 240cm
COMPUTER_X = 2.40

class World(object):
    def __init__(self, robot_name=FRANKA_CARTER, use_gui=True):
        self.task = None
        self.client = connect(use_gui=use_gui)
        disable_gravity()
        add_data_path()
        set_camera_pose(camera_point=[2, -1.5, 1])
        draw_pose(Pose(), length=3)

        self.kitchen_yaml = load_yaml(KITCHEN_YAML)
        with HideOutput(enable=True):
            self.kitchen = load_pybullet(KITCHEN_PATH, fixed_base=True)

        self.floor = load_pybullet('plane.urdf', fixed_base=True)
        z = stable_z(self.kitchen, self.floor) - get_point(self.floor)[2]
        point = np.array(get_point(self.kitchen)) - np.array([0, 0, z])
        set_point(self.floor, point)

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

        self._initialize_environment()
        self._initialize_ik(urdf_path)
        self.initial_saver = WorldSaver()

        self.body_from_name = {}
        self.path_from_name = {}
        self.custom_limits = {}
        self.base_limits_handles = []
        self.cameras = {}

        self.disabled_collisions = set()
        if self.robot_name == FRANKA_CARTER:
            self.disabled_collisions.update(tuple(link_from_name(self.robot, link) for link in pair)
                                            for pair in DISABLED_FRANKA_COLLISIONS)

        self.carry_conf = FConf(self.robot, self.arm_joints, self.default_conf)
        #self.calibrate_conf = Conf(self.robot, self.arm_joints, load_calibrate_conf(side='left'))
        self.calibrate_conf = FConf(self.robot, self.arm_joints, self.default_conf) # Must differ from carry_conf
        self.special_confs = [self.carry_conf, self.calibrate_conf]
        self.open_gq = FConf(self.robot, self.gripper_joints,
                            get_max_limits(self.robot, self.gripper_joints))
        self.closed_gq = FConf(self.robot, self.gripper_joints,
                              get_min_limits(self.robot, self.gripper_joints))
        self.gripper_confs = [self.open_gq, self.closed_gq]
        self._update_custom_limits()
        self._update_initial()

    def _initialize_environment(self):
        # wall to fridge: 4cm
        # fridge to goal: 1.5cm
        # hitman to range: 3.5cm
        # range to indigo: 3.5cm
        self.environment_bodies = {}
        self.environment_poses = read_json(POSES_PATH)
        root_from_world = get_link_pose(self.kitchen, self.world_link)
        for name, world_from_part in self.environment_poses.items():
            visual_path = os.path.join(IKEA_PATH, '{}.obj'.format(name))
            collision_path = os.path.join(IKEA_PATH, '{}_collision.obj'.format(name))
            mesh_path = None
            for path in [collision_path, visual_path]:
                if os.path.exists(path):
                    mesh_path = path
                    break
            if mesh_path is None:
                continue
            body = load_pybullet(mesh_path, fixed_base=True)
            root_from_part = multiply(root_from_world, world_from_part)
            if name in ['axe', 'dishwasher', 'echo', 'fox', 'golf']:
                (pos, quat) = root_from_part
                # TODO: still not totally aligned
                pos = np.array(pos) + np.array([0, -0.035, 0])  # , -0.005])
                root_from_part = (pos, quat)
            self.environment_bodies[name] = body
            set_pose(body, root_from_part)
        # TODO: release bounding box or convex hull
        # TODO: static object nonconvex collisions

        if TABLE_NAME in self.environment_bodies:
            body = self.environment_bodies[TABLE_NAME]
            _, (w, l, _) = approximate_as_prism(body)
            _, _, z = get_point(body)
            new_pose = Pose(Point(TABLE_X + l / 2, -TABLE_Y, z), Euler(yaw=np.pi / 2))
            set_pose(body, new_pose)

    def _initialize_ik(self, urdf_path):
        if not USE_TRACK_IK:
            self.ik_solver = None
            return
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
        buffer = JOINT_LIMITS_BUFFER*np.ones(len(self.ik_solver.joint_names))
        buffer[-1] *= 2
        self.ik_solver.set_joint_limits(lower + buffer, upper - buffer)

    def _update_initial(self):
        # TODO: store initial poses as well?
        self.initial_saver = WorldSaver()
        self.goal_bq = FConf(self.robot, self.base_joints)
        self.goal_aq = FConf(self.robot, self.arm_joints)
        if are_confs_close(self.goal_aq, self.carry_conf):
            self.goal_aq = self.carry_conf
        self.goal_gq = FConf(self.robot, self.gripper_joints)
        self.initial_confs = [self.goal_bq, self.goal_aq, self.goal_gq]

    @property
    def constants(self):
        return self.special_confs + self.gripper_confs + self.initial_confs

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
    def world_link(self): # for kitchen
        return BASE_LINK
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
                set(get_links(self.kitchen)) - self.door_links} | \
               {(body, None) for body in self.environment_bodies.values()}
    @property
    def movable(self): # movable base
        return set(self.body_from_name) # frozenset?
    @property
    def fixed(self): # fixed base
        return set(self.environment_bodies.values()) | {self.kitchen}
    @property
    def all_bodies(self):
        return self.movable | self.fixed | {self.robot}
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
        return aabb_union(get_aabb(body) for body in self.fixed) # self.all_bodies

    def _update_custom_limits(self, min_extent=0.0):
        #robot_extent = get_aabb_extent(get_aabb(self.robot))
        # Scaling by 0.5 to prevent getting caught in corners
        #min_extent = 0.5 * min(robot_extent[:2]) * np.ones(2) / 2
        world_aabb = self.get_world_aabb()
        full_lower, full_upper = world_aabb
        base_limits = (full_lower[:2] - min_extent, full_upper[:2] + min_extent)
        base_limits[1][0] = COMPUTER_X - min_extent # TODO: robot radius
        base_limits[0][1] += 0.1
        base_limits[1][1] -= 0.1
        for handle in self.base_limits_handles:
            remove_debug(handle)
        self.base_limits_handles = draw_aabb(world_aabb)
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
    def fix_pose(self, name, pose=None):
        body = self.get_body(name)
        if pose is None:
            pose = get_pose(body)
        else:
            set_pose(body, pose)
        # TODO: can also drop in simulation
        x, y, z = point_from_pose(pose)
        roll, pitch, yaw = euler_from_quat(quat_from_pose(pose))
        quat = quat_from_euler(Euler(yaw=yaw))
        set_quat(body, quat)
        surface_name = self.get_supporting(name)
        if surface_name is None:
            return None, None
        surface_aabb = compute_surface_aabb(self, surface_name)
        new_z = stable_z_on_aabb(body, surface_aabb)
        point = Point(x, y, new_z)
        set_point(body, point)
        # TODO: rotate objects that are symmetrical about xy 180 to ensure the start upright
        print('{} error: roll={:.3f}, pitch={:.3f}, z-delta: {:.3f}'.format(
            name, roll, pitch, new_z - z))
        new_pose = (point, quat)
        return new_pose, surface_name

    # def fix_geometry(self):
    #    for name in self.movable:
    #        fixed_pose, _ = self.fix_pose(name)
    #        if fixed_pose is not None:
    #            set_pose(self.get_body(name), fixed_pose)

    #########################

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
