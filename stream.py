import copy
import random
import numpy as np

from itertools import islice

from pybullet_tools.pr2_primitives import Pose, Conf
from pybullet_tools.utils import pairwise_collision, multiply, invert, get_joint_positions, BodySaver, get_distance, set_joint_positions, plan_direct_joint_motion, plan_joint_motion, \
    get_custom_limits, all_between, uniform_pose_generator, plan_nonholonomic_motion, link_from_name, get_max_limit, \
    get_extend_fn, joint_from_name, get_link_subtree, get_link_name, get_link_pose, \
    get_aabb, unit_point, Euler, quat_from_euler, get_collision_data, read_obj, \
    tform_mesh, point_from_pose, aabb_from_points, get_data_pose, sample_placement_on_aabb, get_sample_fn, \
    stable_z_on_aabb, \
    is_placed_on_aabb, euler_from_quat, quat_from_pose, wrap_angle, \
    get_distance_fn, get_unit_vector, unit_quat, wait_for_user

from utils import get_grasps, SURFACES, LINK_SHAPE_FROM_JOINT, iterate_approach_path, \
    set_tool_pose, close_until_collision, get_descendant_obstacles
from command import Sequence, Trajectory, Attach, State, DoorTrajectory
from database import load_placements, get_surface_reference_pose, load_place_base_poses, load_pull_base_poses


BASE_CONSTANT = 1
BASE_VELOCITY = 0.25
SELF_COLLISIONS = False # TODO: include self-collisions

# TODO: need to wrap trajectory when executing in simulation or running on the robot

def base_cost_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY


def trajectory_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY

################################################################################

def compute_surface_aabb(world, surface_name):
    if surface_name in LINK_SHAPE_FROM_JOINT:
        link_name, shape_name = LINK_SHAPE_FROM_JOINT[surface_name]
    else:
        link_name, shape_name = surface_name, None
    surface_link = link_from_name(world.kitchen, link_name)
    surface_pose = get_link_pose(world.kitchen, surface_link)
    if shape_name is None:
        surface_aabb = get_aabb(world.kitchen, surface_link)
    else:
        [data] = get_collision_data(world.kitchen, surface_link)
        local_pose = get_data_pose(data)
        meshes = read_obj(data.filename)
        # colors = spaced_colors(len(meshes))
        # for i, (name, mesh) in enumerate(meshes.items()):
        mesh = meshes[shape_name]
        mesh = tform_mesh(multiply(surface_pose, local_pose), mesh=mesh)
        surface_aabb = aabb_from_points(mesh.vertices)
        #add_text(surface_name, position=surface_aabb[1])
        #draw_mesh(mesh, color=colors[i])
        #wait_for_user()
    # draw_aabb(surface_aabb)
    return surface_aabb


def get_door_obstacles(world, surface_name):
    if surface_name not in LINK_SHAPE_FROM_JOINT:
        return set() # Could just return the link I suppose
    joint = joint_from_name(world.kitchen, surface_name)
    world.open_door(joint)
    # Be careful to call this before each check
    return get_descendant_obstacles(world.kitchen, joint)

################################################################################

def get_stable_gen(world, learned=True, collisions=True, pos_scale=0.01, rot_scale=np.pi/16,
                   z_offset=5e-3, **kwargs):

    def gen(body_name, surface_name):
        body = world.get_body(body_name)
        surface_names = SURFACES if surface_name is None else [surface_name]
        while True:
            selected_name = random.choice(surface_names)
            surface_aabb = compute_surface_aabb(world, selected_name)
            if learned:
                poses = load_placements(world, selected_name)
                if not poses:
                    break
                surface_pose = get_surface_reference_pose(world.kitchen, selected_name)
                body_pose = multiply(surface_pose, random.choice(poses))
                [x, y, _] = point_from_pose(body_pose)
                _, _, yaw = euler_from_quat(quat_from_pose(body_pose))
                dx, dy = np.random.normal(scale=pos_scale, size=2)
                z = stable_z_on_aabb(body, surface_aabb)
                theta = wrap_angle(yaw + np.random.normal(scale=rot_scale))
                #yaw = np.random.uniform(*CIRCULAR_LIMITS)
                quat = quat_from_euler(Euler(yaw=theta))
                body_pose = (x+dx, y+dy, z+z_offset), quat
                # TODO: project onto the surface
            else:
                body_pose = sample_placement_on_aabb(body, surface_aabb, epsilon=z_offset)
                if body_pose is None:
                    break
            p = Pose(body, body_pose, support=selected_name)
            p.assign()
            if not is_placed_on_aabb(body, surface_aabb):
                continue
            obstacles = world.static_obstacles | get_door_obstacles(world, selected_name)
            if not collisions:
                obstacles = set()
            #print([get_link_name(obst[0], *obst[1]) for obst in obstacles
            #       if pairwise_collision(body, obst)])
            #wait_for_user()
            if not any(pairwise_collision(body, obst) for obst in obstacles):
                       #if obst not in {body, surface}):
                yield (p,)
    return gen


def get_grasp_gen(world, collisions=False, randomize=True, **kwargs): # teleport=False,
    def gen(name):
        for grasp in get_grasps(world, name, **kwargs):
            yield (grasp,)
    return gen

################################################################################

def inverse_reachability(world, base_generator, obstacles=[], max_attempts=25, **kwargs):
    default_conf = world.initial_conf  # arm_conf(arm, grasp.carry)
    lower_limits, upper_limits = get_custom_limits(world.robot, world.base_joints, world.custom_limits)
    while True:
        for i, base_conf in enumerate(islice(base_generator, max_attempts)):
            if not all_between(lower_limits, base_conf, upper_limits):
                continue
            #pose.assign()
            bq = Conf(world.robot, world.base_joints, base_conf)
            bq.assign()
            set_joint_positions(world.robot, world.arm_joints, default_conf)
            if any(pairwise_collision(world.robot, b) for b in obstacles): #  + [obj]
                continue
            #print('IR attempts:', i)
            yield (bq,)
            break
        else:
            yield None

def compose_ir_ik(ir_sampler, ik_fn, inputs, max_attempts=25, max_successes=1, max_failures=0, **kwargs):
    successes = 0
    failures = 0
    ir_generator = ir_sampler(*inputs)
    while True:
        for attempt in range(max_attempts):
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            successes += 1
            print('IK attempt:', attempt)
            yield ir_outputs + ik_outputs
            if max_successes < successes:
                return
            break
        else:
            failures += 1
            if max_failures < failures: # pose.init
                return
            yield None

################################################################################

def get_pick_ir_gen(world, collisions=True, learned=True, **kwargs):

    def gen_fn(name, pose, grasp):
        assert pose.support is not None
        obj = world.get_body(name)
        obstacles = world.static_obstacles | get_door_obstacles(world, pose.support)
        if not collisions:
            obstacles = set()
        for _ in iterate_approach_path(world, pose, grasp, body=obj):
            if any(pairwise_collision(world.gripper, b) or pairwise_collision(obj, b)
                   for b in obstacles):
                return iter([])

        # TODO: check collisions with obj at pose
        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        if learned:
            base_generator = load_place_base_poses(world, gripper_pose, pose.support, grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(world.robot, gripper_pose)
        pose.assign()
        return inverse_reachability(world, base_generator, obstacles=obstacles, **kwargs)
    return gen_fn

ARM_RESOLUTION = 0.05
GRIPPER_RESOLUTION = 0.01
DOOR_RESOLUTION = 0.025

def plan_approach(world, approach_pose, obstacles=[], attachments=[],
                  teleport=False, switches_only=False):
    grasp_conf = get_joint_positions(world.robot, world.arm_joints)
    if switches_only:
        return [world.initial_conf, grasp_conf]

    full_approach_conf = world.solve_inverse_kinematics(approach_pose)
    if (full_approach_conf is None) or \
            any(pairwise_collision(world.robot, b) for b in obstacles): # TODO: | {obj}
        # print('Approach IK failure', approach_conf)
        return None
    approach_conf = get_joint_positions(world.robot, world.arm_joints)
    if teleport:
        return [world.initial_conf, approach_conf, grasp_conf]

    resolutions = ARM_RESOLUTION * np.ones(len(world.arm_joints))
    grasp_path = plan_direct_joint_motion(world.robot, world.arm_joints, grasp_conf,
                                          attachments=attachments,
                                          obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                          custom_limits=world.custom_limits, resolutions=resolutions / 2.)
    if grasp_path is None:
        print('Grasp path failure')
        return None
    set_joint_positions(world.robot, world.arm_joints, world.initial_conf)
    # TODO: plan one with attachment placed and one held
    approach_path = plan_joint_motion(world.robot, world.arm_joints, approach_conf,
                                      attachments=attachments,
                                      obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                      custom_limits=world.custom_limits, resolutions=resolutions,
                                      restarts=2, iterations=25, smooth=25)
    if approach_path is None:
        print('Approach path failure')
        return None
    return approach_path + grasp_path

def plan_gripper_path(world, grasp_width, teleport=False):
    open_conf = [get_max_limit(world.robot, joint) for joint in world.gripper_joints]
    extend_fn = get_extend_fn(world.robot, world.gripper_joints,
                              resolutions=GRIPPER_RESOLUTION*np.ones(len(world.gripper_joints)))
    holding_conf = [grasp_width] * len(world.gripper_joints)
    if teleport:
        return [open_conf, holding_conf]
    return [open_conf] + list(extend_fn(open_conf, holding_conf))

def get_pick_ik_fn(world, randomize=False, collisions=True, **kwargs):
    sample_fn = get_sample_fn(world.robot, world.arm_joints)

    def fn(name, pose, grasp, base_conf):
        obj = world.get_body(name)
        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.pregrasp_pose))
        attachment = grasp.get_attachment()

        finger_path = plan_gripper_path(world, grasp.grasp_width, **kwargs)
        obstacles = world.static_obstacles | get_door_obstacles(world, pose.support) # | {obj}
        if not collisions:
            obstacles = set()

        pose.assign()
        base_conf.assign()
        world.open_gripper()
        robot_saver = BodySaver(world.robot)
        obj_saver = BodySaver(obj)

        set_joint_positions(world.robot, world.arm_joints, sample_fn() if randomize else world.initial_conf)
        full_grasp_conf = world.solve_inverse_kinematics(gripper_pose)
        if (full_grasp_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles):
            # print('Grasp IK failure', grasp_conf)
            return None
        approach_path = plan_approach(world, approach_pose, obstacles=obstacles,
                                      attachments=[attachment], **kwargs)
        if approach_path is None:
            return None

        aq = Conf(world.robot, world.arm_joints, world.initial_conf)
        cmd = Sequence(State(savers=[robot_saver, obj_saver]), commands=[
            Trajectory(world, world.robot, world.arm_joints, approach_path),
            Trajectory(world, world.robot, world.gripper_joints, finger_path),
            Attach(world, world.robot, world.tool_link, obj),
            Trajectory(world, world.robot, world.arm_joints, reversed(approach_path)),
        ])
        return (aq, cmd,)
    return fn


def get_pick_gen(world, max_attempts=25, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_pick_ir_gen(world, max_attempts=1, **kwargs)
    ik_fn = get_pick_ik_fn(world, teleport=teleport, **kwargs)

    def gen(*inputs):
        _, pose, _ = inputs
        return compose_ir_ik(ir_sampler, ik_fn, inputs, max_attempts=max_attempts, **kwargs)
    return gen

################################################################################

def get_handle_grasp(world, joint, pre_distance=0.1):
    pre_direction = pre_distance * get_unit_vector([0, 0, 1])

    for link in get_link_subtree(world.kitchen, joint):
        if 'handle' in get_link_name(world.kitchen, link):
            # TODO: can adjust the position and orientation on the handle
            handle_grasp = (unit_point(), quat_from_euler(Euler(roll=np.pi, pitch=np.pi/2)))
            handle_pregrasp = multiply((pre_direction, unit_quat()), handle_grasp)
            return link, handle_grasp, handle_pregrasp
    raise RuntimeError()

def plan_pull(world, door_joint, door_path, handle_path, tool_path, bq,
              randomize=True, collisions=True, teleport=False, max_distance=0.75, **kwargs):
    handle_link, handle_grasp, handle_pregrasp = get_handle_grasp(world, door_joint)
    door_joints = [door_joint]
    obstacles = world.static_obstacles | get_descendant_obstacles(world.kitchen, door_joint)
    if not collisions:
        obstacles = set()
    # TODO: could allow handle collisions

    bq.assign()
    world.open_gripper()
    #door_saver = BodySaver()
    robot_saver = BodySaver(world.robot)
    sample_fn = get_sample_fn(world.robot, world.arm_joints)
    distance_fn = get_distance_fn(world.robot, world.arm_joints)
    set_joint_positions(world.robot, world.arm_joints,
                        sample_fn() if randomize else world.initial_conf)

    arm_path = []
    for i, tool_pose in enumerate(tool_path):
        set_joint_positions(world.kitchen, door_joints, door_path[i])
        full_arm_conf = world.solve_inverse_kinematics(tool_pose)
        # TODO: only check moving links
        if (full_arm_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles):
            # print('Approach IK failure', approach_conf)
            return None
        arm_conf = get_joint_positions(world.robot, world.arm_joints)
        if arm_path and not teleport:
            distance = distance_fn(arm_path[-1], arm_conf)
            # print(distance)
            if max_distance < distance:
                return None
        arm_path.append(arm_conf)
        # wait_for_user()

    approach_paths = []
    for index in [0, -1]:
        set_joint_positions(world.kitchen, door_joints, door_path[index])
        set_joint_positions(world.robot, world.arm_joints, arm_path[index])
        tool_pose = multiply(handle_path[index], invert(handle_pregrasp))
        approach_path = plan_approach(world, tool_pose, obstacles=obstacles,
                                      teleport=teleport, **kwargs)
        if approach_path is None:
            return None
        approach_paths.append(approach_path)

    set_joint_positions(world.kitchen, door_joints, door_path[0])
    set_joint_positions(world.robot, world.arm_joints, arm_path[0])
    grasp_width = close_until_collision(world.robot, world.gripper_joints,
                                        bodies=[(world.kitchen, [handle_link])])
    finger_path = plan_gripper_path(world, grasp_width, teleport=teleport)

    aq = Conf(world.robot, world.arm_joints, world.initial_conf)
    cmd = Sequence(State(savers=[robot_saver]), commands=[
        Trajectory(world, world.robot, world.arm_joints, approach_paths[0]),
        Trajectory(world, world.robot, world.gripper_joints, finger_path),
        DoorTrajectory(world, world.robot, world.arm_joints, arm_path,
                       world.kitchen, door_joints, door_path),
        Trajectory(world, world.robot, world.gripper_joints, reversed(finger_path)),
        Trajectory(world, world.robot, world.arm_joints, reversed(approach_paths[-1])),
    ])
    return (bq, aq, cmd,)


def get_pull_gen(world, teleport=False, learned=True, **kwargs):

    def gen(joint_name, door_conf1, door_conf2):
        if door_conf1 == door_conf2:
            return
        door_joint = joint_from_name(world.kitchen, joint_name)
        door_joints = [door_joint]
        # TODO: could unify with grasp path
        door_extend_fn = get_extend_fn(world.kitchen, door_joints, resolutions=[DOOR_RESOLUTION])
        door_path = [door_conf1.values] + list(door_extend_fn(door_conf1.values, door_conf2.values))
        if teleport:
            door_path = [door_conf1.values, door_conf2.values]

        #door_obstacles = get_descendant_obstacles(world.kitchen, door_joint)
        handle_link, handle_grasp, handle_pregrasp = get_handle_grasp(world, door_joint)
        handle_path = []
        for door_conf in door_path:
            set_joint_positions(world.kitchen, door_joints, door_conf)
            #if any(pairwise_collision(door_obst, obst)
            #       for door_obst, obst in product(door_obstacles, obstacles)):
            #    return
            handle_path.append(get_link_pose(world.kitchen, handle_link))
            # Collide due to adjacency

        tool_path = [multiply(handle_pose, invert(handle_grasp)) for handle_pose in handle_path]
        for i, tool_pose in enumerate(tool_path):
            set_joint_positions(world.kitchen, door_joints, door_path[i])
            set_tool_pose(world, tool_pose) # TODO: open gripper
            #handles = draw_pose(handle_path[i], length=0.25)
            #handles.extend(draw_aabb(get_aabb(world.kitchen, link=handle_link)))
            #wait_for_user()
            #for handle in handles:
            #    remove_debug(handle)
            if any(pairwise_collision(world.gripper, obst) for obst in world.static_obstacles):
                return

        index = int(len(tool_path)/2)
        #index = 0
        target_pose = tool_path[index]
        if learned:
            base_generator = load_pull_base_poses(world, joint_name)
        else:
            base_generator = uniform_pose_generator(world.robot, target_pose)

        for ir_outputs in inverse_reachability(world, base_generator, obstacles=world.static_obstacles):
            # TODO: check door/bq collisions
            if ir_outputs is None:
                yield None
            bq, = ir_outputs
            ik_outputs = plan_pull(world, door_joint, door_path, handle_path, tool_path, bq, **kwargs)
            if ik_outputs is None:
                continue
            yield ik_outputs
    return gen

################################################################################

def get_motion_gen(world, collisions=True, teleport=False):
    # TODO: ensure only forward drive?
    default_saver = BodySaver(world.robot)

    def fn(bq1, bq2, fluents=[]):
        default_saver.restore()
        bq1.assign()
        initial_saver = BodySaver(world.robot)
        obstacles = set(world.static_obstacles)
        attachments = []
        for fluent in fluents:
            predicate, args = fluent[0], fluent[1:]
            if predicate == 'AtAngle'.lower():
                j, a = args
                a.assign()
                obstacles.update(get_descendant_obstacles(a.body, a.joints[0]))
            elif predicate == 'AtPose'.lower():
                b, p = args
                p.assign()
                obstacles.add(world.get_body(b))
            elif predicate == 'AtGrasp'.lower():
                b, g = args
                attachments.append(g.get_attachment())
                attachments[-1].assign()
            else:
                raise NotImplementedError(predicate)
        # TODO: need to collision check with doors otherwise collision

        if not collisions:
            obstacles = set()
        if teleport:
            path = [bq1.values, bq2.values]
        else:
            path = plan_nonholonomic_motion(world.robot, bq2.joints, bq2.values, attachments=attachments,
                                            obstacles=obstacles, custom_limits=world.custom_limits,
                                            self_collisions=False,
                                            restarts=4, iterations=75, smooth=100)
            if path is None:
                print('Failed motion plan!')
                #for bq in [bq1, bq2]:
                #    bq.assign()
                #    wait_for_user()
                return None
        # TODO: could actually plan with all joints as long as we return to the same config
        cmd = Sequence(State(savers=[initial_saver]), commands=[
            Trajectory(world, world.robot, world.base_joints, path),
        ])
        return (cmd,)
    return fn

################################################################################

OPEN = 'open'
CLOSED = 'closed'
DOOR_STATUSES = [OPEN, CLOSED]
JOINT_THRESHOLD = 1e-3

def get_door_test(world):
    def test(joint_name, conf, status):
        [joint] = conf.joints
        [position] = conf.values
        if status == OPEN:
            status_position = world.open_conf(joint)
        elif status == CLOSED:
            status_position = world.closed_conf(joint)
        else:
            raise NotImplementedError(status)
        return abs(position - status_position) <= JOINT_THRESHOLD
    return test

################################################################################

def get_cfree_pose_pose_test(collisions=True, **kwargs):
    def test(o1, p1, o2, p2):
        if not collisions or (o1 == o2):
            return True
        p1.assign()
        p2.assign()
        return not pairwise_collision(p1.body, p2.body)
    return test

def get_cfree_approach_pose_test(world, collisions=True, **kwargs):
    def test(o1, p1, g1, o2, p2):
        if not collisions:
            return True
        return True
    return test

def get_cfree_approach_angle_test(world, collisions=True, **kwargs):
    def test(o1, p1, g1, j, a):
        if not collisions:
            return True
        return True
    return test

################################################################################

def check_collision_free(world, state, sequence, obstacles):
    if not obstacles:
        return True
    # TODO: check door collisions
    state.assign()
    for command in sequence.commands:
        for _ in command.iterate(world, state):
            state.derive()
            for attachment in state.attachments.values():
                if any(pairwise_collision(attachment.child, obst) for obst in obstacles):
                    return False
            if any(pairwise_collision(world.robot, obst) for obst in obstacles):
                return False
    # TODO: just check collisions with moving links
    return True

def get_cfree_traj_pose_test(world, collisions=True, **kwargs):
    def test(at, o2, p2):
        if not collisions:
            return True
        obstacles = {world.get_body(o2)} - at.bodies
        state = copy.copy(at.context)
        p2.assign()
        return check_collision_free(world, state, at, obstacles)
    return test

def get_cfree_traj_angle_test(world, collisions=True, **kwargs):
    def test(at, j, q):
        if not collisions:
            return True
        joint = joint_from_name(world.kitchen, j)
        obstacles = get_descendant_obstacles(world.kitchen, joint) - at.bodies
        state = copy.copy(at.context)
        q.assign()
        return check_collision_free(world, state, at, obstacles)
    return test
