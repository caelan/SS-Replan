import random
import numpy as np

from itertools import islice

from pybullet_tools.pr2_primitives import Pose, Conf, get_side_grasps
from pybullet_tools.utils import sample_placement, pairwise_collision, multiply, invert, sub_inverse_kinematics, \
    get_joint_positions, BodySaver, get_distance, set_joint_positions, plan_direct_joint_motion, plan_joint_motion, \
    get_custom_limits, all_between, uniform_pose_generator, plan_nonholonomic_motion, link_from_name, get_max_limit, \
    get_extend_fn, joint_from_name, wait_for_user, get_link_subtree, get_link_name, draw_pose, get_link_pose, \
    remove_debug, draw_aabb, get_aabb, unit_point, Euler, quat_from_euler, plan_cartesian_motion, \
    plan_waypoints_joint_motion, INF, set_color, get_links, get_collision_data, read_obj, \
    draw_mesh, tform_mesh, add_text, point_from_pose, aabb_from_points, get_face_edges, CIRCULAR_LIMITS, \
    get_data_pose, sample_placement_on_aabb, get_sample_fn, get_pose, stable_z_on_aabb, \
    is_placed_on_aabb, spaced_colors, euler_from_quat, quat_from_pose, wrap_angle

from utils import get_grasps, SURFACES, LINK_SHAPE_FROM_JOINT, iterate_approach_path
from command import Sequence, Trajectory, Attach, Detach, State, DoorTrajectory
from database import load_placements, get_surface_reference_pose, load_base_poses


BASE_CONSTANT = 1
BASE_VELOCITY = 0.25
SELF_COLLISIONS = False # TODO: include self-collisions

# TODO: need to wrap trajectory when executing in simulation or running on the robot

def distance_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY


def move_cost_fn(t):
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

def get_descendant_obstacles(kitchen, joint):
    return {(kitchen, frozenset([link]))
            for link in get_link_subtree(kitchen, joint)}

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
            print('IR attempts:', i)
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
        for _ in iterate_approach_path(world.robot, world.gripper, pose, grasp, body=obj):
            #wait_for_user()
            if any(pairwise_collision(world.gripper, b) or pairwise_collision(obj, b)
                   for b in obstacles):
                #print([b for b in obstacles if pairwise_collision(world.gripper, b)])
                #print([b for b in obstacles if pairwise_collision(obj, b)])
                #print(pose, grasp, 'collide!')
                #print(get_pose(obj))
                #wait_for_user()
                return iter([])

        # TODO: check collisions with obj at pose
        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        if learned:
            base_generator = load_base_poses(world, gripper_pose, pose.support, grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(world.robot, gripper_pose)
        pose.assign()
        return inverse_reachability(world, base_generator, obstacles=obstacles, **kwargs)
    return gen_fn

def get_pick_ik_fn(world, randomize=False, collisions=True,
                   switches=False, teleport=True, **kwargs):

    resolutions = 0.05 * np.ones(len(world.arm_joints))
    open_conf = [get_max_limit(world.robot, joint) for joint in world.gripper_joints]
    extend_fn = get_extend_fn(world.robot, world.gripper_joints,
                              resolutions=0.01*np.ones(len(world.gripper_joints)))
    sample_fn = get_sample_fn(world.robot, world.arm_joints)

    def fn(name, pose, grasp, base_conf):
        obj = world.get_body(name)
        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.pregrasp_pose))

        holding_conf = [grasp.grasp_width] * len(world.gripper_joints)
        finger_path = [open_conf] + list(extend_fn(open_conf, holding_conf))
        obstacles = world.static_obstacles | get_door_obstacles(world, pose.support)
        if not collisions:
            obstacles = set()

        # TODO: could search over multiple arm confs
        default_conf = sample_fn() if randomize else world.initial_conf
        pose.assign()
        base_conf.assign()
        #open_arm(robot, arm)
        set_joint_positions(world.robot, world.arm_joints, default_conf) # default_conf | sample_fn()
        full_grasp_conf = world.solve_inverse_kinematics(gripper_pose)
        if (full_grasp_conf is None): # or any(pairwise_collision(world.robot, b) for b in obstacles):
            return None
        grasp_conf = get_joint_positions(world.robot, world.arm_joints)
        if (grasp_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles): # [obj]
            #print('Grasp IK failure', grasp_conf)
            return None

        if switches:
            aq = Conf(world.robot, world.arm_joints, grasp_conf)
            cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[
                Trajectory(world, world.robot, world.arm_joints, [grasp_conf, grasp_conf]),
                #Trajectory(world, world.robot, world.gripper_joints, finger_path),
                Attach(world, world.robot, world.tool_link, obj),
                Trajectory(world, world.robot, world.arm_joints, [grasp_conf, grasp_conf]),
            ])
            return (aq, cmd,)

        full_approach_conf = world.solve_inverse_kinematics(approach_pose)
        if (full_approach_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles | {obj}):
            #print('Approach IK failure', approach_conf)
            return None
        approach_conf = get_joint_positions(world.robot, world.arm_joints)

        attachment = grasp.get_attachment()
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            grasp_path = plan_direct_joint_motion(world.robot, world.arm_joints, grasp_conf,
                                                  attachments=[attachment],
                                                  obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=world.custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(world.robot, world.arm_joints, default_conf)
            # TODO: plan one with attachment placed and one held
            approach_path = plan_joint_motion(world.robot, world.arm_joints, approach_conf,
                                              attachments=[attachment],
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=world.custom_limits, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path

        aq = Conf(world.robot, world.arm_joints, approach_conf)
        cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[
            Trajectory(world, world.robot, world.arm_joints, path),
            Trajectory(world, world.robot, world.gripper_joints, finger_path),
            Attach(world, world.robot, world.tool_link, obj),
            Trajectory(world, world.robot, world.arm_joints, reversed(path)),
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

def get_handle_link(world, joint):
    for link in get_link_subtree(world.kitchen, joint):
        if 'handle' in get_link_name(world.kitchen, link):
            return link
    raise RuntimeError()

def get_pull_gen(world, collisions=True, teleport=False, learned=False):
    grasp_pose = (unit_point(), quat_from_euler(Euler(pitch=np.pi / 2)))
    # TODO: can adjust the position and orientation on the handle

    def gen(joint_name, conf1, conf2):
        if conf1 == conf2:
            return
        door_joint = joint_from_name(world.kitchen, joint_name)
        handle_link = get_handle_link(world, door_joint)
        #conf1.assign()
        door_joints = [door_joint]
        extend_fn = get_extend_fn(world.kitchen, door_joints, resolutions=[0.05])
        door_path = [conf1.values] + list(extend_fn(conf1.values, conf2.values))
        tool_path = []
        set_joint_positions(world.robot, world.base_joints, [0.75, -0.5, np.pi])
        for q in door_path:
            set_joint_positions(world.kitchen, door_joints, q)
            handle_pose = get_link_pose(world.kitchen, handle_link)
            tool_pose = multiply(handle_pose, invert(grasp_pose))
            tool_path.append(tool_pose)
            #handles = draw_pose(handle_pose, length=0.25)
            #handles.extend(draw_aabb(get_aabb(world.kitchen, link=handle_link)))
            #wait_for_user()
            #for handle in handles:
            #    remove_debug(handle)


        grasp_waypoints = plan_cartesian_motion(world.robot, world.arm_joints[0], world.tool_link, tool_path,
                                                custom_limits=world.custom_limits, pos_tolerance=1e-3)
        if grasp_waypoints is None:
            return
        #pull_joint_path = plan_waypoints_joint_motion(combined_joints, combined_waypoints,
        #                                              collision_fn=lambda q: False)


        aq = Conf(world.robot, world.arm_joints, approach_conf)
        cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[ # , attachments=attachments
            Trajectory(world, world.robot, world.arm_joints, path),
            Trajectory(world, world.robot, world.gripper_joints, finger_path),
            DoorTrajectory(world, world.robot, world.arm_joints, path,
                           world.kitchen, door_joints, door_path),
            Trajectory(world, world.robot, world.gripper_joints, reversed(path)),
            Trajectory(world, world.robot, world.gripper_joints, finger_path),
        ])
        return (aq, cmd,)
    return gen

################################################################################

def get_motion_gen(world, collisions=True, teleport=False):
    # TODO: ensure only forward drive?
    saver = BodySaver(world.robot)

    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
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
        cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[
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
