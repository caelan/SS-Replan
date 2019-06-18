import random
import numpy as np

from itertools import islice

from pybullet_tools.pr2_primitives import Pose, Conf, get_side_grasps
from pybullet_tools.utils import sample_placement, pairwise_collision, multiply, invert, sub_inverse_kinematics, \
    get_joint_positions, BodySaver, get_distance, set_joint_positions, plan_direct_joint_motion, plan_joint_motion, \
    get_custom_limits, all_between, uniform_pose_generator, plan_nonholonomic_motion, link_from_name, get_max_limit, \
    get_extend_fn, joint_from_name, wait_for_user, get_link_subtree, get_link_name, draw_pose, get_link_pose, \
    remove_debug, draw_aabb, get_aabb, unit_point, Euler, quat_from_euler, plan_cartesian_motion, \
    plan_waypoints_joint_motion, INF, set_color, get_links

from utils import get_grasps, SURFACES
from command import Sequence, Trajectory, Attach, Detach, State, DoorTrajectory


BASE_CONSTANT = 1
BASE_VELOCITY = 0.25

# TODO: need to wrap trajectory when executing in simulation or running on the robot

def distance_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY


def move_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY

################################################################################

def get_stable_gen(world, collisions=True, **kwargs):
    obstacles = world.static_obstacles if collisions else []

    def gen(body_name, surface_name):
        body = world.get_body(body_name)
        surface_names = SURFACES if surface_name is None else [surface_name]
        while True:
            surface_link = link_from_name(world.kitchen, random.choice(surface_names))
            body_pose = sample_placement(body, world.kitchen, bottom_link=surface_link)
            if body_pose is None:
                break
            p = Pose(body, body_pose)
            p.assign()
            #print([get_link_name(obst, link) for obst, links in obstacles for link in links
            #       if pairwise_collision(body, (obst, [link]))])
            #for link in get_links(world.kitchen):
            #    if link != 1:
            #        set_color(world.kitchen, np.zeros(4), link=link)
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

def get_ik_fn(world, custom_limits={}, collisions=True, teleport=False, **kwargs):
    obstacles = world.static_obstacles if collisions else []
    resolutions = 0.05 * np.ones(len(world.arm_joints))
    open_conf = [get_max_limit(world.robot, joint) for joint in world.gripper_joints]
    extend_fn = get_extend_fn(world.robot, world.gripper_joints,
                              resolutions=0.01*np.ones(len(world.gripper_joints)))

    def fn(name, pose, grasp, base_conf):
        obj = world.get_body(name)
        #approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        approach_obstacles = obstacles
        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.pregrasp_pose))

        default_conf = world.initial_conf
        #sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        #open_arm(robot, arm)
        set_joint_positions(world.robot, world.arm_joints, default_conf) # default_conf | sample_fn()

        full_grasp_conf = sub_inverse_kinematics(world.robot, world.arm_joints[0], world.tool_link, gripper_pose,
                                          custom_limits=custom_limits)
        if (full_grasp_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles):
            return None
        grasp_conf = get_joint_positions(world.robot, world.arm_joints)

        #grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
        #                                    #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles): # [obj]
            #print('Grasp IK failure', grasp_conf)
            return None

        full_approach_conf = sub_inverse_kinematics(world.robot, world.arm_joints[0], world.tool_link, approach_pose,
                                               custom_limits=custom_limits)
        if (full_approach_conf is None) or any(pairwise_collision(world.robot, b) for b in obstacles + [obj]):
            #print('Approach IK failure', approach_conf)
            return None
        approach_conf = get_joint_positions(world.robot, world.arm_joints)

        attachment = grasp.get_attachment()
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            grasp_path = plan_direct_joint_motion(world.robot, world.arm_joints, grasp_conf,
                                                  attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=False,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(world.robot, world.arm_joints, default_conf)
            # TODO: plan one with attachment placed and one held
            approach_path = plan_joint_motion(world.robot, world.arm_joints, approach_conf,
                                              attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=False,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path

        holding_conf = [grasp.grasp_width] * len(world.gripper_joints)
        finger_path = [open_conf] + list(extend_fn(open_conf, holding_conf))

        aq = Conf(world.robot, world.arm_joints, approach_conf)
        cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[ # , attachments=attachments
            Trajectory(world, world.robot, world.arm_joints, path),
            Trajectory(world, world.robot, world.gripper_joints, finger_path),
            Attach(world, world.robot, world.tool_link, obj),
            Trajectory(world, world.robot, world.arm_joints, reversed(path)),
        ])
        return (aq, cmd,)
    return fn


def get_ir_sampler(world, custom_limits={}, max_attempts=25, collisions=True, learned=False, **kwargs):
    obstacles = world.static_obstacles if collisions else []
    #gripper = problem.get_gripper()

    def gen_fn(name, pose, grasp):
        obj = world.get_body(name)
        pose.assign()
        #approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        #for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
        #    if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
        #        return

        gripper_pose = multiply(pose.value, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = world.initial_conf  # arm_conf(arm, grasp.carry)
        if learned:
            raise NotImplementedError()
            #base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(world.robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(world.robot, world.base_joints, custom_limits)

        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                pose.assign()
                bq = Conf(world.robot, world.base_joints, base_conf)
                bq.assign()
                set_joint_positions(world.robot, world.arm_joints, default_conf)
                if any(pairwise_collision(world.robot, b) for b in obstacles + [obj]):
                    continue
                #print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None
    return gen_fn


def get_ik_ir_gen(world, max_attempts=25, max_successes=1, max_failures=0, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(world, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(world, teleport=teleport, **kwargs)

    def gen(*inputs):
        _, pose, _ = inputs
        ir_generator = ir_sampler(*inputs)
        successes = 0
        failures = 0
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
                if not pose.init and (max_successes < successes):
                    return
                break
            else:
                failures += 1
                if not pose.init and (max_failures < failures):
                    return
                yield None
    return gen

################################################################################

def get_handle_link(world, joint):
    for link in get_link_subtree(world.kitchen, joint):
        if 'handle' in get_link_name(world.kitchen, link):
            return link
    raise RuntimeError()

def get_pull_gen(world, custom_limits={}, collisions=True, teleport=False):
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


        grasp_wayoinpts = plan_cartesian_motion(world.robot, world.arm_joints[0], world.tool_link, tool_path,
                                                custom_limits=custom_limits, pos_tolerance=1e-3)
        if grasp_wayoinpts is None:
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

def get_motion_gen(world, custom_limits={}, collisions=True, teleport=False):
    # TODO: include fluents
    saver = BodySaver(world.robot)
    obstacles = world.static_obstacles if collisions else []

    def fn(bq1, bq2):
        saver.restore()
        bq1.assign()
        if teleport:
            path = [bq1.values, bq2.values]
        else:
            path = plan_nonholonomic_motion(world.robot, bq2.joints, bq2.values, attachments=[],
                                            obstacles=obstacles, custom_limits=custom_limits, self_collisions=False,
                                            restarts=4, iterations=50, smooth=50)
            if path is None:
                print('Failed motion plan!')
                return None
        cmd = Sequence(State(savers=[BodySaver(world.robot)]), commands=[
            Trajectory(world, world.robot, world.base_joints, path),
        ])
        return (cmd,)
    return fn
