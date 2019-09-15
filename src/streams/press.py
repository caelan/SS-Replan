import random
from itertools import cycle

from pybullet_tools.pr2_utils import get_top_presses
from pybullet_tools.utils import BodySaver, get_sample_fn, set_joint_positions, multiply, invert, get_moving_links, \
    pairwise_collision, link_from_name, get_unit_vector, unit_point, Pose, get_link_pose, \
    uniform_pose_generator, INF
from src.command import Sequence, State, ApproachTrajectory, Wait
from src.stream import plan_approach, MOVE_ARM, inverse_reachability, P_RANDOMIZE_IK, PRINT_FAILURES, FIXED_FAILURES
from src.utils import FConf, APPROACH_DISTANCE, TOOL_POSE, FINGER_EXTENT, Grasp, TOP_GRASP
from src.database import load_pull_base_poses

def get_grasp_presses(world, knob, pre_distance=APPROACH_DISTANCE):
    knob_link = link_from_name(world.kitchen, knob)
    pre_direction = pre_distance * get_unit_vector([0, 0, 1])
    post_direction = unit_point()
    for i, grasp_pose in enumerate(get_top_presses(world.kitchen, link=knob_link,
                                                   tool_pose=TOOL_POSE, top_offset=FINGER_EXTENT[0]/2 + 5e-3)):
        pregrasp_pose = multiply(Pose(point=pre_direction), grasp_pose,
                                 Pose(point=post_direction))
        grasp = Grasp(world, knob, TOP_GRASP, i, grasp_pose, pregrasp_pose)
        yield grasp

def plan_press(world, knob_name, pose, grasp, base_conf, obstacles, randomize=True, **kwargs):
    base_conf.assign()
    world.close_gripper()
    robot_saver = BodySaver(world.robot)

    if randomize:
        sample_fn = get_sample_fn(world.robot, world.arm_joints)
        set_joint_positions(world.robot, world.arm_joints, sample_fn())
    else:
        world.carry_conf.assign()
    gripper_pose = multiply(pose, invert(grasp.grasp_pose))  # w_f_g = w_f_o * (g_f_o)^-1
    #set_joint_positions(world.gripper, get_movable_joints(world.gripper), world.closed_gq.values)
    #set_tool_pose(world, gripper_pose)
    full_grasp_conf = world.solve_inverse_kinematics(gripper_pose)
    #wait_for_user()
    if full_grasp_conf is None:
        # if PRINT_FAILURES: print('Grasp kinematic failure')
        return
    robot_obstacle = (world.robot, frozenset(get_moving_links(world.robot, world.arm_joints)))
    if any(pairwise_collision(robot_obstacle, b) for b in obstacles):
        #if PRINT_FAILURES: print('Grasp collision failure')
        return
    approach_pose = multiply(pose, invert(grasp.pregrasp_pose))
    approach_path = plan_approach(world, approach_pose, obstacles=obstacles, **kwargs)
    if approach_path is None:
        return
    aq = FConf(world.robot, world.arm_joints, approach_path[0]) if MOVE_ARM else world.carry_conf

    #gripper_motion_fn = get_gripper_motion_gen(world, **kwargs)
    #finger_cmd, = gripper_motion_fn(world.open_gq, world.closed_gq)
    objects = []
    cmd = Sequence(State(world, savers=[robot_saver]), commands=[
        #finger_cmd.commands[0],
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, approach_path),
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, reversed(approach_path)),
        #finger_cmd.commands[0].reverse(),
        Wait(world, duration=1.0),
    ], name='press')
    yield (aq, cmd,)

################################################################################

def get_fixed_press_gen_fn(world, max_attempts=25, collisions=True, teleport=False, **kwargs):

    def gen(knob_name, base_conf):
        knob_link = link_from_name(world.kitchen, knob_name)
        pose = get_link_pose(world.kitchen, knob_link)
        presses = cycle(get_grasp_presses(world, knob_name))
        max_failures = FIXED_FAILURES if world.task.movable_base else INF
        failures = 0
        while failures <= max_failures:
            for i in range(max_attempts):
                grasp = next(presses)
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_press(world, knob_name, pose, grasp, base_conf, world.static_obstacles,
                                             randomize=randomize, **kwargs), None)
                if ik_outputs is not None:
                    print('Fixed press succeeded after {} attempts'.format(i))
                    yield ik_outputs
                    break  # return
            else:
                if PRINT_FAILURES: print('Fixed pull failure after {} attempts'.format(max_attempts))
                yield None
                max_failures += 1
    return gen

def get_press_gen_fn(world, max_attempts=50, collisions=True, teleport=False, learned=True, **kwargs):
    def gen(knob_name):
        obstacles = world.static_obstacles
        knob_link = link_from_name(world.kitchen, knob_name)
        pose = get_link_pose(world.kitchen, knob_link)
        #pose = RelPose(world.kitchen, knob_link, init=True)
        presses = cycle(get_grasp_presses(world, knob_name))
        grasp = next(presses)
        gripper_pose = multiply(pose, invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        if learned:
            base_generator = cycle(load_pull_base_poses(world, knob_name))
        else:
            base_generator = uniform_pose_generator(world.robot, gripper_pose)
        safe_base_generator = inverse_reachability(world, base_generator, obstacles=obstacles, **kwargs)
        while True:
            for i in range(max_attempts):
                try:
                    base_conf = next(safe_base_generator)
                except StopIteration:
                    return
                if base_conf is None:
                    yield None
                    continue
                grasp = next(presses)
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_press(world, knob_name, pose, grasp, base_conf, obstacles,
                                             randomize=randomize, **kwargs), None)
                if ik_outputs is not None:
                    print('Press succeeded after {} attempts'.format(i))
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Press failure after {} attempts'.format(max_attempts))
                #if not pose.init:
                #    break
                yield None
    return gen
