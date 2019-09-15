import random
from itertools import cycle

from pybullet_tools.pr2_utils import close_until_collision
from pybullet_tools.utils import multiply, joint_from_name, set_joint_positions, invert, \
    pairwise_collision, BodySaver, uniform_pose_generator, INF
from src.command import ApproachTrajectory, DoorTrajectory, Sequence, State
from src.database import load_pull_base_poses
from src.stream import PRINT_FAILURES, plan_workspace, plan_approach, MOVE_ARM, \
    P_RANDOMIZE_IK, inverse_reachability, compute_door_paths, FIXED_FAILURES
from src.streams.move import get_gripper_motion_gen
from src.utils import get_descendant_obstacles, FConf


def is_pull_safe(world, door_joint, door_plan):
    obstacles = get_descendant_obstacles(world.kitchen, door_joint)
    door_path, handle_path, handle_plan, tool_path = door_plan
    for door_conf in [door_path[0], door_path[-1]]:
        # TODO: check the whole door trajectory
        set_joint_positions(world.kitchen, [door_joint], door_conf)
        # TODO: just check collisions with the base of the robot
        if any(pairwise_collision(world.robot, b) for b in obstacles):
            if PRINT_FAILURES: print('Door start/end failure')
            return False
    return True


def plan_pull(world, door_joint, door_plan, base_conf,
              randomize=True, collisions=True, teleport=False, **kwargs):
    door_path, handle_path, handle_plan, tool_path = door_plan
    handle_link, handle_grasp, handle_pregrasp = handle_plan

    door_obstacles = get_descendant_obstacles(world.kitchen, door_joint) # if collisions else set()
    obstacles = (world.static_obstacles | door_obstacles) # if collisions else set()

    base_conf.assign()
    world.open_gripper()
    world.carry_conf.assign()
    robot_saver = BodySaver(world.robot) # TODO: door_saver?
    if not is_pull_safe(world, door_joint, door_plan):
        return

    arm_path = plan_workspace(world, tool_path, world.static_obstacles,
                              randomize=randomize, teleport=collisions)
    if arm_path is None:
        return
    approach_paths = []
    for index in [0, -1]:
        set_joint_positions(world.kitchen, [door_joint], door_path[index])
        set_joint_positions(world.robot, world.arm_joints, arm_path[index])
        tool_pose = multiply(handle_path[index], invert(handle_pregrasp))
        approach_path = plan_approach(world, tool_pose, obstacles=obstacles, teleport=teleport, **kwargs)
        if approach_path is None:
            return
        approach_paths.append(approach_path)

    if MOVE_ARM:
        aq1 = FConf(world.robot, world.arm_joints, approach_paths[0][0])
        aq2 = FConf(world.robot, world.arm_joints, approach_paths[-1][0])
    else:
        aq1 = world.carry_conf
        aq2 = aq1

    set_joint_positions(world.kitchen, [door_joint], door_path[0])
    set_joint_positions(world.robot, world.arm_joints, arm_path[0])
    grasp_width = close_until_collision(world.robot, world.gripper_joints,
                                        bodies=[(world.kitchen, [handle_link])])
    gripper_motion_fn = get_gripper_motion_gen(world, teleport=teleport, collisions=collisions, **kwargs)
    gripper_conf = FConf(world.robot, world.gripper_joints, [grasp_width] * len(world.gripper_joints))
    finger_cmd, = gripper_motion_fn(world.open_gq, gripper_conf)

    objects = []
    commands = [
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, approach_paths[0]),
        DoorTrajectory(world, world.robot, world.arm_joints, arm_path,
                       world.kitchen, [door_joint], door_path),
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, reversed(approach_paths[-1])),
    ]
    door_path, _, _, _ = door_plan
    sign = world.get_door_sign(door_joint)
    pull = (sign*door_path[0][0] < sign*door_path[-1][0])
    if pull:
        commands.insert(1, finger_cmd.commands[0])
        commands.insert(3, finger_cmd.commands[0].reverse())
    cmd = Sequence(State(world, savers=[robot_saver]), commands, name='pull')
    yield (aq1, aq2, cmd,)

################################################################################

def get_fixed_pull_gen_fn(world, max_attempts=25, collisions=True, teleport=False, **kwargs):

    def gen(joint_name, door_conf1, door_conf2, base_conf):
        #if door_conf1 == door_conf2:
        #    return
        # TODO: check if within database convex hull
        door_joint = joint_from_name(world.kitchen, joint_name)
        obstacles = (world.static_obstacles | get_descendant_obstacles(
            world.kitchen, door_joint)) # if collisions else set()

        base_conf.assign()
        world.carry_conf.assign()
        door_plans = [door_plan for door_plan in compute_door_paths(
            world, joint_name, door_conf1, door_conf2, obstacles, teleport=teleport)
                      if is_pull_safe(world, door_joint, door_plan)]
        if not door_plans:
            print('Unable to open door {} at fixed config'.format(joint_name))
            return
        max_failures = FIXED_FAILURES if world.task.movable_base else INF
        failures = 0
        while failures <= max_failures:
            for i in range(max_attempts):
                door_path = random.choice(door_plans)
                # TracIK is itself stochastic
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pull(world, door_joint, door_path, base_conf,
                                            randomize=randomize, collisions=collisions, teleport=teleport, **kwargs),
                                  None)
                if ik_outputs is not None:
                    print('Fixed pull succeeded after {} attempts'.format(i))
                    yield ik_outputs
                    break  # return
            else:
                if PRINT_FAILURES: print('Fixed pull failure')
                yield None
                failures += 1
    return gen


def get_pull_gen_fn(world, max_attempts=50, collisions=True, teleport=False, learned=True, **kwargs):
    # TODO: could condition pick/place into cabinet on the joint angle
    obstacles = world.static_obstacles
    #if not collisions:
    #    obstacles = set()

    def gen(joint_name, door_conf1, door_conf2, *args):
        if door_conf1 == door_conf2:
            return
        door_joint = joint_from_name(world.kitchen, joint_name)
        door_paths = compute_door_paths(world, joint_name, door_conf1, door_conf2, obstacles, teleport=teleport)
        if not door_paths:
            return
        if learned:
            base_generator = cycle(load_pull_base_poses(world, joint_name))
        else:
            _, _, _, tool_path = door_paths[0]
            index = int(len(tool_path) / 2)  # index = 0
            target_pose = tool_path[index]
            base_generator = uniform_pose_generator(world.robot, target_pose)
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
                door_path = random.choice(door_paths)
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pull(world, door_joint, door_path, base_conf,
                                            randomize=randomize, collisions=collisions, teleport=teleport, **kwargs), None)
                if ik_outputs is not None:
                    print('Pull succeeded after {} attempts'.format(i))
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Pull failure')
                yield None
    return gen