import random
from itertools import cycle

from pybullet_tools.utils import BodySaver, get_sample_fn, set_joint_positions, multiply, invert, get_moving_links, \
    pairwise_collision, uniform_pose_generator, get_movable_joints, wait_for_user, INF
from src.command import Sequence, State, ApproachTrajectory, Detach, AttachGripper
from src.database import load_place_base_poses
from src.stream import PRINT_FAILURES, plan_approach, MOVE_ARM, P_RANDOMIZE_IK, inverse_reachability, FIXED_FAILURES
from src.streams.move import get_gripper_motion_gen
from src.utils import FConf, create_surface_attachment, get_surface_obstacles, iterate_approach_path

def is_approach_safe(world, obj_name, pose, grasp, obstacles):
    assert pose.support is not None
    obj_body = world.get_body(obj_name)
    pose.assign()  # May set the drawer confs as well
    set_joint_positions(world.gripper, get_movable_joints(world.gripper), world.open_gq.values)
    #set_renderer(enable=True)
    for _ in iterate_approach_path(world, pose, grasp, body=obj_body):
        #for link in get_all_links(world.gripper):
        #    set_color(world.gripper, apply_alpha(np.zeros(3)), link)
        #wait_for_user()
        if any(pairwise_collision(world.gripper, obst) # or pairwise_collision(obj_body, obst)
               for obst in obstacles):
            print('Unsafe approach!')
            #wait_for_user()
            return False
    return True

def plan_pick(world, obj_name, pose, grasp, base_conf, obstacles, randomize=True, **kwargs):
    # TODO: check if within database convex hull
    # TODO: flag to check if initially in collision

    obj_body = world.get_body(obj_name)
    pose.assign()
    base_conf.assign()
    world.open_gripper()
    robot_saver = BodySaver(world.robot)
    obj_saver = BodySaver(obj_body)

    if randomize:
        sample_fn = get_sample_fn(world.robot, world.arm_joints)
        set_joint_positions(world.robot, world.arm_joints, sample_fn())
    else:
        world.carry_conf.assign()
    world_from_body = pose.get_world_from_body()
    gripper_pose = multiply(world_from_body, invert(grasp.grasp_pose))  # w_f_g = w_f_o * (g_f_o)^-1
    full_grasp_conf = world.solve_inverse_kinematics(gripper_pose)
    if full_grasp_conf is None:
        if PRINT_FAILURES: print('Grasp kinematic failure')
        return
    moving_links = get_moving_links(world.robot, world.arm_joints)
    robot_obstacle = (world.robot, frozenset(moving_links))
    #robot_obstacle = get_descendant_obstacles(world.robot, child_link_from_joint(world.arm_joints[0]))
    #robot_obstacle = world.robot
    if any(pairwise_collision(robot_obstacle, b) for b in obstacles):
        if PRINT_FAILURES: print('Grasp collision failure')
        #set_renderer(enable=True)
        #wait_for_user()
        #set_renderer(enable=False)
        return
    approach_pose = multiply(world_from_body, invert(grasp.pregrasp_pose))
    approach_path = plan_approach(world, approach_pose,  # attachments=[grasp.get_attachment()],
                                  obstacles=obstacles, **kwargs)
    if approach_path is None:
        if PRINT_FAILURES: print('Approach plan failure')
        return
    if MOVE_ARM:
        aq = FConf(world.robot, world.arm_joints, approach_path[0])
    else:
        aq = world.carry_conf

    gripper_motion_fn = get_gripper_motion_gen(world, **kwargs)
    finger_cmd, = gripper_motion_fn(world.open_gq, grasp.get_gripper_conf())
    attachment = create_surface_attachment(world, obj_name, pose.support)
    objects = [obj_name]
    cmd = Sequence(State(world, savers=[robot_saver, obj_saver],
                         attachments=[attachment]), commands=[
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, approach_path),
        finger_cmd.commands[0],
        Detach(world, attachment.parent, attachment.parent_link, attachment.child),
        AttachGripper(world, obj_body, grasp=grasp),
        ApproachTrajectory(objects, world, world.robot, world.arm_joints, reversed(approach_path)),
    ], name='pick')
    yield (aq, cmd,)

################################################################################

def get_fixed_pick_gen_fn(world, max_attempts=25, collisions=True, **kwargs):

    def gen(obj_name, pose, grasp, base_conf):
        obstacles = world.static_obstacles | get_surface_obstacles(world, pose.support)  # | {obj_body}
        #if not collisions:
        #    obstacles = set()
        if not is_approach_safe(world, obj_name, pose, grasp, obstacles):
            return
        # TODO: increase timeouts if a previously successful value
        # TODO: seed IK using the previous solution
        max_failures = FIXED_FAILURES if world.task.movable_base else INF
        failures = 0
        while failures <= max_failures:
            for i in range(max_attempts):
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pick(world, obj_name, pose, grasp, base_conf, obstacles,
                                            randomize=randomize, **kwargs), None)
                if ik_outputs is not None:
                    print('Fixed pick succeeded after {} attempts'.format(i))
                    yield ik_outputs
                    break  # return
            else:
                if PRINT_FAILURES: print('Fixed pick failure after {} attempts'.format(max_attempts))
                #if not pose.init:
                #    break
                yield None
                failures += 1
    return gen


def get_pick_gen_fn(world, max_attempts=25, collisions=True, learned=True, **kwargs):
    # TODO: sample in the neighborhood of the base conf to ensure robust

    def gen(obj_name, pose, grasp, *args):
        obstacles = world.static_obstacles | get_surface_obstacles(world, pose.support)
        #if not collisions:
        #    obstacles = set()
        if not is_approach_safe(world, obj_name, pose, grasp, obstacles):
            return

        # TODO: check collisions with obj at pose
        gripper_pose = multiply(pose.get_world_from_body(), invert(grasp.grasp_pose)) # w_f_g = w_f_o * (g_f_o)^-1
        if learned:
            base_generator = cycle(load_place_base_poses(world, gripper_pose, pose.support, grasp.grasp_type))
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
                    continue # TODO: could break if not pose.init
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pick(world, obj_name, pose, grasp, base_conf, obstacles,
                                            randomize=randomize, **kwargs), None)
                if ik_outputs is not None:
                    print('Pick succeeded after {} attempts'.format(i))
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Pick failure after {} attempts'.format(max_attempts))
                #if not pose.init: # Might be an intended placement blocked by a drawer
                #    break
                yield None
    return gen
