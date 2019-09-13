import numpy as np
import random

from pybullet_tools.utils import approximate_as_cylinder, approximate_as_prism, \
    multiply, invert, BodySaver, Euler, set_pose, wait_for_user, \
    Point, Pose, uniform_pose_generator
from src.stream import plan_approach, MOVE_ARM, inverse_reachability, P_RANDOMIZE_IK, PRINT_FAILURES
from src.command import Sequence, ApproachTrajectory, State, Wait
from src.stream import MOVE_ARM, plan_workspace
from src.utils import FConf, type_from_name, MUSTARD


def pour_path_from_parameter(world, bowl_name, cup_name):
    bowl_body = world.get_body(bowl_name)
    bowl_center, (bowl_d, bowl_h) = approximate_as_cylinder(bowl_body)
    cup_body = world.get_body(cup_name)
    cup_center, (cup_d, _, cup_h) = approximate_as_prism(cup_body)

    #####

    obj_type = type_from_name(cup_name)
    if obj_type in [MUSTARD]:
        initial_pitch = final_pitch = -np.pi
        radius = 0
    else:
        initial_pitch = 0 # different if mustard
        final_pitch = -3 * np.pi / 4
        radius = bowl_d / 2

    #axis_in_cup_center_x = -0.05
    axis_in_cup_center_x = 0 # meters
    #axis_in_cup_center_z = -cup_h/2.
    axis_in_cup_center_z = 0. # meters
    #axis_in_cup_center_z = +cup_h/2.

    # tl := top left | tr := top right
    cup_tl_in_center = np.array([-cup_d/2, 0, cup_h/2])
    cup_tl_in_axis = cup_tl_in_center - Point(z=axis_in_cup_center_z)
    cup_tl_angle = np.math.atan2(cup_tl_in_axis[2], cup_tl_in_axis[0])
    cup_tl_pour_pitch = final_pitch - cup_tl_angle

    cup_radius2d = np.linalg.norm([cup_tl_in_axis])
    pivot_in_bowl_tr = Point(
        x=-(cup_radius2d * np.math.cos(cup_tl_pour_pitch) + 0.01),
        z=(cup_radius2d * np.math.sin(cup_tl_pour_pitch) + 0.03))

    pivot_in_bowl_center = Point(x=radius, z=bowl_h / 2) + pivot_in_bowl_tr
    base_from_pivot = Pose(Point(x=axis_in_cup_center_x, z=axis_in_cup_center_z))

    #####

    assert -np.pi <= final_pitch <= initial_pitch
    pitches = [initial_pitch]
    if final_pitch != initial_pitch:
        pitches = list(np.arange(final_pitch, initial_pitch, np.pi/16)) + pitches
    cup_path_in_bowl = []
    for pitch in pitches:
        rotate_pivot = Pose(euler=Euler(pitch=pitch)) # Can also interpolate directly between start and end quat
        cup_path_in_bowl.append(multiply(Pose(point=bowl_center), Pose(pivot_in_bowl_center),
                                         rotate_pivot, invert(base_from_pivot),
                                         invert(Pose(point=cup_center))))
    return cup_path_in_bowl

def visualize_cartesian_path(body, pose_path):
    for i, pose in enumerate(pose_path):
        set_pose(body, pose)
        print('{}/{}) continue?'.format(i, len(pose_path)))
        wait_for_user()
    #handles = draw_pose(get_pose(body))
    #handles.extend(draw_aabb(get_aabb(body)))
    #print('Finish?')
    #wait_for_user()
    #for h in handles:
    #    remove_debug(h)

def get_fixed_pour_gen_fn(world, max_attempts=50, collisions=True, teleport=False, **kwargs):
    def gen(bowl_name, wp, cup_name, grasp, bq):
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/plan_tools/samplers/pour.py
        if bowl_name == cup_name:
            return
        #attachment = get_grasp_attachment(world, arm, grasp)
        bowl_body = world.get_body(bowl_name)
        #cup_body = world.get_body(cup_name)
        obstacles = (world.static_obstacles | {bowl_body}) if collisions else set()
        cup_path_bowl = pour_path_from_parameter(world, bowl_name, cup_name)
        for _ in range(max_attempts):
            bowl_pose = wp.get_world_from_body()
            rotate_bowl = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
            rotate_cup = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
            cup_path = [multiply(bowl_pose, invert(rotate_bowl), cup_pose_bowl, rotate_cup)
                        for cup_pose_bowl in cup_path_bowl]
            #visualize_cartesian_path(cup_body, cup_path)
            #if cartesian_path_collision(cup_body, cup_path, obstacles + [bowl_body]):
            #    continue
            tool_path = [multiply(p, invert(grasp.grasp_pose)) for p in cup_path]
            # TODO: better looking robot
            # TODO: extra collision test
            # TODO: orientation constraint

            bq.assign()
            grasp.set_gripper()
            world.carry_conf.assign()
            arm_path = plan_workspace(world, tool_path, obstacles, randomize=True) # tilt to upright
            if arm_path is None:
                continue
            assert MOVE_ARM
            aq = FConf(world.robot, world.arm_joints, arm_path[-1])
            robot_saver = BodySaver(world.robot)

            obj_type = type_from_name(cup_name)
            duration = 5.0 if obj_type in [MUSTARD] else 1.0
            objects = [bowl_name, cup_name]
            cmd = Sequence(State(world, savers=[robot_saver]), commands=[
                ApproachTrajectory(objects, world, world.robot, world.arm_joints, arm_path[::-1]),
                Wait(world, duration=duration),
                ApproachTrajectory(objects, world, world.robot, world.arm_joints, arm_path),
            ], name='pour')
            yield (aq, cmd,)
    return gen


def get_pour_gen_fn(world, max_attempts=50, learned=False, **kwargs):
    ik_gen = get_fixed_pour_gen_fn(world, max_attempts=1, **kwargs)

    def gen(bowl_name, wp, cup_name, grasp):
        if bowl_name == cup_name:
            return
        obstacles = world.static_obstacles
        pose = wp.get_world_from_body()
        if learned:
            #base_generator = cycle(load_place_base_poses(world, gripper_pose, pose.support, grasp.grasp_type))
            raise NotImplementedError()
        else:
            base_generator = uniform_pose_generator(world.robot, pose)
        safe_base_generator = inverse_reachability(world, base_generator, obstacles=obstacles, **kwargs)
        while True:
            for i in range(max_attempts):
                try:
                    base_conf, = next(safe_base_generator)
                except StopIteration:
                    return
                #randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(ik_gen(bowl_name, wp, cup_name, grasp, base_conf), None)
                if ik_outputs is not None:
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Pour failure')
                if not pose.init:
                    break
                yield None
    return gen
