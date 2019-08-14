import numpy as np
import random
from itertools import islice, cycle
from collections import namedtuple

from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import pairwise_collision, multiply, invert, get_joint_positions, BodySaver, get_distance, \
    set_joint_positions, plan_direct_joint_motion, plan_joint_motion, \
    get_custom_limits, all_between, uniform_pose_generator, plan_nonholonomic_motion, link_from_name, get_extend_fn, \
    joint_from_name, get_link_subtree, get_link_name, get_link_pose, \
    Euler, quat_from_euler, set_pose, point_from_pose, sample_placement_on_aabb, get_sample_fn, get_pose, \
    stable_z_on_aabb, euler_from_quat, quat_from_pose, wrap_angle, wait_for_user, \
    Ray, get_distance_fn, get_unit_vector, unit_quat, Point, set_configuration, \
    is_point_in_polygon, grow_polygon, Pose, user_input, get_moving_links, \
    child_link_from_joint, set_renderer, apply_alpha, get_all_links, set_color, \
    get_max_limits, dump_body, get_movable_joints, add_segments, draw_pose, \
    get_difference, INF
from src.command import Sequence, Trajectory, Attach, Detach, State, DoorTrajectory, Detect
from src.database import load_placements, get_surface_reference_pose, load_place_base_poses, \
    load_pull_base_poses, load_forward_placements, load_inverse_placements
from src.utils import get_grasps, iterate_approach_path, APPROACH_DISTANCE, ALL_SURFACES, \
    set_tool_pose, close_until_collision, get_descendant_obstacles, surface_from_name, RelPose, FINGER_EXTENT, create_surface_attachment, \
    compute_surface_aabb, create_relative_pose, Z_EPSILON, get_surface_obstacles, test_supported, \
    get_link_obstacles, ENV_SURFACES, FConf
from src.visualization import GROW_INVERSE_BASE, GROW_FORWARD_RADIUS
from src.inference import SurfaceDist, NUM_PARTICLES
from examples.discrete_belief.run import revisit_mdp_cost, MAX_COST, clip_cost

DETECT_COST = 1.

BASE_CONSTANT = 1. # 1 | 10
BASE_VELOCITY = 0.25
SELF_COLLISIONS = True

PRINT_FAILURES = True
MOVE_ARM = True
ARM_RESOLUTION = 0.05
GRIPPER_RESOLUTION = 0.01
DOOR_RESOLUTION = 0.025
P_RANDOMIZE_IK = 0.5 # 0.0 | 0.5

MAX_CONF_DISTANCE = 0.75
NEARBY_APPROACH = MAX_CONF_DISTANCE
NEARBY_PULL = 0.25

# TODO: TracIK might not be deterministic in which case it might make sense to try a few
# http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/trac_ik/trac_ik_tutorial.html
# http://wiki.ros.org/trac_ik
# https://traclabs.com/projects/trac-ik/
# https://bitbucket.org/traclabs/trac_ik/src/master/
# https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_lib/
# Speed: returns very quickly the first solution found
# Distance: runs for the full timeout_in_secs, then returns the solution that minimizes SSE from the seed
# Manip1: runs for full timeout, returns solution that maximizes sqrt(det(J*J^T))
# Manip2: runs for full timeout, returns solution that minimizes cond(J) = |J|*|J^-1|

# ik_solver.set_joint_limits([0.0]* ik_solver.number_of_joints, upper_bound)

# TODO: need to wrap trajectory when executing in simulation or running on the robot

################################################################################

def base_cost_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY

def trajectory_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY

def compute_detect_cost(prob):
    success_cost = DETECT_COST
    failure_cost = success_cost
    cost = revisit_mdp_cost(success_cost, failure_cost, prob)
    return cost

def detect_cost_fn(rp_dist, rp_sample):
    # TODO: extend to continuous rp_sample controls using densities
    prob = rp_dist.discrete_prob(rp_sample)
    cost = clip_cost(compute_detect_cost(prob))
    print('{}) Detect Prob: {:.3f} | Detect Cost: {:.3f}'.format(
        rp_dist.surface_name, prob, cost))
    return cost

def opt_detect_cost_fn(rp_dist, rp_sample):
    # TODO: prune these surfaces if the cost is already too high
    if isinstance(rp_sample, RelPose):
        # This shouldn't be needed if eager=True
        return detect_cost_fn(rp_dist, rp_sample)
    prob = rp_dist.surface_prob(rp_dist.surface_name)
    #print(rp_dist.surface_name, prob)
    cost = clip_cost(compute_detect_cost(prob))
    print('{}) Opt Detect Prob: {:.3f} | Opt Detect Cost: {:.3f} | Type: {}'.format(
        rp_dist.surface_name, prob, cost, rp_sample.__class__.__name__))
    return cost

################################################################################

# TODO: more general forward kinematics

def get_compute_pose_kin(world):
    def fn(o1, rp, o2, p2):
        if o1 == o2:
            return None
        if isinstance(rp, SurfaceDist):
            p1 = rp.project(lambda x: fn(o1, x, o2, p2)[0])
            return (p1,)
        #if np.allclose(p2.value, unit_pose()):
        #    return (rp,)
        #if np.allclose(rp.value, unit_pose()):
        #    return (p2,)
        # TODO: assert that the links align?
        body = world.get_body(o1)
        p1 = RelPose(body, #reference_body=p2.reference_body, reference_link=p2.reference_link,
                     support=rp.support, confs=(p2.confs + rp.confs),
                     init=(rp.init and p2.init))
        return (p1,)
    return fn

def get_compute_angle_kin(world):
    def fn(o, j, a):
        link = link_from_name(world.kitchen, o) # link not surface
        p = RelPose(world.kitchen, # link,
                    confs=[a], init=a.init)
        return (p,)
    return fn

################################################################################

def get_compute_detect(world, **kwargs):
    obstacles = world.static_obstacles

    def fn(obj_name, pose):
        # TODO: condition that the drawer is open
        for camera_name in world.cameras:
            camera_body, camera_matrix, camera_depth = world.cameras[camera_name]
            camera_pose = get_pose(camera_body)
            camera_point = point_from_pose(camera_pose)
            # TODO: could sample multiple rays around the object
            obj_point = point_from_pose(pose.get_world_from_body())
            if not is_visible_point(camera_matrix, camera_depth, obj_point, camera_pose):
                continue
            ray = Ray(camera_point, obj_point)
            detect = Detect(world, camera_name, obj_name, pose, [ray])
            # TODO: how should doors be handled?
            if obstacles & detect.compute_occluding():
                continue
            #detect.draw()
            #wait_for_user()
            return (detect,)
        return None
    return fn

def move_occluding(world, ray, obj_name):
    if obj_name is None:
        movable = world.movable - {ray.name}
    else:
        world.set_base_conf([-5.0, 0, 0])
        movable = world.movable - {ray.name, obj_name}
    # Prevent obstruction by other objects
    for name in movable:
        set_pose(world.get_body(name), Pose(Point(z=-5.0)))

def get_ofree_ray_pose_test(world, **kwargs):
    def test(ray, obj_name, pose):
        if ray.name == obj_name:
            return True
        # TODO: some of the rays clip the top of objects. Might be related to the slight z translation
        ray.pose.assign()
        pose.assign()
        body = world.get_body(ray.name)
        obstacles = get_link_obstacles(world, obj_name)
        if any(pairwise_collision(body, obst) for obst in obstacles):
            return False
        move_occluding(world, ray, obj_name)
        #ray.draw()
        #wait_for_user()
        return not obstacles & ray.compute_occluding()
    return test

def get_ofree_ray_grasp_test(world, **kwargs):
    def test(ray, bconf, aconf, obj_name, grasp):
        if ray.name == obj_name:
            return True
        bconf.assign()
        aconf.assign()
        ray.pose.assign()
        if obj_name is not None:
            grasp.assign()
            obstacles = get_link_obstacles(world, obj_name)
        else:
            obstacles = get_descendant_obstacles(world.robot)
        move_occluding(world, ray, obj_name)
        #ray.draw()
        #wait_if_unlocked()
        return not obstacles & ray.compute_occluding()
    return test

def get_sample_belief_gen(world, min_prob=1. / NUM_PARTICLES, # TODO: relative instead?
                          mlo=False, **kwargs):

    def gen(obj_name, pose_dist, surface_name):
        valid_samples = { }
        for rp in pose_dist.dist.support():
            prob = pose_dist.discrete_prob(rp)
            cost = detect_cost_fn(pose_dist, rp)
            if (min_prob <= prob) and (cost <= MAX_COST):
                valid_samples[rp] = prob
                #print(obj_name, surface_name, prob)
                #user_input('Continue?')
        if not valid_samples:
            return
        # TODO: open the door to the full extend to see if visible
        if mlo:
            rp = max(valid_samples, key=valid_samples.__getitem__)
            yield (rp,)
        else:
            # TODO: random sample without replacement
            for rp in sorted(valid_samples, key=valid_samples.__getitem__, reverse=True):
                yield (rp,)
    return gen

################################################################################

def get_test_near_pose(world, **kwargs):
    base_from_objects = grow_polygon(map(point_from_pose, load_forward_placements(world)), radius=GROW_FORWARD_RADIUS)
    vertices_from_surface = {}
    # TODO: alternatively, distance to hull

    def test(object_name, pose, base_conf):
        if object_name in ALL_SURFACES:
            surface_name = object_name
            if surface_name not in vertices_from_surface:
                vertices_from_surface[surface_name] = grow_polygon(
                    map(point_from_pose, load_inverse_placements(world, surface_name)), radius=GROW_INVERSE_BASE)
            if not vertices_from_surface[surface_name]:
                return False
            base_conf.assign()
            pose.assign()
            surface = surface_from_name(surface_name)
            world_from_surface = get_link_pose(world.kitchen, link_from_name(world.kitchen, surface.link))
            world_from_base = get_link_pose(world.robot, world.base_link)
            surface_from_base = multiply(invert(world_from_surface), world_from_base)
            #result = is_point_in_polygon(point_from_pose(surface_from_base), vertices_from_surface[surface_name])
            #if not result:
            #    draw_pose(surface_from_base)
            #    points = [Point(x, y, 0) for x, y, in vertices_from_surface[surface_name]]
            #    add_segments(points, closed=True)
            #    wait_for_user()
            return is_point_in_polygon(point_from_pose(surface_from_base), vertices_from_surface[surface_name])
        else:
            if not base_from_objects:
                return False
            base_conf.assign()
            pose.assign()
            world_from_base = get_link_pose(world.robot, world.base_link)
            world_from_object = pose.get_world_from_body()
            base_from_object = multiply(invert(world_from_base), world_from_object)
            return is_point_in_polygon(point_from_pose(base_from_object), base_from_objects)
    return test

def get_test_near_joint(world, **kwargs):
    vertices_from_joint = {}

    def test(joint_name, base_conf):
        if joint_name not in vertices_from_joint:
            base_confs = list(load_pull_base_poses(world, joint_name))
            vertices_from_joint[joint_name] = grow_polygon(base_confs, radius=GROW_INVERSE_BASE)
        if not vertices_from_joint[joint_name]:
            return False
        # TODO: can't open hitman_drawer_top_joint any more
        # Likely due to conservative carter geometry
        base_conf.assign()
        base_point = point_from_pose(get_link_pose(world.robot, world.base_link))
        return is_point_in_polygon(base_point[:2], vertices_from_joint[joint_name])
    return test

################################################################################

def get_stable_gen(world, max_attempts=100,
                   learned=True, collisions=True,
                   pos_scale=0.01, rot_scale=np.pi/16,
                   z_offset=Z_EPSILON, **kwargs):

    # TODO: remove fixed collisions with contained surfaces
    # TODO: place where currently standing
    def gen(obj_name, surface_name):
        obj_body = world.get_body(obj_name)
        surface_body = world.kitchen
        if surface_name in ENV_SURFACES:
            surface_body = world.environment_bodies[surface_name]
        surface_aabb = compute_surface_aabb(world, surface_name)
        learned_poses = None
        if learned:
            learned_poses = load_placements(world, surface_name) # TODO: GROW_PLACEMENT
            if not learned_poses:
                return
        while True:
            for _ in range(max_attempts):
                if learned:
                    surface_pose_world = get_surface_reference_pose(surface_body, surface_name)
                    sampled_pose_surface = multiply(surface_pose_world, random.choice(learned_poses))
                    [x, y, _] = point_from_pose(sampled_pose_surface)
                    _, _, yaw = euler_from_quat(quat_from_pose(sampled_pose_surface))
                    dx, dy = np.random.normal(scale=pos_scale, size=2)
                    # TODO: avoid reloading
                    z = stable_z_on_aabb(obj_body, surface_aabb)
                    theta = wrap_angle(yaw + np.random.normal(scale=rot_scale))
                    #yaw = np.random.uniform(*CIRCULAR_LIMITS)
                    quat = quat_from_euler(Euler(yaw=theta))
                    body_pose_world = ([x+dx, y+dy, z+z_offset], quat)
                    # TODO: project onto the surface
                else:
                    # TODO: halton sequence
                    # unit_generator(d, use_halton=True)
                    body_pose_world = sample_placement_on_aabb(obj_body, surface_aabb, epsilon=z_offset)
                    if body_pose_world is None:
                        continue # return?
                set_pose(obj_body, body_pose_world)
                # TODO: make sure the surface is open when doing this
                if test_supported(world, obj_body, surface_name, collisions=collisions):
                    rp = create_relative_pose(world, obj_name, surface_name)
                    yield (rp,)
                    break
            else:
                yield None
    return gen

def get_nearby_stable_gen(world, max_attempts=25, **kwargs):
    stable_gen = get_stable_gen(world, **kwargs)
    test_near_pose = get_test_near_pose(world, **kwargs)
    compute_pose_kin = get_compute_pose_kin(world)

    def gen(obj_name, surface_name, pose2, base_conf):
        #base_conf.assign()
        #pose2.assign()
        while True:
            for rel_pose, in islice(stable_gen(obj_name, surface_name), max_attempts):
                pose1, = compute_pose_kin(obj_name, rel_pose, surface_name, pose2)
                if test_near_pose(obj_name, pose1, base_conf):
                    yield (pose1, rel_pose)
                    break
            else:
                yield None
    return gen

def get_grasp_gen(world, collisions=False, randomize=True, **kwargs): # teleport=False,
    # TODO: produce carry arm confs here
    def gen(name, grasp_type):
        for grasp in get_grasps(world, name, grasp_types=[grasp_type], **kwargs):
            yield (grasp,)
    return gen

################################################################################

def inverse_reachability(world, base_generator, obstacles=set(),
                         max_attempts=50, min_distance=0.01, **kwargs):
    lower_limits, upper_limits = get_custom_limits(
        world.robot, world.base_joints, world.custom_limits)
    while True:
        for i, base_conf in enumerate(islice(base_generator, max_attempts)):
            if not all_between(lower_limits, base_conf, upper_limits):
                continue
            # TODO: account for doors and placed-object collisions here
            #pose.assign()
            bq = FConf(world.robot, world.base_joints, base_conf)
            bq.assign()
            for conf in world.special_confs:
                # TODO: ensure the end-effector is visible at the calibrate_conf
                # Could even sample a special visible conf for this base_conf
                conf.assign()
                if any(pairwise_collision(world.robot, b, max_distance=min_distance) for b in obstacles):
                    break
            else:
                # print('IR attempts:', i)
                yield (bq,)
                break
        else:
            if PRINT_FAILURES: print('Failed after {} IR attempts:'.format(max_attempts))
            return
            #yield None # Break or yield none?

def plan_approach(world, approach_pose, attachments=[], obstacles=set(),
                  teleport=False, switches_only=False,
                  approach_path=not MOVE_ARM, **kwargs):
    # TODO: use velocities in the distance function
    distance_fn = get_distance_fn(world.robot, world.arm_joints)
    aq = world.carry_conf
    grasp_conf = get_joint_positions(world.robot, world.arm_joints)
    if switches_only:
        return [aq.values, grasp_conf]

    # TODO: could extract out collision function
    # TODO: track the full approach motion
    full_approach_conf = world.solve_inverse_kinematics(
        approach_pose, nearby_tolerance=NEARBY_APPROACH)
    if full_approach_conf is None: # TODO: | {obj}
        if PRINT_FAILURES: print('Pregrasp kinematic failure')
        return None
    moving_links = get_moving_links(world.robot, world.arm_joints)
    robot_obstacle = (world.robot, frozenset(moving_links))
    #robot_obstacle = world.robot
    if any(pairwise_collision(robot_obstacle, b) for b in obstacles): # TODO: | {obj}
        if PRINT_FAILURES: print('Pregrasp collision failure')
        return None
    approach_conf = get_joint_positions(world.robot, world.arm_joints)
    if teleport:
        return [aq.values, approach_conf, grasp_conf]
    distance = distance_fn(grasp_conf, approach_conf)
    if MAX_CONF_DISTANCE < distance:
        if PRINT_FAILURES: print('Pregrasp proximity failure (distance={:.5f})'.format(distance))
        return None

    resolutions = ARM_RESOLUTION * np.ones(len(world.arm_joints))
    grasp_path = plan_direct_joint_motion(world.robot, world.arm_joints, grasp_conf,
                                          attachments=attachments, obstacles=obstacles,
                                          self_collisions=SELF_COLLISIONS,
                                          disabled_collisions=world.disabled_collisions,
                                          custom_limits=world.custom_limits, resolutions=resolutions / 4.)
    if grasp_path is None:
        if PRINT_FAILURES: print('Pregrasp path failure')
        return None
    if not approach_path:
        return grasp_path
    # TODO: plan one with attachment placed and one held
    # TODO: can still use this as a witness that the conf is reachable
    aq.assign()
    approach_path = plan_joint_motion(world.robot, world.arm_joints, approach_conf,
                                      attachments=attachments,
                                      obstacles=obstacles,
                                      self_collisions=SELF_COLLISIONS,
                                      disabled_collisions=world.disabled_collisions,
                                      custom_limits=world.custom_limits, resolutions=resolutions,
                                      restarts=2, iterations=25, smooth=25)
    if approach_path is None:
        if PRINT_FAILURES: print('Approach path failure')
        return None
    return approach_path + grasp_path

################################################################################

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
        return
    if MOVE_ARM:
        aq = FConf(world.robot, world.arm_joints, approach_path[0])
    else:
        aq = world.carry_conf

    gripper_motion_fn = get_gripper_motion_gen(world, **kwargs)
    finger_cmd, = gripper_motion_fn(world.open_gq, grasp.get_gripper_conf())
    attachment = create_surface_attachment(world, obj_name, pose.support)
    cmd = Sequence(State(world, savers=[robot_saver, obj_saver],
                         attachments=[attachment]), commands=[
        Trajectory(world, world.robot, world.arm_joints, approach_path),
        finger_cmd.commands[0],
        Detach(world, attachment.parent, attachment.parent_link, attachment.child),
        Attach(world, world.robot, world.tool_link, obj_body, grasp=grasp),
        Trajectory(world, world.robot, world.arm_joints, reversed(approach_path)),
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
        for i in range(max_attempts):
            # TracIK is itself stochastic
            randomize = (random.random() < P_RANDOMIZE_IK)
            ik_outputs = next(plan_pick(world, obj_name, pose, grasp, base_conf, obstacles,
                                        randomize=randomize, **kwargs), None)
            if ik_outputs is not None:
                yield ik_outputs
                return
        if PRINT_FAILURES: print('Fixed pick failure')
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
                    base_conf, = next(safe_base_generator)
                except StopIteration:
                    return
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pick(world, obj_name, pose, grasp, base_conf, obstacles,
                                            randomize=randomize, **kwargs), None)
                if ik_outputs is not None:
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Pick failure')
                yield None
    return gen

################################################################################

HandleGrasp = namedtuple('HandleGrasp', ['link', 'handle_grasp', 'handle_pregrasp'])
DoorPath = namedtuple('DoorPath', ['link_path', 'handle_path', 'handle_grasp', 'tool_path'])

def get_handle_grasps(world, joint, pre_distance=APPROACH_DISTANCE):
    pre_direction = pre_distance * get_unit_vector([0, 0, 1])
    #half_extent = 1.0*FINGER_EXTENT[2] # Collides
    half_extent = 1.1*FINGER_EXTENT[2]

    grasps = []
    for link in get_link_subtree(world.kitchen, joint):
        if 'handle' in get_link_name(world.kitchen, link):
            # TODO: can adjust the position and orientation on the handle
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/kitchen_poses.py
            for yaw in [0, np.pi]: # yaw=0 DOESN'T WORK WITH LULA
                handle_grasp = (Point(z=-half_extent), quat_from_euler(Euler(roll=np.pi, pitch=np.pi/2, yaw=yaw)))
                handle_pregrasp = multiply((pre_direction, unit_quat()), handle_grasp)
                grasps.append(HandleGrasp(link, handle_grasp, handle_pregrasp))
    return grasps

def compute_door_paths(world, joint_name, door_conf1, door_conf2, obstacles, teleport=False):
    door_paths = []
    if door_conf1 == door_conf2:
        return door_paths
    door_joint = joint_from_name(world.kitchen, joint_name)
    door_joints = [door_joint]
    # TODO: could unify with grasp path
    door_extend_fn = get_extend_fn(world.kitchen, door_joints, resolutions=[DOOR_RESOLUTION])
    door_path = [door_conf1.values] + list(door_extend_fn(door_conf1.values, door_conf2.values))
    if teleport:
        door_path = [door_conf1.values, door_conf2.values]

    # door_obstacles = get_descendant_obstacles(world.kitchen, door_joint)
    for handle_grasp in get_handle_grasps(world, door_joint):
        link, grasp, pregrasp = handle_grasp
        handle_path = []
        for door_conf in door_path:
            set_joint_positions(world.kitchen, door_joints, door_conf)
            # if any(pairwise_collision(door_obst, obst)
            #       for door_obst, obst in product(door_obstacles, obstacles)):
            #    return
            handle_path.append(get_link_pose(world.kitchen, link))
            # Collide due to adjacency

        # TODO: check pregrasp path as well
        # TODO: check gripper self-collisions with the robot
        set_configuration(world.gripper, world.open_gq.values)
        tool_path = [multiply(handle_pose, invert(grasp))
                     for handle_pose in handle_path]
        for i, tool_pose in enumerate(tool_path):
            set_joint_positions(world.kitchen, door_joints, door_path[i])
            set_tool_pose(world, tool_pose)
            # handles = draw_pose(handle_path[i], length=0.25)
            # handles.extend(draw_aabb(get_aabb(world.kitchen, link=link)))
            # wait_for_user()
            # for handle in handles:
            #    remove_debug(handle)
            if any(pairwise_collision(world.gripper, obst) for obst in obstacles):
                break
        else:
            door_paths.append(DoorPath(door_path, handle_path, handle_grasp, tool_path))
    return door_paths

def is_pull_safe(world, door_joint, door_plan, obstacles):
    door_path, handle_path, handle_plan, tool_path = door_plan
    for door_conf in [door_path[0], door_path[-1]]:
        # TODO: check the whole door trajectory
        set_joint_positions(world.kitchen, [door_joint], door_conf)
        if any(pairwise_collision(world.robot, b) for b in obstacles):
            if PRINT_FAILURES: print('Door start/end failure')
            return False
    return True

def plan_pull(world, door_joint, door_plan, base_conf,
              randomize=True, collisions=True, teleport=False, **kwargs):
    door_path, handle_path, handle_plan, tool_path = door_plan
    handle_link, handle_grasp, handle_pregrasp = handle_plan
    # TODO: could push if the goal is to be fully closed

    door_obstacles = get_descendant_obstacles(world.kitchen, door_joint) # if collisions else set()
    obstacles = (world.static_obstacles | door_obstacles) # if collisions else set()

    base_conf.assign()
    world.open_gripper()
    world.carry_conf.assign()
    robot_saver = BodySaver(world.robot) # TODO: door_saver?
    if not is_pull_safe(world, door_joint, door_plan, obstacles):
        return

    # Assuming that pairs of fixed things aren't in collision at this point
    moving_links = get_moving_links(world.robot, world.arm_joints)
    robot_obstacle = (world.robot, frozenset(moving_links))
    distance_fn = get_distance_fn(world.robot, world.arm_joints)
    if randomize:
        sample_fn = get_sample_fn(world.robot, world.arm_joints)
        set_joint_positions(world.robot, world.arm_joints, sample_fn())
    else:
        world.carry_conf.assign()
    arm_path = []
    for i, tool_pose in enumerate(tool_path):
        set_joint_positions(world.kitchen, [door_joint], door_path[i])
        tolerance = INF if i == 0 else NEARBY_PULL
        full_arm_conf = world.solve_inverse_kinematics(tool_pose, nearby_tolerance=tolerance)
        if full_arm_conf is None:
            if PRINT_FAILURES: print('Door kinematic failure')
            return
        if any(pairwise_collision(robot_obstacle, b) for b in obstacles):
            if PRINT_FAILURES: print('Door collision failure')
            return
        arm_conf = get_joint_positions(world.robot, world.arm_joints)
        if arm_path and not teleport:
            distance = distance_fn(arm_path[-1], arm_conf)
            if MAX_CONF_DISTANCE < distance:
                if PRINT_FAILURES: print('Door proximity failure (distance={:.5f})'.format(distance))
                return
        arm_path.append(arm_conf)
        # wait_for_user()

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

    cmd = Sequence(State(world, savers=[robot_saver]), commands=[
        Trajectory(world, world.robot, world.arm_joints, approach_paths[0]),
        finger_cmd.commands[0],
        DoorTrajectory(world, world.robot, world.arm_joints, arm_path,
                       world.kitchen, [door_joint], door_path),
        finger_cmd.commands[0].reverse(),
        Trajectory(world, world.robot, world.arm_joints, reversed(approach_paths[-1])),
    ], name='pull')
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
        door_plans = [door_plan for door_plan in compute_door_paths(
            world, joint_name, door_conf1, door_conf2, obstacles, teleport=teleport)
                      if is_pull_safe(world, door_joint, door_plan, obstacles)]
        if not door_plans:
            return
        for i in range(max_attempts):
            door_path = random.choice(door_plans)
            # TracIK is itself stochastic
            randomize = (random.random() < P_RANDOMIZE_IK)
            ik_outputs = next(plan_pull(world, door_joint, door_path, base_conf,
                              randomize=randomize, collisions=collisions, teleport=teleport, **kwargs), None)
            if ik_outputs is not None:
                yield ik_outputs
                return
        if PRINT_FAILURES: print('Fixed pull failure')
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
                    base_conf, = next(safe_base_generator)
                except StopIteration:
                    return
                door_path = random.choice(door_paths)
                randomize = (random.random() < P_RANDOMIZE_IK)
                ik_outputs = next(plan_pull(world, door_joint, door_path, base_conf,
                                            randomize=randomize, collisions=collisions, teleport=teleport, **kwargs), None)
                if ik_outputs is not None:
                    yield (base_conf,) + ik_outputs
                    break
            else:
                if PRINT_FAILURES: print('Pull failure')
                yield None
    return gen

################################################################################

def parse_fluents(world, fluents):
    attachments = []
    obstacles = set()
    for fluent in fluents:
        predicate, args = fluent[0], fluent[1:]
        if predicate in {p.lower() for p in ['AtBConf', 'AtAConf', 'AtGConf']}:
            q, = args
            q.assign()
        elif predicate == 'AtAngle'.lower():
            raise RuntimeError()
            # j, a = args
            # a.assign()
            # obstacles.update(get_descendant_obstacles(a.body, a.joints[0]))
        elif predicate in {p.lower() for p in ['AtPose', 'AtWorldPose']}:
            b, p = args
            if isinstance(p, SurfaceDist):
                continue
            p.assign()
            obstacles.update(get_link_obstacles(world, b))
        elif predicate == 'AtGrasp'.lower():
            b, g = args
            if b is not None:
                attachments.append(g.get_attachment())
                attachments[-1].assign()
        else:
            raise NotImplementedError(predicate)
    return attachments, obstacles

def get_base_motion_fn(world, collisions=True, teleport=False,
                       restarts=4, iterations=75, smooth=100):

    def fn(bq1, bq2, aq, fluents=[]):
        #if bq1 == bq2:
        #    return None
        bq1.assign()
        aq.assign()
        attachments, obstacles = parse_fluents(world, fluents)
        if not collisions:
            obstacles = set()
        obstacles.update(world.static_obstacles)
        robot_saver = BodySaver(world.robot)
        if (bq1 == bq2) or teleport:
            path = [bq1.values, bq2.values]
        else:
            # It's important that the extend function is reversible to avoid getting trapped
            path = plan_nonholonomic_motion(world.robot, bq2.joints, bq2.values, attachments=attachments,
                                            obstacles=obstacles, custom_limits=world.custom_limits,
                                            reversible=True, self_collisions=False,
                                            restarts=restarts, iterations=iterations, smooth=smooth)
            if path is None:
                print('Failed to find a base motion plan!')
                set_renderer(enable=True)
                #print(fluents)
                for bq in [bq1, bq2]:
                    bq.assign()
                    wait_for_user()
                set_renderer(enable=False)
                return None
        # TODO: could actually plan with all joints as long as we return to the same config
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            Trajectory(world, world.robot, world.base_joints, path),
        ], name='base')
        return (cmd,)
    return fn

def get_reachability_test(world, **kwargs):
    base_motion_fn = get_base_motion_fn(world, restarts=2, iterations=50, smooth=0, **kwargs)
    bq0 = FConf(world.robot, world.base_joints)
    # TODO: can check for arm motions as well

    def test(bq):
        aq = world.carry_conf
        outputs = base_motion_fn(aq, bq0, bq, fluents=[])
        return outputs is not None
    return test

################################################################################

def get_arm_motion_gen(world, collisions=True, teleport=False):
    resolutions = ARM_RESOLUTION * np.ones(len(world.arm_joints))

    def fn(bq, aq1, aq2, fluents=[]):
        #if aq1 == aq2:
        #    return None
        bq.assign()
        aq1.assign()
        attachments, obstacles = parse_fluents(world, fluents)
        if not collisions:
            obstacles = set()
        obstacles.update(world.static_obstacles)
        robot_saver = BodySaver(world.robot)
        if teleport:
            path = [aq1.values, aq2.values]
        else:
            path = plan_joint_motion(world.robot, aq2.joints, aq2.values,
                                     attachments=attachments, obstacles=obstacles,
                                     self_collisions=SELF_COLLISIONS,
                                     disabled_collisions=world.disabled_collisions,
                                     custom_limits=world.custom_limits, resolutions=resolutions,
                                     restarts=2, iterations=25, smooth=25)
            if path is None:
                print('Failed to find an arm motion plan!')
                set_renderer(enable=True)
                #print(fluents)
                for bq in [aq1, aq2]:
                    bq.assign()
                    wait_for_user()
                set_renderer(enable=False)
                return None
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            Trajectory(world, world.robot, world.arm_joints, path),
        ], name='arm')
        return (cmd,)
    return fn

def get_gripper_motion_gen(world, teleport=False, **kwargs):
    resolutions = GRIPPER_RESOLUTION * np.ones(len(world.gripper_joints))

    def fn(gq1, gq2):
        #if gq1 == gq2:
        #    return None
        if teleport:
            path = [gq1.values, gq2.values]
        else:
            extend_fn = get_extend_fn(gq2.body, gq2.joints, resolutions=resolutions)
            path = [gq1.values] + list(extend_fn(gq1.values, gq2.values))
        cmd = Sequence(State(world), commands=[
            Trajectory(world, gq2.body, gq2.joints, path),
        ], name='gripper')
        return (cmd,)
    return fn

################################################################################

def get_calibrate_gen(world, collisions=True, teleport=False):

    def fn(bq, *args): #, aq):
        # TODO: include if holding anything?
        bq.assign()
        aq = world.carry_conf
        #aq.assign() # TODO: could sample aq instead achieve it by move actions
        #world.open_gripper()
        robot_saver = BodySaver(world.robot)
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            #Trajectory(world, world.robot, world.arm_joints, approach_path),
            # TODO: calibrate command
        ], name='calibrate')
        return (cmd,)
    return fn

################################################################################

OPEN = 'open'
CLOSED = 'closed'
DOOR_STATUSES = [OPEN, CLOSED]
# TODO: retrieve from entity

def get_gripper_open_test(world, percent=1e-1): #, tolerance=1e-2
    open_gq = percent*np.array(world.closed_gq.values) + \
              (1-percent)*np.array(world.open_gq.values)
    #open_gq = world.open_gq.values - tolerance * np.ones(len(world.gripper_joints))
    def test(gq):
        #if gq == world.open_gq:
        #    print('Initial grasp:', gq)
        return np.less_equal(open_gq, gq.values).all()
    return test

def get_door_test(world, percent=1e-1): #, tolerance=1e-2):
    def test(joint_name, conf, status):
        [joint] = conf.joints
        sign = world.get_door_sign(joint)
        #print(joint_name, world.closed_conf(joint), conf.values[0],
        #      world.open_conf(joint), status)
        position = sign*conf.values[0]
        if status == OPEN:
            open_position = sign * (percent * world.closed_conf(joint) +
                                    (1-percent) * world.open_conf(joint))
            #open_position = sign * world.open_conf(joint) - tolerance
            return open_position <= position
        elif status == CLOSED:
            closed_position = sign * ((1-percent) * world.closed_conf(joint) +
                                      percent * world.open_conf(joint))
            #closed_position = sign * world.closed_conf(joint) + tolerance
            return position <= closed_position
        raise NotImplementedError(status)
    return test

################################################################################

def get_cfree_pose_pose_test(world, collisions=True, **kwargs):
    def test(o1, rp1, o2, rp2, s):
        if not collisions or (o1 == o2):
            return True
        if isinstance(rp1, SurfaceDist) or isinstance(rp2, SurfaceDist):
            return True # TODO: perform this probabilistically
        rp1.assign()
        rp2.assign()
        return not pairwise_collision(world.get_body(o1), world.get_body(o2))
    return test

def get_cfree_bconf_pose_test(world, collisions=True, **kwargs):
    def test(bq, o2, p2):
        if not collisions:
            return True
        if isinstance(p2, SurfaceDist):
            return True # TODO: perform this probabilistically
        bq.assign()
        world.carry_conf.assign()
        p2.assign()
        obstacles = get_link_obstacles(world, o2)
        return not any(pairwise_collision(world.robot, obst) for obst in obstacles)
    return test

def get_cfree_approach_pose_test(world, collisions=True, **kwargs):
    def test(o1, p1, g1, o2, p2):
        # o1 will always be a movable object
        if not collisions or (o1 == o2):
            return True
        if isinstance(p2, SurfaceDist):
            return True # TODO: perform this probabilistically
        body = world.get_body(o1)
        p2.assign()
        obstacles = get_link_obstacles(world, o2) # - {body}
        if not obstacles:
            return True
        for _ in iterate_approach_path(world, p1, g1, body=body):
            if any(pairwise_collision(part, obst) for part in
                   [world.gripper, body] for obst in obstacles):
                return False
        return True
    return test

################################################################################

def get_cfree_traj_pose_test(world, collisions=True, **kwargs):
    def test(at, o, p):
        if not collisions:
            return True
        # TODO: check door collisions
        # TODO: still need to check static links at least once
        if isinstance(p, SurfaceDist):
            return True # TODO: perform this probabilistically
        p.assign()
        state = at.context.copy()
        state.assign()
        for command in at.commands:
            obstacles = get_link_obstacles(world, o) - command.bodies - p.bodies # Doesn't include o at p
            for _ in command.iterate(state):
                state.derive()
                for attachment in state.attachments.values():
                    if any(pairwise_collision(attachment.child, obst) for obst in obstacles):
                        return False
                # TODO: just check collisions with moving links
                if any(pairwise_collision(world.robot, obst) for obst in obstacles):
                    return False
        return True
    return test
