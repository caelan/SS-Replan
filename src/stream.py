import numpy as np
import random
import math
from itertools import islice
from collections import namedtuple

from pybullet_tools.pr2_utils import is_visible_point, get_view_aabb, support_from_aabb
from pybullet_tools.utils import pairwise_collision, multiply, invert, get_joint_positions, BodySaver, get_distance, \
    set_joint_positions, plan_direct_joint_motion, plan_joint_motion, \
    get_custom_limits, all_between, link_from_name, get_link_pose, \
    Euler, quat_from_euler, set_pose, point_from_pose, sample_placement_on_aabb, get_sample_fn, get_pose, \
    stable_z_on_aabb, euler_from_quat, quat_from_pose, Ray, get_distance_fn, Point, set_configuration, \
    is_point_in_polygon, grow_polygon, Pose, get_moving_links, get_aabb_extent, get_aabb_center, \
    INF, apply_affine, get_joint_name, get_unit_vector, get_link_subtree, get_link_name, unit_quat, joint_from_name, \
    get_extend_fn, wait_for_user, set_renderer, child_link_from_joint, unit_from_theta
from pddlstream.algorithms.downward import MAX_FD_COST #, get_cost_scale

from src.command import Sequence, State, Detect, DoorTrajectory
from src.database import load_placements, get_surface_reference_pose, load_pull_base_poses, load_forward_placements, load_inverse_placements
from src.utils import get_grasps, iterate_approach_path, ALL_SURFACES, \
    get_descendant_obstacles, surface_from_name, RelPose, compute_surface_aabb, create_relative_pose, Z_EPSILON, \
    get_surface_obstacles, test_supported, \
    get_link_obstacles, ENV_SURFACES, FConf, open_surface_joints, DRAWERS, STOVES, \
    TOP_GRASP, KNOBS, APPROACH_DISTANCE, FINGER_EXTENT, set_tool_pose, translate_linearly
from src.visualization import GROW_INVERSE_BASE, GROW_FORWARD_RADIUS
from src.inference import SurfaceDist
from examples.discrete_belief.run import revisit_mdp_cost, clip_cost, DDist #, MAX_COST

COST_SCALE = 1 # costs will always be greater than one
MAX_COST = MAX_FD_COST / (25*COST_SCALE)
#MAX_COST = MAX_FD_COST / get_cost_scale()
# TODO: move this to FD

# TODO: ensure the top part of object is visible
DETECT_COST = 1.0
BASE_CONSTANT = 1.0 # 1 | 10
BASE_VELOCITY = 0.25
SELF_COLLISIONS = True

PRINT_FAILURES = True
MOVE_ARM = True
ARM_RESOLUTION = 0.05
GRIPPER_RESOLUTION = 0.01
DOOR_RESOLUTION = 0.025

# TracIK is itself stochastic
#P_RANDOMIZE_IK = 0.25  # 0.0 | 0.5
P_RANDOMIZE_IK = 1.0

MAX_CONF_DISTANCE = 0.75
NEARBY_APPROACH = MAX_CONF_DISTANCE
NEARBY_PULL = 0.25
FIXED_FAILURES = INF # 5
REVERSE_DISTANCE = 0.1

DOOR_PROXIMITY = True

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

def detect_cost_fn(obj_name, rp_dist, obs, rp_sample):
    # TODO: extend to continuous rp_sample controls using densities
    # TODO: count samples in a nearby vicinity to be invariant to number of samples
    prob = 1. if rp_dist == rp_sample else rp_dist.discrete_prob(rp_sample)
    cost = clip_cost(compute_detect_cost(prob), max_cost=MAX_COST)
    #print('{}) Detect Prob: {:.3f} | Detect Cost: {:.3f}'.format(
    #    rp_dist.surface_name, prob, cost))
    return cost

################################################################################

# TODO: more general forward kinematics

def get_compute_pose_kin(world):
    #obstacles = world.static_obstacles

    def fn(o1, rp, o2, p2):
        if o1 == o2:
            return None
        if isinstance(rp, SurfaceDist):
            p1 = rp.project(lambda x: fn(o1, x, o2, p2)[0]) # TODO: filter if any in collision
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
        #p1.assign()
        #if any(pairwise_collision(body, obst) for obst in obstacles):
        #    return None
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

def is_visible_by_camera(world, point):
    for camera_name in world.cameras:
        camera_body, camera_matrix, camera_depth = world.cameras[camera_name]
        camera_pose = get_pose(camera_body)
        #camera_point = point_from_pose(camera_pose)
        if is_visible_point(camera_matrix, camera_depth, point, camera_pose):
            return True
    return False

def get_compute_detect(world, ray_trace=True, **kwargs):
    obstacles = world.static_obstacles
    detect_scale = 1.25 if world.is_real() else 0.5 # 0.05 | 0.5 | 1.0 | 1.25

    def fn(obj_name, pose):
        # TODO: incorporate probability mass
        # Ether sample observation (control) or target belief (next state)
        body = world.get_body(obj_name)
        open_surface_joints(world, pose.support)
        for camera_name in world.cameras:
            camera_body, camera_matrix, camera_depth = world.cameras[camera_name]
            camera_pose = get_pose(camera_body)
            camera_point = point_from_pose(camera_pose)
            obj_point = point_from_pose(pose.get_world_from_body())

            aabb = get_view_aabb(body, camera_pose)
            center = get_aabb_center(aabb)
            extent = np.multiply([detect_scale, detect_scale, 1], get_aabb_extent(aabb))
            view_aabb = (center - extent / 2, center + extent / 2)
            # print(is_visible_aabb(view_aabb, camera_matrix=camera_matrix))
            obj_points = apply_affine(camera_pose, support_from_aabb(view_aabb)) + [obj_point]
            # obj_points = [obj_point]
            if not all(is_visible_point(camera_matrix, camera_depth, point, camera_pose)
                       for point in obj_points):
                continue
            rays = [Ray(camera_point, point) for point in obj_points]
            detect = Detect(world, camera_name, obj_name, pose, rays)
            if ray_trace:
                # TODO: how should doors be handled?
                move_occluding(world)
                open_surface_joints(world, pose.support)
                detect.pose.assign()
                if obstacles & detect.compute_occluding():
                    continue
            #detect.draw()
            #wait_for_user()
            return (detect,)
        return None
    return fn


def move_occluding(world):
    # Prevent obstruction by other objects
    # TODO: this is a bit of a hack due to pybullet
    world.set_base_conf([-5.0, 0, 0])
    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        if joint_name in DRAWERS:
            world.open_door(joint)
        else:
            world.close_door(joint)
    for name in world.movable:
        set_pose(world.get_body(name), Pose(Point(z=-5.0)))

def get_ofree_ray_pose_test(world, **kwargs):
    # TODO: detect the configuration of joints
    def test(detect, obj_name, pose):
        if (detect.name == obj_name) or (detect.surface_name == obj_name) or isinstance(pose, SurfaceDist):
            return True
        move_occluding(world)
        detect.pose.assign()
        pose.assign()
        body = world.get_body(detect.name)
        obstacles = get_link_obstacles(world, obj_name)
        if any(pairwise_collision(body, obst) for obst in obstacles):
            return False
        visible = not obstacles & detect.compute_occluding()
        #if not visible:
        #    handles = detect.draw()
        #    wait_for_user()
        #    remove_handles(handles)
        return visible
    return test

def get_ofree_ray_grasp_test(world, **kwargs):
    def test(detect, bconf, aconf, obj_name, grasp):
        if detect.name == obj_name:
            return True
        # TODO: check collisions with the placement distribution
        # Move top grasps more vertically
        move_occluding(world)
        bconf.assign()
        aconf.assign()
        detect.pose.assign()
        if obj_name is not None:
            grasp.assign()
            obstacles = get_link_obstacles(world, obj_name)
        else:
            obstacles = get_descendant_obstacles(world.robot)
        visible = not obstacles & detect.compute_occluding()
        #if not visible:
        #    handles = detect.draw()
        #    wait_for_user()
        #    remove_handles(handles)
        return visible
    return test

class Observation(object):
    # Primary motivation is to seperate the object
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return 'obs({})'.format(self.value)

def get_sample_belief_gen(world, # min_prob=1. / NUM_PARTICLES,  # TODO: relative instead?
                          max_observations=10,
                          mlo_only=False, ordered=False, **kwargs):
    # TODO: incorporate ray tracing
    detect_fn = get_compute_detect(world, ray_trace=False, **kwargs)
    def gen(obj_name, pose_dist, surface_name):
        # TODO: apply these checks to the whole surfaces
        if isinstance(pose_dist, RelPose):
            yield (pose_dist,)
            return
        valid_samples = {}
        for rp in pose_dist.dist.support():
            if 1 <= rp.observations:
                continue
            prob = pose_dist.discrete_prob(rp)
            #cost = detect_cost_fn(obj_name, pose_dist, obs=None, rp_sample=rp)
            #if (cost < MAX_COST): # and (min_prob < prob):
            # pose = rp.get_world_from_body()
            result = detect_fn(obj_name, rp)
            if result is not None:
                # detect, = result
                # detect.draw()
                valid_samples[rp] = prob
        if not valid_samples:
            return

        if mlo_only:
            rp = max(valid_samples, key=valid_samples.__getitem__)
            obs = Observation(rp)
            yield (obs,)
            return
        if ordered:
            for rp in sorted(valid_samples, key=valid_samples.__getitem__, reverse=True):
                obs = Observation(rp)
                yield (obs,)
            return
        observations = 0
        while valid_samples and (observations < max_observations):
            dist = DDist(valid_samples)
            rp = dist.sample()
            del valid_samples[rp]
            obs = Observation(rp)
            yield (obs,)
            observations += 1
    return gen

def update_belief_fn(world, **kwargs):
    def fn(obj_name, pose_dist, surface_name, obs):
        rp = obs.value # TODO: proper Bayesian update
        return (rp,)
    return fn

################################################################################

def get_test_near_pose(world, grow_entity=GROW_FORWARD_RADIUS, collisions=False, teleport=False, **kwargs):
    base_from_objects = grow_polygon(map(point_from_pose, load_forward_placements(world, **kwargs)), radius=grow_entity)
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
        if not DOOR_PROXIMITY:
            return True
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
                   visibility=True, learned=True, collisions=True,
                   pos_scale=0.01, rot_scale=np.pi/16, robust_radius=0.0,
                   z_offset=Z_EPSILON, **kwargs):

    # TODO: remove fixed collisions with contained surfaces
    # TODO: place where currently standing
    def gen(obj_name, surface_name):
        obj_body = world.get_body(obj_name)
        surface_body = world.kitchen
        if surface_name in ENV_SURFACES:
            surface_body = world.environment_bodies[surface_name]
        surface_aabb = compute_surface_aabb(world, surface_name)
        learned_poses = load_placements(world, surface_name) if learned else [] # TODO: GROW_PLACEMENT

        yaw_range = (-np.pi, np.pi)
        #if world.is_real():
        #    center = -np.pi/4
        #    half_extent = np.pi / 16
        #    yaw_range = (center-half_extent, center+half_extent)
        while True:
            for _ in range(max_attempts):
                if surface_name in STOVES:
                    surface_link = link_from_name(world.kitchen, surface_name)
                    world_from_surface = get_link_pose(world.kitchen, surface_link)
                    z = stable_z_on_aabb(obj_body, surface_aabb) - point_from_pose(world_from_surface)[2]
                    yaw = random.uniform(*yaw_range)
                    body_pose_surface = Pose(Point(z=z + z_offset), Euler(yaw=yaw))
                    body_pose_world = multiply(world_from_surface, body_pose_surface)
                elif learned:
                    if not learned_poses:
                        return
                    surface_pose_world = get_surface_reference_pose(surface_body, surface_name)
                    sampled_pose_surface = multiply(surface_pose_world, random.choice(learned_poses))
                    [x, y, _] = point_from_pose(sampled_pose_surface)
                    _, _, yaw = euler_from_quat(quat_from_pose(sampled_pose_surface))
                    dx, dy = np.random.normal(scale=pos_scale, size=2) if pos_scale else np.zeros(2)
                    # TODO: avoid reloading
                    z = stable_z_on_aabb(obj_body, surface_aabb)
                    yaw = random.uniform(*yaw_range)
                    #yaw = wrap_angle(yaw + np.random.normal(scale=rot_scale))
                    quat = quat_from_euler(Euler(yaw=yaw))
                    body_pose_world = ([x+dx, y+dy, z+z_offset], quat)
                    # TODO: project onto the surface
                else:
                    # TODO: halton sequence
                    # unit_generator(d, use_halton=True)
                    body_pose_world = sample_placement_on_aabb(obj_body, surface_aabb,
                                                               epsilon=z_offset, percent=2.0)
                    if body_pose_world is None:
                        continue # return?
                if visibility and not is_visible_by_camera(world, point_from_pose(body_pose_world)):
                    continue
                # TODO: make sure the surface is open when doing this

                robust = True
                if robust_radius != 0.:
                    for theta in np.linspace(0, 5 * np.pi, num=8):
                        x, y = robust_radius*unit_from_theta(theta)
                        delta_body = Pose(Point(x, y))
                        delta_world = multiply(body_pose_world, delta_body)
                        set_pose(obj_body, delta_world)
                        if not test_supported(world, obj_body, surface_name, collisions=collisions):
                            robust = False
                            break

                set_pose(obj_body, body_pose_world)
                if robust and test_supported(world, obj_body, surface_name, collisions=collisions):
                    rp = create_relative_pose(world, obj_name, surface_name)
                    yield (rp,)
                    break
            else:
                yield None
    return gen

def get_nearby_stable_gen(world, max_attempts=25, **kwargs):
    stable_gen = get_stable_gen(world, **kwargs)
    test_near_pose = get_test_near_pose(world, #surface_names=[],
                                        grasp_types=[TOP_GRASP], grow_entity=0.0)
    compute_pose_kin = get_compute_pose_kin(world)

    def gen(obj_name, surface_name, pose2, base_conf):
        #base_conf.assign()
        #pose2.assign()
        max_failures = FIXED_FAILURES if world.task.movable_base else INF
        failures = 0
        while failures <= max_failures:
            for rel_pose, in islice(stable_gen(obj_name, surface_name), max_attempts):
                pose1, = compute_pose_kin(obj_name, rel_pose, surface_name, pose2)
                if test_near_pose(obj_name, pose1, base_conf):
                    yield (pose1, rel_pose)
                    break
            else:
                yield None
                failures += 1
    return gen

def get_grasp_gen(world, collisions=False, randomize=True, **kwargs): # teleport=False,
    # TODO: produce carry arm confs here
    def gen(name, grasp_type):
        for grasp in get_grasps(world, name, grasp_types=[grasp_type], **kwargs):
            yield (grasp,)
    return gen

################################################################################

def is_robot_visible(world, links):
    for link in links:
        link_point = point_from_pose(get_link_pose(world.robot, link))
        visible = False
        for camera_body, camera_matrix, camera_depth in world.cameras.values():
            camera_pose = get_pose(camera_body)
            #camera_point = point_from_pose(camera_pose)
            #add_line(link_point, camera_point)
            if is_visible_point(camera_matrix, camera_depth, link_point, camera_pose):
                visible = True
                break
        if not visible:
            return False
    #wait_for_user()
    return True

def test_base_conf(world, bq, obstacles, min_distance=0.0):
    robot_links = [world.franka_link, world.gripper_link] if world.is_real() else []
    bq.assign()
    for conf in world.special_confs:
        # Could even sample a special visible conf for this base_conf
        conf.assign()
        if not is_robot_visible(world, robot_links) or any(pairwise_collision(
                world.robot, b, max_distance=min_distance) for b in obstacles):
            return False
    return True

def inverse_reachability(world, base_generator, obstacles=set(),
                         max_attempts=25, **kwargs):
    min_distance = 0.01 #if world.is_real() else 0.0
    min_nearby_distance = 0.1 # if world.is_real() else 0.0
    lower_limits, upper_limits = get_custom_limits(
        world.robot, world.base_joints, world.custom_limits)
    while True:
        attempt = 0
        for base_conf in islice(base_generator, max_attempts):
            attempt += 1
            if not all_between(lower_limits, base_conf, upper_limits):
                continue
            bq = FConf(world.robot, world.base_joints, base_conf)
            #wait_for_user()
            if not test_base_conf(world, bq, obstacles, min_distance=min_distance):
                continue
            if world.is_real():
                # TODO: could also rotate in place
                # TODO: restrict orientation to face the counter
                nearby_values = translate_linearly(world, distance=-REVERSE_DISTANCE)
                bq.nearby_bq = FConf(world.robot, world.base_joints, nearby_values)
                if not test_base_conf(world, bq.nearby_bq, obstacles, min_distance=min_nearby_distance):
                    continue
            #if PRINT_FAILURES: print('Success after {} IR attempts:'.format(attempt))
            bq.assign()
            #wait_for_user()
            yield bq
            break
        else:
            if PRINT_FAILURES: print('Failed after {} IR attempts:'.format(attempt))
            if attempt < max_attempts - 1:
                return
            yield None

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

def plan_workspace(world, tool_path, obstacles, randomize=True, teleport=False):
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
        #set_joint_positions(world.kitchen, [door_joint], door_path[i])
        tolerance = INF if i == 0 else NEARBY_PULL
        full_arm_conf = world.solve_inverse_kinematics(tool_pose, nearby_tolerance=tolerance)
        if full_arm_conf is None:
            # TODO: this fails when teleport=True
            if PRINT_FAILURES: print('Workspace kinematic failure')
            return None
        if any(pairwise_collision(robot_obstacle, b) for b in obstacles):
            if PRINT_FAILURES: print('Workspace collision failure')
            return None
        arm_conf = get_joint_positions(world.robot, world.arm_joints)
        if arm_path and not teleport:
            distance = distance_fn(arm_path[-1], arm_conf)
            if MAX_CONF_DISTANCE < distance:
                if PRINT_FAILURES: print('Workspace proximity failure (distance={:.5f})'.format(distance))
                return None
        arm_path.append(arm_conf)
        # wait_for_user()
    return arm_path

################################################################################

HandleGrasp = namedtuple('HandleGrasp', ['link', 'handle_grasp', 'handle_pregrasp'])
DoorPath = namedtuple('DoorPath', ['link_path', 'handle_path', 'handle_grasp', 'tool_path'])

def get_handle_grasps(world, joint, pull=True, pre_distance=APPROACH_DISTANCE):
    pre_direction = pre_distance * get_unit_vector([0, 0, 1])
    #half_extent = 1.0*FINGER_EXTENT[2] # Collides
    half_extent = 1.05*FINGER_EXTENT[2]

    grasps = []
    for link in get_link_subtree(world.kitchen, joint):
        if 'handle' in get_link_name(world.kitchen, link):
            # TODO: can adjust the position and orientation on the handle
            for yaw in [0, np.pi]: # yaw=0 DOESN'T WORK WITH LULA
                handle_grasp = (Point(z=-half_extent), quat_from_euler(Euler(roll=np.pi, pitch=np.pi/2, yaw=yaw)))
                #if not pull:
                #    handle_pose = get_link_pose(world.kitchen, link)
                #    for distance in np.arange(0., 0.05, step=0.001):
                #        pregrasp = multiply(([0, 0, -distance], unit_quat()), handle_grasp)
                #        tool_pose = multiply(handle_pose, invert(pregrasp))
                #        set_tool_pose(world, tool_pose)
                #        # TODO: check collisions
                #        wait_for_user()
                handle_pregrasp = multiply((pre_direction, unit_quat()), handle_grasp)
                grasps.append(HandleGrasp(link, handle_grasp, handle_pregrasp))
    return grasps

def compute_door_paths(world, joint_name, door_conf1, door_conf2, obstacles=set(), teleport=False):
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
    # TODO: open until collision for the drawers

    sign = world.get_door_sign(door_joint)
    pull = (sign*door_path[0][0] < sign*door_path[-1][0])
    # door_obstacles = get_descendant_obstacles(world.kitchen, door_joint)
    for handle_grasp in get_handle_grasps(world, door_joint, pull=pull):
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

def get_gripper_open_test(world, error_percent=0.1): #, tolerance=1e-2
    open_gq = error_percent * np.array(world.closed_gq.values) + \
              (1 - error_percent) * np.array(world.open_gq.values)
    #open_gq = world.open_gq.values - tolerance * np.ones(len(world.gripper_joints))
    def test(gq):
        #if gq == world.open_gq:
        #    print('Initial grasp:', gq)
        return np.less_equal(open_gq, gq.values).all()
    return test

def get_door_test(world, error_percent=0.35): #, tolerance=1e-2):
    # TODO: separate error for open/closed
    def test(joint_name, conf, status):
        [joint] = conf.joints
        sign = world.get_door_sign(joint)
        #print(joint_name, world.closed_conf(joint), conf.values[0],
        #      world.open_conf(joint), status)
        position = sign*conf.values[0]
        if status == OPEN:
            open_position = sign * (error_percent * world.closed_conf(joint) +
                                    (1 - error_percent) * world.open_conf(joint))
            #open_position = sign * world.open_conf(joint) - tolerance
            return open_position <= position
        elif status == CLOSED:
            closed_position = sign * ((1 - error_percent) * world.closed_conf(joint) +
                                      error_percent * world.open_conf(joint))
            #closed_position = sign * world.closed_conf(joint) + tolerance
            return position <= closed_position
        raise NotImplementedError(status)
    return test

################################################################################

def get_cfree_relpose_relpose_test(world, collisions=True, **kwargs):
    def test(o1, rp1, o2, rp2, s):
        if not collisions or (o1 == o2):
            return True
        if isinstance(rp1, SurfaceDist) or isinstance(rp2, SurfaceDist):
            return True # TODO: perform this probabilistically
        rp1.assign()
        rp2.assign()
        return not pairwise_collision(world.get_body(o1), world.get_body(o2))
    return test

def get_cfree_worldpose_test(world, collisions=True, **kwargs):
    def test(o1, wp1):
        if isinstance(wp1, SurfaceDist):
            return True
        if not collisions or (wp1.support not in DRAWERS):
            return True
        body = world.get_body(o1)
        wp1.assign()
        obstacles = world.static_obstacles
        if any(pairwise_collision(body, obst) for obst in obstacles):
            return False
        return True
    return test

def get_cfree_worldpose_worldpose_test(world, collisions=True, **kwargs):
    def test(o1, wp1, o2, wp2):
        if isinstance(wp1, SurfaceDist) or isinstance(wp2, SurfaceDist):
            return True
        if not collisions or (o1 == o2) or (o2 == wp1.support): # DRAWERS
            return True
        body = world.get_body(o1)
        wp1.assign()
        wp2.assign()
        if any(pairwise_collision(body, obst) for obst in get_surface_obstacles(world, o2)):
            return False
        return True
    return test

def get_cfree_bconf_pose_test(world, collisions=True, **kwargs):
    def test(bq, o2, wp2):
        if not collisions:
            return True
        if isinstance(wp2, SurfaceDist):
            return True # TODO: perform this probabilistically
        bq.assign()
        world.carry_conf.assign()
        wp2.assign()
        obstacles = get_link_obstacles(world, o2)
        return not any(pairwise_collision(world.robot, obst) for obst in obstacles)
    return test

def get_cfree_approach_pose_test(world, collisions=True, **kwargs):
    def test(o1, wp1, g1, o2, wp2):
        # o1 will always be a movable object
        if isinstance(wp2, SurfaceDist):
            return True # TODO: perform this probabilistically
        if not collisions or (o1 == o2) or (o2 == wp1.support):
            return True
        # TODO: could define these on sets of samples to prune all at once
        body = world.get_body(o1)
        wp2.assign()
        obstacles = get_link_obstacles(world, o2) # - {body}
        if not obstacles:
            return True
        for _ in iterate_approach_path(world, wp1, g1, body=body):
            if any(pairwise_collision(part, obst) for part in
                   [world.gripper, body] for obst in obstacles):
                # TODO: some collisions the bottom drawer and the top drawer handle
                #print(o1, wp1.support, o2, wp2.support)
                #wait_for_user()
                return False
        return True
    return test

def get_cfree_angle_angle_test(world, collisions=True, **kwargs):
    def test(j1, a1, a2, o2, wp):
        if not collisions or (o2 in j1): # (j1 == JOINT_TEMPLATE.format(o2)):
            return True
        # TODO: check pregrasp path as well
        # TODO: pull path collisions
        wp.assign()
        set_configuration(world.gripper, world.open_gq.values)
        status = compute_door_paths(world, j1, a1, a2, obstacles=get_link_obstacles(world, o2))
        #print(j1, a1, a2, o2, wp)
        #if not status:
        #    set_renderer(enable=True)
        #    for a in [a1, a2]:
        #        a.assign()
        #        wait_for_user()
        #    set_renderer(enable=False)
        return status
    return test

################################################################################

def get_cfree_traj_pose_test(world, collisions=True, **kwargs):
    def test(at, o, wp):
        if not collisions:
            return True
        # TODO: check door collisions
        # TODO: still need to check static links at least once
        if isinstance(wp, SurfaceDist):
            return True # TODO: perform this probabilistically
        wp.assign()
        state = at.context.copy()
        state.assign()
        all_bodies = {body for command in at.commands for body in command.bodies}
        for command in at.commands:
            obstacles = get_link_obstacles(world, o) - all_bodies
            # TODO: why did I previously remove o at p?
            #obstacles = get_link_obstacles(world, o) - command.bodies  # - p.bodies # Doesn't include o at p
            if not obstacles:
                continue
            if isinstance(command, DoorTrajectory):
                [door_joint] = command.door_joints
                surface_name = get_link_name(world.kitchen, child_link_from_joint(door_joint))
                if wp.support == surface_name:
                    return True
            for _ in command.iterate(state):
                state.derive()
                #for attachment in state.attachments.values():
                #    if any(pairwise_collision(attachment.child, obst) for obst in obstacles):
                #        return False
                # TODO: just check collisions with moving links
                if any(pairwise_collision(world.robot, obst) for obst in obstacles):
                    #print(at, o, p)
                    #wait_for_user()
                    return False
        return True
    return test
