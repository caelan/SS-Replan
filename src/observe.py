from __future__ import print_function

import math
import numpy as np
import random

from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import get_pose, point_from_pose, Ray, batch_ray_collision, has_gui, add_line, BLUE, \
    wait_for_duration, remove_handles, Pose, Point, Euler, multiply, set_pose, aabb_contains_point, tform_point, angle_between
from src.utils import CAMERA_MATRIX, KINECT_DEPTH, create_relative_pose, create_world_pose

OBS_P_FP, OBS_P_FN = 0.0, 0.0
#OBS_POS_STD, OBS_ORI_STD = 0.01, np.pi / 8
OBS_POS_STD, OBS_ORI_STD = 0., 0.
ELSEWHERE = None # symbol for elsewhere pose


################################################################################

def are_visible(world):
    ray_names = []
    rays = []
    for name in world.movable:
        for camera, info in world.cameras.items():
            camera_pose = get_pose(info.body)
            camera_point = point_from_pose(camera_pose)
            point = point_from_pose(get_pose(world.get_body(name)))
            if is_visible_point(CAMERA_MATRIX, KINECT_DEPTH, point, camera_pose=camera_pose):
                ray_names.append(name)
                rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    visible_indices = [idx for idx, (name, result) in enumerate(zip(ray_names, ray_results))
                       if result.objectUniqueId == world.get_body(name)]
    visible_names = {ray_names[idx] for idx in visible_indices}
    print('Detected:', sorted(visible_names))
    if has_gui():
        handles = [add_line(rays[idx].start, rays[idx].end, color=BLUE)
                   for idx in visible_indices]
        wait_for_duration(1.0)
        remove_handles(handles)
    # TODO: the object drop seems to happen around here
    return visible_names

################################################################################

def fully_observe_pybullet(world):
    return {name: get_pose(body) for name, body in world.body_from_name.items()}


def observe_pybullet(world):
    # TODO: randomize robot's pose
    # TODO: probabilities based on whether in viewcone or not
    # TODO: sample from poses on table
    # world_saver = WorldSaver()
    visible_entities = are_visible(world)
    detections = {}
    assert OBS_P_FP == 0
    for name in visible_entities:
        # TODO: false positives
        if random.random() < OBS_P_FN:
            continue
        body = world.get_body(name)
        pose = get_pose(body)
        dx, dy = np.random.multivariate_normal(
            mean=np.zeros(2), cov=math.pow(OBS_POS_STD, 2) * np.eye(2))
        dyaw, = np.random.multivariate_normal(
            mean=np.zeros(1), cov=math.pow(OBS_ORI_STD, 2) * np.eye(1))
        print('{}: dx={:.3f}, dy={:.3f}, dyaw={:.5f}'.format(name, dx, dy, dyaw))
        noise_pose = Pose(Point(x=dx, y=dy), Euler(yaw=dyaw))
        observed_pose = multiply(pose, noise_pose)
        #world.get_body_type(name)
        detections.setdefault(name, []).append(observed_pose) # TODO: use type instead
    #world_saver.restore()
    return detections

################################################################################

def fix_detections(belief, detections, **kwargs):
    # TODO: move directly to belief?
    world = belief.world
    fixed_detections = {}
    for name in detections:
        if name == belief.holding:
            continue
        for observed_pose in detections[name]:
            fixed_pose, support = world.fix_pose(name, observed_pose, **kwargs)
            if fixed_pose is not None:
                fixed_detections.setdefault(name, []).append(fixed_pose)
    return fixed_detections


def relative_detections(belief, detections):
    world = belief.world
    rel_detections = {}
    world_aabb = world.get_world_aabb()
    for name in detections:
        if name == belief.holding:
            continue
        body = world.get_body(name)
        for observed_pose in detections[name]:
            world_z_axis = np.array([0, 0, 1])
            local_z_axis = tform_point(observed_pose, world_z_axis)
            if np.pi/2 < angle_between(world_z_axis, local_z_axis):
                observed_pose = multiply(observed_pose, Pose(euler=Euler(roll=np.pi)))
            if not aabb_contains_point(point_from_pose(observed_pose), world_aabb):
                continue
            set_pose(body, observed_pose)
            support = world.get_supporting(name)
            #assert support is not None
            # Could also fix as relative to the world
            if support is None:
                # TODO: prune if nowhere near a surface (e.g. on the robot)
                relative_pose = create_world_pose(world, name, init=True)
            else:
                relative_pose = create_relative_pose(world, name, support, init=True)
            rel_detections.setdefault(name, []).append(relative_pose)
            # relative_pose.assign()
    return rel_detections
