from __future__ import print_function

import numpy as np
import random
import math
import time

from pddlstream.utils import str_from_object
from examples.discrete_belief.dist import UniformDist, DeltaDist
#from examples.discrete_belief.run import continue_mdp_cost
#from examples.pybullet.pr2_belief.primitives import get_observation_fn
#from examples.pybullet.pr2_belief.problems import BeliefState, BeliefTask

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import point_from_pose, Ray, batch_ray_collision, Point, Pose, Euler, set_pose, get_pose, BodySaver, \
    LockRenderer, multiply, spaced_colors, WorldSaver, \
    pairwise_collision, elapsed_time, randomize, remove_handles, add_line, BLUE, has_gui, wait_for_duration
from src.belief import NUM_PARTICLES, PoseDist
from src.stream import get_stable_gen
from src.utils import KINECT_DEPTH, CAMERA_MATRIX, create_relative_pose, \
    RelPose

OBS_P_FP, OBS_P_FN = 0.0, 0.0

#OBS_POS_STD, OBS_ORI_STD = 0.01, np.pi / 8
OBS_POS_STD, OBS_ORI_STD = 0., 0.

ELSEWHERE = None # symbol for elsewhere pose

# TODO: prior on the number of false detections to ensure correlated
# TODO: could do open world or closed world. For open world, can sum independent probabilities
# TODO: use a proper probabilistic programming library rather than dist.py

# Detect preconditions and cost
# * Most general would be conditioning the success prob on the full state via a cost
# * Does not admit factoring though
# * Instead, produce a detection for a subset of the region
# * Preconditions involving the likelihood something is interfering

################################################################################

# https://github.com/tlpmit/hpn
# https://github.mit.edu/tlp/bhpn

# https://github.com/caelan/pddlstream/tree/stable/examples/discrete_belief
# https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_online.py
# https://github.mit.edu/caelan/stripstream/blob/master/robotics/openrave/belief_tamp.py
# https://github.mit.edu/caelan/ss/blob/master/belief/belief_online.py
# https://github.com/caelan/pddlstream/blob/stable/examples/pybullet/pr2_belief/run.py

################################################################################

# TODO: point estimates and confidence intervals/regions

class Belief(object):
    def __init__(self, world, pose_dists={}, grasped=None):
        self.world = world
        self.pose_dists = pose_dists
        self.grasped = grasped # grasped or holding?
        self.color_from_name = dict(zip(self.names, spaced_colors(len(self.names))))
        self.observations = []
        self.handles = []
        # TODO: belief fluents
        # TODO: report grasp failure if gripper is ever fully closed
    @property
    def names(self):
        return sorted(self.pose_dists)
    @property
    def holding(self):
        if self.grasped is None:
            return None
        return self.grasped.body_name
    def update(self, observation, n_samples=25):
        start_time = time.time()
        detections = fix_detections(self, observation)
        self.observations.append(detections)
        # Processing detected first
        # Could simply sample from the set of worlds and update
        # Would need to sample many worlds with name at different poses
        # Instead, let the moving object take on different poses
        order = [name for name in detections] # Detected
        order.extend(set(self.pose_dists) - set(order)) # Not detected
        with WorldSaver():
            with LockRenderer():
                for name in order:
                    self.pose_dists[name] = self.pose_dists[name].update(
                        self, detections, n_samples=n_samples)
        print('Update time: {:.3f} sec for {} objects and {} samples'.format(
            elapsed_time(start_time), len(order), n_samples))
        return self
    def sample(self):
        while True:
            poses = {}
            for name, pose_dist in randomize(self.pose_dists.items()):
                body = self.world.get_body(name)
                pose = pose_dist.sample()
                pose.assign()
                if any(pairwise_collision(body, self.world.get_body(other)) for other in poses):
                    break
                poses[name] = pose
            else:
                return poses
    #def resample(self):
    #    for pose_dist in self.pose_dists:
    #        pose_dist.resample() # Need to update distributions
    def dump(self):
        print(self)
        for i, name in enumerate(sorted(self.pose_dists)):
            #self.pose_dists[name].dump()
            print(i, name, self.pose_dists[name])
    def draw(self, **kwargs):
        with LockRenderer():
            remove_handles(self.handles)
            self.handles = []
            with WorldSaver():
                for name, pose_dist in self.pose_dists.items():
                    self.handles.extend(pose_dist.draw(
                            color=self.color_from_name[name], **kwargs))
    def __repr__(self):
        return '{}(holding={}, placed={})'.format(self.__class__.__name__, self.holding, str_from_object(
            {name: self.pose_dists[name].surface_dist for name in self.names}))

################################################################################

def create_observable_pose_dist(world, obj_name):
    body = world.get_body(obj_name)
    surface_name = world.get_supporting(obj_name)
    if surface_name is None:
        pose = RelPose(body, init=True) # Treats as obstacle
    else:
        pose = create_relative_pose(world, obj_name, surface_name, init=True)
    return PoseDist(world, obj_name, DeltaDist(pose))

def create_observable_belief(world, **kwargs):
    with WorldSaver():
        return Belief(world, pose_dists={
            name: create_observable_pose_dist(world, name)
            for name in world.movable}, **kwargs)

def create_surface_pose_dist(world, obj_name, surface_dist, n=NUM_PARTICLES):
    placement_gen = get_stable_gen(world, max_attempts=100, learned=True,
                                   pos_scale=1e-3, rot_scale=1e-2)
    poses = []
    with LockRenderer():
        with BodySaver(world.get_body(obj_name)):
            while len(poses) < n:
                surface_name = surface_dist.sample()
                result = next(placement_gen(obj_name, surface_name), None)
                if result is None:
                    surface_dist = surface_dist.condition(lambda s: s != surface_name)
                else:
                    (rel_pose,) = result
                    poses.append(rel_pose)
    pose_dist = PoseDist(world, obj_name, UniformDist(poses))
    return pose_dist

def create_surface_belief(world, surface_dist, **kwargs):
    with WorldSaver():
        return Belief(world, pose_dists={
            name: create_surface_pose_dist(world, name, surface_dist)
            for name in world.movable}, **kwargs)

################################################################################

def compute_cfree(body, poses, obstacles=[]):
    cfree_poses = set()
    for pose in poses:
        pose.assign()
        if not any(pairwise_collision(body, obst) for obst in obstacles):
            cfree_poses.add(pose)
    return cfree_poses


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
    return visible_names

def observe_scene(world):
    visible_entities = are_visible(world)
    detections = {}
    # TODO: randomize robot's pose
    # TODO: probabilities based on whether in viewcone or not
    # TODO: sample from poses on table
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
    return detections

################################################################################

def fix_detections(belief, detections):
    # TODO: move directly to belief?
    world = belief.world
    fixed_detections = {}
    for name in detections:
        if name == belief.holding:
            continue
        body = world.get_body(name)
        for observed_pose in detections[name]:
            fixed_pose = world.fix_pose(name, observed_pose)
            set_pose(body, fixed_pose)
            support = world.get_supporting(name)
            if support is None:
                continue # Could also fix as relative to the world
            assert support is not None
            relative_pose = create_relative_pose(world, name, support, init=False)
            fixed_detections.setdefault(name, []).append(relative_pose)
            # relative_pose.assign()
    return fixed_detections

def transition_belief_update(belief, plan):
    if plan is None:
        return None
    # TODO: check that actually holding
    for action, params in plan:
        if action in ['move_base', 'move_arm', 'move_gripper', 'pull',
                      'calibrate', 'detect']:
            pass
        elif action == 'pick':
            o, p, g, rp = params[:4]
            del belief.pose_dists[o]
            belief.grasped = g
        elif action == 'place':
            o, p, g, rp = params[:4]
            belief.grasped = None
            belief.pose_dists[o] = PoseDist(belief.world, o, DeltaDist(rp))
        elif action == 'cook':
            pass
        else:
            raise NotImplementedError(action)
