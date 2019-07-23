from __future__ import print_function

import numpy as np
import random
import math

from collections import namedtuple
from itertools import islice

from sklearn.neighbors import KernelDensity

from examples.discrete_belief.dist import UniformDist, DDist, GaussianDistribution, \
    ProductDistribution, CUniformDist, DeltaDist
#from examples.pybullet.pr2_belief.primitives import get_observation_fn
#from examples.pybullet.pr2_belief.problems import BeliefState, BeliefTask

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.pr2_primitives import Pose as WorldPose
from pybullet_tools.utils import point_from_pose, Ray, draw_point, RED, batch_ray_collision, draw_ray, wait_for_user, \
    CIRCULAR_LIMITS, stable_z_on_aabb, Point, Pose, Euler, set_pose, get_pose, BodySaver, \
    LockRenderer, multiply, remove_all_debug, base_values_from_pose, GREEN, unit_generator
from src.stream import get_stable_gen, test_supported, RelPose
from src.utils import OPEN_SURFACES, compute_surface_aabb, KINECT_DEPTH, CAMERA_MATRIX

P_FALSE_POSITIVE = 0.0
P_FALSE_NEGATIVE = 0.0 # 0.1
POSITION_STD = 0.01
ORIENTATION_STD = np.pi / 8

# Prior on the number of false detections to ensure correlated

# TODO: could do open world or closed world
# For open world, can sum independent probabilities

# TODO: how to factor visibility observation costs such that the appropriate actions are selected
# i.e. what to move out of the way

# TODO: use a proper probabilistic programming library rather than dist.py

################################################################################

# https://github.com/tlpmit/hpn
# https://github.mit.edu/tlp/bhpn

# https://github.com/caelan/pddlstream/tree/stable/examples/discrete_belief
# https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_online.py
# https://github.mit.edu/caelan/stripstream/blob/master/robotics/openrave/belief_tamp.py
# https://github.mit.edu/caelan/ss/blob/master/belief/belief_online.py
# https://github.com/caelan/pddlstream/blob/stable/examples/pybullet/pr2_belief/run.py

# TODO: symbol for elsewhere pose

class PoseDist(object):
    def __init__(self, world, name, dist):
        self.world = world
        self.name = name
        self.dist = dist
    def poses_from_surface(self):
        poses_from_surface = {}
        for pose in self.dist.support():
            poses_from_surface.setdefault(pose.support, set()).add(pose)
        return poses_from_surface
    def update(self, observation):
        has_detection = self.name in observation.detections
        pose_estimate = None
        if has_detection:
            pose_estimate = base_values_from_pose(observation.detections[self.name][0])
        print('Detection: {} | Pose: {}'.format(has_detection, pose_estimate))
        # dist.conditionOnVar(index=1, has_detection=True)

        poses = self.dist.support()
        # TODO: do these updates simultaneously for each object
        detectable_poses = compute_detectable(poses, observation.camera_pose)
        visible_poses = compute_visible(poses, observation.camera_pose)
        print('Total: {} | Detectable: {} | Visible: {}'.format(
            len(poses), len(detectable_poses), len(visible_poses)))
        assert set(visible_poses) <= set(detectable_poses)
        # obs_fn = get_observation_fn(surface)
        wait_for_user()

        print('Prior:', self.dist)
        detection_fn = get_detection_fn(poses, visible_poses)
        registration_fn = get_registration_fn(poses, visible_poses)
        new_dist = self.dist.copy()
        # dist.obsUpdate(detection_fn, has_detection)
        new_dist.obsUpdates([detection_fn, registration_fn], [has_detection, pose_estimate])
        # dist = bayesEvidence(dist, detection_fn, has_detection) # projects out b and computes joint
        # joint_dist = JDist(dist, detection_fn, registration_fn)
        print('Posterior:', new_dist)
        return self.__class__(self.world, self.name, new_dist)
    def resample(self):
        pass
    def draw(self):
        handles = []
        for pose in self.dist.support():
            # TODO: draw weights using color, length, or thickness
            color = GREEN if pose == self.dist.mode() else RED
            handles.extend(pose.draw(color=color, width=1))
        return handles

class BeliefState(object):
    def __init__(self, world, pose_dists={}, holding=None):
        self.world = world
        self.pose_dists = pose_dists
        self.holding = holding
        # TODO: belief fluents

    def update(self, observation):
        for pose_dist in self.pose_dists:
            pose_dist.update(observation)
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__)

################################################################################

def are_visible(world, camera_pose):
    ray_names = []
    rays = []
    camera_point = point_from_pose(camera_pose)
    for name in world.movable:
        point = point_from_pose(get_pose(world.get_body(name)))
        if is_visible_point(CAMERA_MATRIX, KINECT_DEPTH, point, camera_pose=camera_pose):
            ray_names.append(name)
            rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    return {name for name, result in zip(ray_names, ray_results)
            if result.objectUniqueId == world.get_body(name)}

def create_belief(world, entity_name, surface_dist, n=100):
    placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)
    handles = []
    placements = []
    with BodySaver(world.get_body(entity_name)):
        while len(placements) < n:
            surface_name = surface_dist.draw()
            #print(len(particles), surface_name)
            rel_pose, = next(placement_gen(entity_name, surface_name), (None,))
            if rel_pose is None:
                continue
            placements.append(rel_pose)
            #pose.assign()
            #wait_for_user()
            handles.extend(rel_pose.draw(color=RED))
    dist = UniformDist(placements)
    return PoseDist(world, entity_name, dist)

################################################################################

def compute_detectable(poses, camera_pose):
    detectable_poses = set()
    for pose in poses:
        point = point_from_pose(pose.get_world_from_body())
        if is_visible_point(CAMERA_MATRIX, KINECT_DEPTH, point, camera_pose=camera_pose):
            detectable_poses.add(pose)
    return detectable_poses

def compute_visible(placements, camera_pose, draw=True):
    detectable_poses = list(compute_detectable(placements, camera_pose))
    rays = []
    camera_point = point_from_pose(camera_pose)
    for pose in detectable_poses:
        point = point_from_pose(pose.get_world_from_body())
        rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    if draw:
        with LockRenderer():
            handles = []
            for ray, result in zip(rays, ray_results):
                handles.extend(draw_ray(ray, result))
    return {pose for pose, result in zip(detectable_poses, ray_results)
            if result.objectUniqueId == -1}

################################################################################

def compute_density(dist, poses, std=0.01):
    points, weights = zip(*{point_from_pose(pose.get_world_from_body())[:2]: dist.prob(pose)
                            for pose in poses if 0 < dist.prob(pose)}.items())
    # from sklearn.mixture import GaussianMixture
    # pip2 install -U --no-deps scikit-learn=0.20

    # TODO: compute area of each surface and use to estimate the total area and the samples to cover the space

    # https://scikit-learn.org/stable/modules/density.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    #KernelDensity kernel
    # ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    #KDTree.valid_metrics
    #['chebyshev', 'euclidean', 'cityblock', 'manhattan', 'infinity', 'minkowski', 'p', 'l2', 'l1']
    #BallTree.valid_metrics
    #['chebyshev', 'sokalmichener', 'canberra', 'haversine', 'rogerstanimoto', 'matching', 'dice', 'euclidean',
    # 'braycurtis', 'russellrao', 'cityblock', 'manhattan', 'infinity', 'jaccard', 'seuclidean', 'sokalsneath',
    # 'kulsinski', 'minkowski', 'mahalanobis', 'p', 'l2', 'hamming', 'l1', 'wminkowski', 'pyfunc']

    density = KernelDensity(bandwidth=std, algorithm='auto',
                            kernel='gaussian', metric="wminkowski", atol=0, rtol=0,
                            breadth_first=True, leaf_size=40, metric_params={'p': 2, 'w': np.ones(2)})
    density.fit(X=points, sample_weight=weights)
    # from scipy.stats.kde import gaussian_kde
    # density = gaussian_kde(points, weights=weights) # No weights in my scipy version
    return density

def density_generator(world, entity_name, surface_name, density):
    entity_body = world.get_body(entity_name)
    surface_aabb = compute_surface_aabb(world, surface_name)
    z = stable_z_on_aabb(entity_body, surface_aabb)

    handles = []
    while True:
        [sample] = density.sample(n_samples=1)
        #[score] = density.score_samples([sample])
        #prob = np.exp(-score)
        x, y = sample
        point = Point(x, y, z)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        pose = Pose(point, Euler(yaw=theta))
        set_pose(entity_body, pose)
        # TODO: additional obstacles
        if test_supported(world, entity_body, surface_name):
            handles.extend(draw_point(point, color=RED))
            yield pose

################################################################################

class Observation(object):
    def __init__(self, camera_pose, detections):
        # TODO: camera name?
        self.camera_pose = camera_pose
        self.detections = detections

def observe_scene(world, camera_pose):
    # TODO: could use an UKF to propagate a GMM
    visible_entities = are_visible(world, camera_pose)
    detections = {}
    # TODO: randomize robot's pose
    # TODO: false positives
    # TODO: difference probabilities based on whether in viewcone or not
    # TODO: sample from poses on table
    assert P_FALSE_POSITIVE == 0
    for visible_name in visible_entities:
        if random.random() < P_FALSE_NEGATIVE:
            continue
        body = world.get_body(visible_name)
        pose = get_pose(body)
        dx, dy = np.random.multivariate_normal(
            mean=np.zeros(2), cov=math.pow(POSITION_STD, 2)*np.eye(2))
        dyaw, = np.random.multivariate_normal(
            mean=np.zeros(1), cov=math.pow(ORIENTATION_STD, 2)*np.eye(1))
        noise_pose = Pose(Point(x=dx, y=dy), Euler(yaw=dyaw))
        observed_pose = multiply(pose, noise_pose)
        detections.setdefault(visible_name, []).append(observed_pose)
        #wait_for_user()
        #set_pose(body, observed_pose)
    return Observation(camera_pose, detections)

################################################################################

# For a point, observation types
# outside cone, visible, occluded
# no detection, detection at point, detection elsewhere

def get_detection_fn(poses, visible, p_fp=P_FALSE_POSITIVE, p_fn=P_FALSE_NEGATIVE):

    def fn(pose):
        # P(detect | s in visible)
        # This could depend on the position as well
        if pose in visible:
            return DDist({True: 1 - p_fn, False: p_fn})
        return DDist({True: p_fp, False: 1 - p_fp})
    return fn

def get_registration_fn(poses, visible, pos_std=POSITION_STD):
    # use distance to z for placing on difference surfaces
    # TODO: clip probabilities so doesn't become zero
    # TODO: nearby objects that might cause misdetections

    def fn(pose, detection):
        # P(obs point | state detect)
        if detection:
            # TODO: proportional weight in the event that no normalization
            world_pose = poses[pose].get_world_from_body()
            x, y, _ = point_from_pose(world_pose)
            return ProductDistribution([
                GaussianDistribution(gmean=x, stdev=pos_std),
                GaussianDistribution(gmean=y, stdev=pos_std),
                CUniformDist(-np.pi, +np.pi),
            ])
            # Could also mix with uniform over the space
            #if not visible[index]: uniform over the space
        else:
            return DeltaDist(None)
    return fn

################################################################################

def test_observation(world, entity_name, n=100):
    [camera_name] = list(world.cameras)
    camera_body, camera_matrix, camera_depth = world.cameras[camera_name]
    camera_pose = get_pose(camera_body)
    surface_dist = UniformDist(OPEN_SURFACES[1:2])
    with LockRenderer():
        pose_dist = create_belief(world, entity_name, surface_dist)
    #pose = random.choice(particles).sample
    # np.random.choice(elements, size=10, replace=True, p=probabilities)
    # random.choice(elements, k=10, weights=probabilities)
    #pose.assign()

    # TODO: really want a piecewise distribution or something
    # The two observations do mimic how the examples are generated though

    observation = observe_scene(world, camera_pose)
    pose_dist = pose_dist.update(observation)

    remove_all_debug()
    with LockRenderer():
        pose_dist.draw()
    wait_for_user()
    remove_all_debug()

    samples_from_surface = pose_dist.poses_from_surface()
    for surface_name in samples_from_surface:
        #surface_particles = samples_from_surface[surface_name]
        #print(surface_name, compute_normalization(surface_particles) / norm_constant)
        surface_poses = samples_from_surface[surface_name] & set(pose_dist.dist.support())
        density = compute_density(pose_dist.dist, surface_poses)
        predictions = list(islice(density_generator(
            world, entity_name, surface_name, density), n))
        wait_for_user()

    return pose_dist
