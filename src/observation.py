from __future__ import print_function

import numpy as np
import random
import math
import time

from collections import namedtuple
from scipy.stats import norm, truncnorm
from sklearn.neighbors import KernelDensity

#from pddlstream.utils import str_from_object
from examples.discrete_belief.dist import UniformDist, DDist, GaussianDistribution, \
    ProductDistribution, CUniformDist, DeltaDist, mixDDists, Distribution
#from examples.pybullet.pr2_belief.primitives import get_observation_fn
#from examples.pybullet.pr2_belief.problems import BeliefState, BeliefTask

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import point_from_pose, Ray, batch_ray_collision, draw_ray, wait_for_user, \
    CIRCULAR_LIMITS, stable_z_on_aabb, Point, Pose, Euler, set_pose, get_pose, BodySaver, \
    LockRenderer, multiply, remove_all_debug, base_values_from_pose, get_aabb_area, spaced_colors, WorldSaver, \
    pairwise_collision, elapsed_time, randomize, draw_circle, wrap_angle, circular_difference
from src.stream import get_stable_gen, test_supported, Z_EPSILON
from src.utils import compute_surface_aabb, KINECT_DEPTH, CAMERA_MATRIX, create_relative_pose
from src.database import get_surface_reference_pose

P_FALSE_POSITIVE = 0.0
#P_FALSE_NEGATIVE = 0.0
P_FALSE_NEGATIVE = 0.01
POSITION_STD = 0.01
ORIENTATION_STD = np.pi / 8
NUM_PARTICLES = 100
NEARBY_RADIUS = 5e-2

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

Neighborhood = namedtuple('Neighborhood', ['poses', 'prob'])


class PoseDist(object):
    def __init__(self, world, name, dist, std=0.01):
        self.world = world
        self.name = name
        self.dist = dist
        self.poses_from_surface = {}
        for pose in self.dist.support():
            self.poses_from_surface.setdefault(pose.support, []).append(pose)
        self.surface_dist = self.dist.project(lambda p: p.support)
        self.density_from_surface = {}
        self.std = std
    def surface_prob(self, surface):
        return self.surface_dist.prob(surface)
    def point_from_pose(self, pose):
        #return base_values_from_pose(pose.get_world_from_body())[:2]
        return base_values_from_pose(pose.get_reference_from_body())[:2]
    def get_density(self, surface):
        if surface in self.density_from_surface:
            return self.density_from_surface[surface]
        if surface not in self.poses_from_surface:
            return None
        points, weights = zip(*[(self.point_from_pose(pose), self.dist.prob(pose))
                                for pose in self.poses_from_surface[surface]])
        # from sklearn.mixture import GaussianMixture
        # pip2 install -U --no-deps scikit-learn=0.20
        # https://scikit-learn.org/stable/modules/density.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
        # KernelDensity kernel: ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        # KDTree.valid_metrics: ['chebyshev', 'euclidean', 'cityblock', 'manhattan', 'infinity', 'minkowski', 'p', 'l2', 'l1']
        # BallTree.valid_metrics: ['chebyshev', 'sokalmichener', 'canberra', 'haversine', 'rogerstanimoto', 'matching', 'dice', 'euclidean',
        # 'braycurtis', 'russellrao', 'cityblock', 'manhattan', 'infinity', 'jaccard', 'seuclidean', 'sokalsneath',
        # 'kulsinski', 'minkowski', 'mahalanobis', 'p', 'l2', 'hamming', 'l1', 'wminkowski', 'pyfunc']
        density = KernelDensity(bandwidth=self.std, algorithm='auto',
                                kernel='gaussian', metric="wminkowski", atol=0, rtol=0,
                                breadth_first=True, leaf_size=40,
                                metric_params={'p': 2, 'w': np.ones(2)})
        density.fit(X=points, sample_weight=1 * np.array(weights)) # Scaling doesn't seem to affect
        #print(weights)
        #scores = density.score_samples(points)
        #probabilities = np.exp(-scores)
        #print('Individual:', probabilities)
        #print(np.sum(probabilities))
        #total_score = density.score(points)
        #total_probability = np.exp(-total_score)
        #print('Total:', total_probability) # total log probability density
        #print(total_probability)
        # TODO: integrate to obtain a probability mass
        # from scipy.stats.kde import gaussian_kde
        # density = gaussian_kde(points, weights=weights) # No weights in my scipy version
        self.density_from_surface[surface] = density
        return density
    def get_nearby(self, target_pose, radius=NEARBY_RADIUS):
        # TODO: could instead use the probability density
        target_point = np.array(point_from_pose(target_pose.get_reference_from_body()))
        draw_circle(target_point, radius, parent=target_pose.reference_body,
                    parent_link=target_pose.reference_link)
        poses = set()
        for pose in self.dist.support():
            if target_pose.support != pose.support:
                continue
            point = point_from_pose(pose.get_reference_from_body())
            delta = target_point - point
            if np.linalg.norm(delta[:2]) < radius:
                poses.add(pose)
        prob = sum(map(self.dist.prob, poses))
        return Neighborhood(poses, prob)
    def pose_from_point(self, sample, surface):
        #assert surface in self.poses_from_surface
        #reference_pose = self.poses_from_surface[surface][0]
        body = self.world.get_body(self.name)
        surface_aabb = compute_surface_aabb(self.world, surface)
        world_from_surface = get_surface_reference_pose(self.world.kitchen, surface)
        z = stable_z_on_aabb(body, surface_aabb) + Z_EPSILON - point_from_pose(world_from_surface)[2]
        x, y = sample
        point = Point(x, y, z)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        surface_from_body = Pose(point, Euler(yaw=theta))
        set_pose(body, multiply(world_from_surface, surface_from_body))
        return create_relative_pose(self.world, self.name, surface)
    def sample_surface_pose(self, surface): # TODO: timeout
        density = self.get_density(surface)
        if density is None:
            return None
        body = self.world.get_body(self.name)
        while True:
            [sample] = density.sample(n_samples=1)
            [score] = density.score_samples([sample])
            prob = np.exp(-score) # TODO: return prob?
            pose = self.pose_from_point(sample, surface)
            pose.assign()
            # TODO: additional obstacles
            if test_supported(self.world, body, surface):
                return pose
    def sample(self):
        return self.sample_surface_pose(self.surface_dist.sample())
    def sample_support(self):
        return self.dist.sample()
    def update_dist(self, observation, obstacles=[], verbose=False):
        # TODO: include collisions with obstacles
        has_detection = self.name in observation.detections
        pose_estimate_2d = None
        if has_detection:
            [detected_pose] = observation.detections[self.name]
            pose_estimate_2d = base_values_from_pose(detected_pose.get_reference_from_body())
        if verbose:
            print('Detection: {} | Pose: {}'.format(has_detection, pose_estimate_2d))
        # cfree_dist.conditionOnVar(index=1, has_detection=True)
        body = self.world.get_body(self.name)
        all_poses = self.dist.support()
        #cfree_poses = all_poses
        cfree_poses = compute_cfree(body, all_poses, obstacles)
        #cfree_dist = self.cfree_dist
        cfree_dist = DDist({pose: self.dist.prob(pose) for pose in cfree_poses})
        # TODO: do these updates simultaneously for each object
        detectable_poses = compute_detectable(cfree_poses, observation.camera_pose)
        visible_poses = compute_visible(body, detectable_poses, observation.camera_pose, draw=False)
        if verbose:
            print('Total: {} | CFree: {} | Detectable: {} | Visible: {}'.format(
                len(all_poses), len(cfree_poses), len(detectable_poses), len(visible_poses)))
        assert set(visible_poses) <= set(detectable_poses)
        # obs_fn = get_observation_fn(surface)
        #wait_for_user()

        # TODO: could use an UKF to propagate a GMM
        new_dist = cfree_dist.copy()
        # cfree_dist.obsUpdate(detection_fn, has_detection)
        new_dist.obsUpdates([
            get_detection_fn(visible_poses),
            get_registration_fn(visible_poses),
        ], [has_detection, pose_estimate_2d])
        # cfree_dist = bayesEvidence(cfree_dist, detection_fn, has_detection) # projects out b and computes joint
        # joint_dist = JDist(cfree_dist, detection_fn, registration_fn)
        return new_dist
    def update(self, belief, observation, n=10, verbose=False, **kwargs):
        if verbose:
            print('Prior:', self.dist)
        obstacles = [self.world.get_body(name) for name in belief.pose_dists if name != self.name]
        dists = []
        for _ in range(n):
            belief.sample()
            with BodySaver(self.world.get_body(self.name)):
                new_dist = self.update_dist(observation, obstacles, **kwargs)
                #new_pose_dist = self.__class__(self.world, self.name, new_dist).resample()
            dists.append(new_dist)
            #remove_all_debug()
            #new_pose_dist.draw(color=belief.color_from_name[self.name])
            #wait_for_user()
        posterior = mixDDists({dist: 1./len(dists) for dist in dists})
        if verbose:
            print('Posterior:', posterior)
        pose_dist = self.__class__(self.world, self.name, posterior)
        #return pose_dist
        return pose_dist.resample()
    def resample(self, n=NUM_PARTICLES):
        if len(self.dist.support()) <= 1:
            return self
        with LockRenderer():
            poses = [self.sample() for _ in range(n)]
        new_dist = UniformDist(poses)
        return self.__class__(self.world, self.name, new_dist)
    def dump(self):
        print(self.name, self.dist)
    def draw(self, **kwargs):
        handles = []
        for pose in self.dist.support():
            # TODO: draw weights using color, length, or thickness
            #color = GREEN if pose == self.dist.mode() else RED
            handles.extend(pose.draw(**kwargs))
        return handles
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.name, self.surface_dist)

class BeliefState(object):
    def __init__(self, world, pose_dists={}, holding=None):
        self.world = world
        self.pose_dists = pose_dists
        self.holding = holding
        names = sorted(self.pose_dists)
        self.color_from_name = dict(zip(names, spaced_colors(len(names))))
        # TODO: belief fluents
    def update(self, observation, n=25):
        # Processing detected first
        # Could simply sample from the set of worlds and update
        # Would need to sample many worlds with name at different poses
        # Instead, let the moving object take on different poses
        start_time = time.time()
        order = [name for name in observation.detections]
        order.extend(set(self.pose_dists) - set(order))
        with LockRenderer():
            for name in order:
                self.pose_dists[name] = self.pose_dists[name].update(self, observation, n=n)
        print('Update time: {:.3f} sec for {} objects and {} samples'.format(
            elapsed_time(start_time), len(order), n))
        return self
    def sample(self):
        while True:
            poses = {}
            for name, pose_dist in randomize(self.pose_dists.items()):
                pose = pose_dist.sample()
                pose.assign()
                if any(pairwise_collision(self.world.get_body(name),
                                          self.world.get_body(other)) for other in poses):
                    break
                poses[name] = pose
            else:
                return poses
    def resample(self):
        for pose_dist in self.pose_dists:
            pose_dist.resample()
    def dump(self):
        print(self)
        for i, name in enumerate(sorted(self.pose_dists)):
            #self.pose_dists[name].dump()
            print(i, name, self.pose_dists[name])
    def draw(self, **kwargs):
        with LockRenderer():
            for name, pose_dist in self.pose_dists.items():
                pose_dist.draw(color=self.color_from_name[name], **kwargs)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, sorted(self.pose_dists))

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

def create_surface_belief(world, entity_name, surface_dist, n=NUM_PARTICLES):
    placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)
    poses = []
    with LockRenderer():
        with BodySaver(world.get_body(entity_name)):
            while len(poses) < n:
                surface_name = surface_dist.sample()
                rel_pose, = next(placement_gen(entity_name, surface_name), (None,))
                if rel_pose is not None:
                    poses.append(rel_pose)
    return PoseDist(world, entity_name, UniformDist(poses))

################################################################################

def compute_cfree(body, poses, obstacles=[]):
    cfree_poses = set()
    for pose in poses:
        pose.assign()
        if not any(pairwise_collision(body, obst) for obst in obstacles):
            cfree_poses.add(pose)
    return cfree_poses

def compute_detectable(poses, camera_pose):
    detectable_poses = set()
    for pose in poses:
        point = point_from_pose(pose.get_world_from_body())
        if is_visible_point(CAMERA_MATRIX, KINECT_DEPTH, point, camera_pose=camera_pose):
            detectable_poses.add(pose)
    return detectable_poses

def compute_visible(body, poses, camera_pose, draw=True):
    ordered_poses = list(poses)
    rays = []
    camera_point = point_from_pose(camera_pose)
    for pose in ordered_poses:
        point = point_from_pose(pose.get_world_from_body())
        rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    if draw:
        with LockRenderer():
            handles = []
            for ray, result in zip(rays, ray_results):
                handles.extend(draw_ray(ray, result))
    # Blocking objects will likely be known with high probability
    # TODO: move objects out of the way?
    return {pose for pose, result in zip(ordered_poses, ray_results)
            if result.objectUniqueId in (body, -1)}

################################################################################

class Observation(object):
    def __init__(self, camera_pose, detections):
        # TODO: camera name?
        self.camera_pose = camera_pose
        self.detections = detections
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, sorted(self.detections))

def observe_scene(world, camera_pose):
    visible_entities = are_visible(world, camera_pose)
    detections = {}
    # TODO: randomize robot's pose
    # TODO: probabilities based on whether in viewcone or not
    # TODO: sample from poses on table
    assert P_FALSE_POSITIVE == 0
    for name in visible_entities:
        # TODO: false positives
        if random.random() < P_FALSE_NEGATIVE:
            continue
        body = world.get_body(name)
        pose = get_pose(body)
        dx, dy = np.random.multivariate_normal(
            mean=np.zeros(2), cov=math.pow(POSITION_STD, 2)*np.eye(2))
        dyaw, = np.random.multivariate_normal(
            mean=np.zeros(1), cov=math.pow(ORIENTATION_STD, 2)*np.eye(1))
        noise_pose = Pose(Point(x=dx, y=dy), Euler(yaw=dyaw))
        observed_pose = multiply(pose, noise_pose)
        fixed_pose = world.fix_pose(name, observed_pose)
        set_pose(body, fixed_pose)
        support = world.get_supporting(name)
        assert support is not None
        relative_pose = create_relative_pose(world, name, support)
        #rp.assign()
        # Important to operate in relative pose land in case the drawer joint changes
        detections.setdefault(name, []).append(relative_pose)
    return Observation(camera_pose, detections)

################################################################################

# For a point, observation types
# outside cone, visible, occluded
# no detection, detection at point, detection elsewhere

# The two observations mimic how the examples are generated

class SE2Distribution(Distribution):
    def __init__(self, x=0., y=0., yaw=0.,
                 pos_std=1., ori_std=1.):
        self.x = x
        self.y = y
        self.yaw = wrap_angle(yaw)
        self.pos_std = pos_std
        self.ori_std = ori_std
    def prob(self, sample):
        x, y, yaw = sample
        dx = x - self.x
        dy = y - self.y
        dyaw = circular_difference(yaw, self.yaw)
        return norm.pdf(dx, scale=self.pos_std) * \
               norm.pdf(dy, scale=self.pos_std) * \
               truncnorm.pdf(dyaw, a=-np.pi, b=np.pi, scale=self.ori_std)
    def __repr__(self):
        return 'N({}, {})'.format(np.array([self.x, self.y, self.yaw]).round(3),
                                  np.array([self.pos_std, self.pos_std, self.ori_std]).round(3)) # Square?


def get_detection_fn(visible, p_fp=P_FALSE_POSITIVE, p_fn=P_FALSE_NEGATIVE):

    def fn(pose):
        # P(detect | s in visible)
        # This could depend on the position as well
        if pose in visible:
            return DDist({True: 1 - p_fn, False: p_fn})
        return DDist({True: p_fp, False: 1 - p_fp})
    return fn

def get_registration_fn(visible):
    # TODO: clip probabilities so doesn't become zero
    # TODO: nearby objects that might cause miss detections

    def fn(pose, detection):
        # TODO: compare the surface for the detection
        # P(obs point | state detect)
        if detection:
            # TODO: proportional weight in the event that no normalization
            x, y, yaw = base_values_from_pose(pose.get_reference_from_body())
            return SE2Distribution(x, y, yaw, pos_std=POSITION_STD, ori_std=ORIENTATION_STD)
            #return ProductDistribution([
            #    GaussianDistribution(gmean=x, stdev=pos_std),
            #    GaussianDistribution(gmean=y, stdev=pos_std),
            #    CUniformDist(-np.pi, +np.pi),
            #])
            # Could also mix with uniform over the space
            #if not visible[index]: uniform over the space
        return DeltaDist(None)
    return fn

################################################################################

def test_observation(world, entity_name):
    saver = WorldSaver()
    [camera_name] = list(world.cameras)
    camera_body, camera_matrix, camera_depth = world.cameras[camera_name]
    camera_pose = get_pose(camera_body)

    # TODO: estimate the fraction of the surface that is actually usable
    surfaces = ['indigo_tmp', 'range']
    surface_areas = {surface: get_aabb_area(compute_surface_aabb(world, surface))
                     for surface in surfaces}
    print('Areas:', surface_areas)
    surface_dist = DDist(surface_areas)
    #surface_dist = UniformDist(surfaces)
    print(surface_dist)

    belief = BeliefState(world, pose_dists={
        name: create_surface_belief(world, name, surface_dist) for name in world.movable})
    belief.dump()
    belief.draw()
    saver.restore()
    #for name in world.movable:
    #    set_pose(world.get_body(name), unit_pose())
    wait_for_user()
    remove_all_debug()

    saver.restore()
    observation = observe_scene(world, camera_pose)
    print(observation)
    belief = belief.update(observation)
    #belief.resample()

    belief.dump()
    belief.draw()
    saver.restore()
    wait_for_user()

    #for i in range(10):
    #    belief.sample()
    #    wait_for_user()

    for i in range(10):
        name = entity_name
        remove_all_debug()
        pose_dist = belief.pose_dists[name]
        target_pose = pose_dist.sample()
        poses, prob = pose_dist.get_nearby(target_pose)
        print('{}) {}, n={}, p={:.3f}'.format(i, name, len(poses), prob))
        for pose in poses:
            pose.draw(color=belief.color_from_name[name])
        wait_for_user()

    wait_for_user()
    remove_all_debug()

    #pose_dist.resample(n=n)
    #wait_for_user()
    #return pose_dist
