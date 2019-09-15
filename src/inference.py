from __future__ import print_function

import numpy as np
import scipy

from collections import namedtuple
from scipy.stats import norm, truncnorm
from sklearn.neighbors import KernelDensity

from examples.discrete_belief.dist import UniformDist, DDist, DeltaDist, mixDDists, ProductDistribution, \
    GaussianDistribution, Distribution
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import base_values_from_pose, CIRCULAR_LIMITS, stable_z_on_aabb, point_from_pose, Point, Pose, \
    Euler, set_pose, multiply, draw_circle, LockRenderer, BodySaver, Ray, batch_ray_collision, draw_ray, wrap_angle, \
    circular_difference, remove_handles, get_pose, pairwise_collision, GREEN
from src.database import get_surface_reference_pose
from src.utils import compute_surface_aabb, create_relative_pose, CAMERA_MATRIX, KINECT_DEPTH, Z_EPSILON, test_supported

BAYESIAN = False
RESAMPLE = False

MODEL_P_FP, MODEL_P_FN = 0.0, 1e-4
MODEL_POS_STD, MODEL_ORI_STD = 0.01, np.pi / 8
#MODEL_POS_STD, MODEL_ORI_STD = OBS_POS_STD, OBS_ORI_STD

DIM = 2
assert DIM in (2, 3)

NUM_PARTICLES = 100 # 100 | 250
NEARBY_RADIUS = 5e-2
Neighborhood = namedtuple('Neighborhood', ['poses', 'prob'])

################################################################################

class PoseDist(object):
    # TODO: maintain one of these for each surface instead?
    # It is nice to treat them all as one distribution though
    def __init__(self, world, name, dist, weight=1.0, bandwidth=0.01):
        self.world = world
        self.name = name
        self.dist = dist
        self.poses_from_surface = {}
        for pose in self.dist.support():
            self.poses_from_surface.setdefault(pose.support, []).append(pose)
        self.surface_dist = self.dist.project(lambda p: p.support)
        self.density_from_surface = {}
        self.weight = weight
        self.bandwidth = bandwidth
        self.handles = []
    def is_localized(self):
        return len(self.dist.support()) == 1
    def surface_prob(self, surface):
        return self.weight * self.surface_dist.prob(surface)
    def discrete_prob(self, pose):
        return self.weight * self.dist.prob(pose)
    def prob(self, pose):
        support = pose.support
        density = self.get_density(support)
        pose2d = self.pose2d_from_pose(pose)
        [score] = density.score_samples([pose2d])
        prob = np.exp(-score)
        return self.surface_prob(support) * prob
    #def support(self):
    #    return self.dist.support()

    def pose2d_from_pose(self, pose):
        return base_values_from_pose(pose.get_reference_from_body())[:DIM]
    def pose_from_pose2d(self, pose2d, surface):
        #assert surface in self.poses_from_surface
        #reference_pose = self.poses_from_surface[surface][0]
        body = self.world.get_body(self.name)
        surface_aabb = compute_surface_aabb(self.world, surface)
        world_from_surface = get_surface_reference_pose(self.world.kitchen, surface)
        if DIM == 2:
            x, y = pose2d[:DIM]
            yaw = np.random.uniform(*CIRCULAR_LIMITS)
        else:
            x, y, yaw = pose2d
        z = stable_z_on_aabb(body, surface_aabb) + Z_EPSILON - point_from_pose(world_from_surface)[2]
        point = Point(x, y, z)
        surface_from_body = Pose(point, Euler(yaw=yaw))
        set_pose(body, multiply(world_from_surface, surface_from_body))
        return create_relative_pose(self.world, self.name, surface)

    def get_density(self, surface):
        if surface in self.density_from_surface:
            return self.density_from_surface[surface]
        if surface not in self.poses_from_surface:
            return None
        points, weights = zip(*[(self.pose2d_from_pose(pose), self.dist.prob(pose))
                                for pose in self.poses_from_surface[surface]])
        #print(weights)
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
        yaw_weight = 0.01*np.pi
        metric_weights = np.array([1., 1., yaw_weight]) # TODO: wrap around and symmetry?
        density = KernelDensity(bandwidth=self.bandwidth, algorithm='auto',
                                kernel='gaussian', metric="wminkowski", atol=0, rtol=0,
                                breadth_first=True, leaf_size=40,
                                metric_params={'p': 2, 'w': metric_weights[:DIM]})
        density.fit(X=points, sample_weight=1 * np.array(weights)) # Scaling doesn't seem to affect
        self.density_from_surface[surface] = density
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
        prob = sum(map(self.discrete_prob, poses))
        #poses = {target_pose}
        return Neighborhood(poses, prob)

    def sample_surface_pose(self, surface): # TODO: timeout
        density = self.get_density(surface)
        if density is None:
            return None
        assert surface is not None
        body = self.world.get_body(self.name)
        while True:
            [sample] = density.sample(n_samples=1)
            #[score] = density.score_samples([sample])
            #prob = np.exp(-score)
            pose = self.pose_from_pose2d(sample, surface)
            pose.assign()
            # TODO: additional obstacles
            if test_supported(self.world, body, surface):
                return pose # TODO: return prob?
    def sample_surface(self):
        return self.surface_dist.sample()
    def sample_discrete(self):
        return self.dist.sample()
    def sample(self):
        return self.sample_surface_pose(self.sample_surface())
    def resample(self, n=NUM_PARTICLES):
        if len(self.dist.support()) <= 1:
            return self
        with LockRenderer():
            poses = [self.sample() for _ in range(n)]
        new_dist = UniformDist(poses)
        return self.__class__(self.world, self.name, new_dist)
    def copy(self):
        return self.__class__(self.world, self.name, self.dist.copy())

    def decompose(self):
        if len(self.dist.support()) == 1:
            return self.dist.support()
        pose_dists = []
        for surface_name in self.surface_dist.support():
            dist = DDist({pose: self.discrete_prob(pose) for pose in self.poses_from_surface[surface_name]})
            weight = self.surface_prob(surface_name)
            pose_dists.append(SurfaceDist(self, weight, dist))
        return pose_dists
    def update_dist(self, observation, obstacles=[], verbose=False):
        # cfree_dist.conditionOnVar(index=1, has_detection=True)
        if not BAYESIAN and (self.name in observation):
            # TODO: convert into a Multivariate Gaussian
            [detected_pose] = observation[self.name]
            return DeltaDist(detected_pose)
        if not self.world.cameras:
            return self.dist.copy()
        body = self.world.get_body(self.name)
        all_poses = self.dist.support()
        cfree_poses = all_poses
        #cfree_poses = compute_cfree(body, all_poses, obstacles)
        #cfree_dist = self.cfree_dist
        cfree_dist = DDist({pose: self.dist.prob(pose) for pose in cfree_poses})
        # TODO: do these updates simultaneously for each object
        # TODO: check all camera poses
        [camera] = self.world.cameras.keys()
        info = self.world.cameras[camera]
        camera_pose = get_pose(info.body)
        detectable_poses = compute_detectable(cfree_poses, camera_pose)
        visible_poses = compute_visible(body, detectable_poses, camera_pose, draw=False)
        if verbose:
            print('Total: {} | CFree: {} | Detectable: {} | Visible: {}'.format(
                len(all_poses), len(cfree_poses), len(detectable_poses), len(visible_poses)))
        assert set(visible_poses) <= set(detectable_poses)
        # obs_fn = get_observation_fn(surface)
        #wait_for_user()
        return self.bayesian_belief_update(cfree_dist, visible_poses, observation, verbose=verbose)
    def bayesian_belief_update(self, prior_dist, visible_poses, observation, verbose=False):
        has_detection = self.name in observation
        detected_surface = None
        pose_estimate_2d = None
        if has_detection:
            [detected_pose] = observation[self.name]
            detected_surface = detected_pose.support
            pose_estimate_2d = self.pose2d_from_pose(detected_pose)
        else:
            for pose in visible_poses:
                pose.observations += 1
        if verbose:
            print('Detection: {} | Pose: {}'.format(has_detection, pose_estimate_2d))
        # TODO: could use an UKF to propagate a GMM
        new_dist = prior_dist.copy()
        # cfree_dist.obsUpdate(detection_fn, has_detection)
        new_dist.obsUpdates([
            get_detection_fn(visible_poses),
            get_registration_fn(visible_poses),
            # ], [has_detection, pose_estimate_2d])
        ], [detected_surface, pose_estimate_2d])
        # cfree_dist = bayesEvidence(cfree_dist, detection_fn, has_detection) # projects out b and computes joint
        # joint_dist = JDist(cfree_dist, detection_fn, registration_fn)
        return new_dist
    def update(self, belief, observation, n_samples=25, verbose=False, **kwargs):
        if verbose:
            print('Prior:', self.dist)
        body = self.world.get_body(self.name)
        obstacles = [self.world.get_body(name) for name in belief.pose_dists if name != self.name]
        dists = []
        for _ in range(n_samples):
            belief.sample(discrete=True)  # Trouble if no support
            with BodySaver(body):
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
        if RESAMPLE:
            pose_dist = pose_dist.resample()
        return pose_dist

    def dump(self):
        print(self.name, self.dist)
    def draw(self, color=GREEN, **kwargs):
        #if self.handles:
        #    return
        poses = list(self.dist.support())
        probs = list(map(self.discrete_prob, poses))
        alphas = np.linspace(0.0, 1.0, num=11, endpoint=True)
        percentiles = np.array([scipy.stats.scoreatpercentile(
            probs, 100 * p, interpolation_method='lower') for p in alphas])  # numpy.percentile
        max_prob = max(probs)
        #print('{}) max prob: {:.3f}'.format(self.name, max_prob))
        print('{}) #poses: {} | percentiles: {}'.format(self.name, len(poses), percentiles.round(3)))
        #remove_handles(self.handles)
        self.handles = []
        for pose, prob in zip(poses, probs):
            # TODO: could instead draw a circle
            fraction = prob / max_prob
            # TODO: draw weights using color, length, or thickness
            #color = GREEN if pose == self.dist.mode() else RED
            self.handles.extend(pose.draw(color=fraction*np.array(color), **kwargs))
        return self.handles
    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.name,
                                       self.surface_dist, len(self.dist.support()))

################################################################################

class SurfaceDist(PoseDist):
    def __init__(self, parent, weight, dist):
        super(SurfaceDist, self).__init__(parent.world, parent.world, dist, weight=weight)
        #self.parent = parent # No point if it evolves
        self.surface_name = self.dist.support()[0].support
    @property
    def support(self):
        return self.surface_name
    def project(self, fn):
        return self.__class__(self, self.weight, self.dist.project(fn))
    def __repr__(self):
        #return '{}({}, {})'.format(self.__class__.__name__, self.name, self.surface_name)
        return 'sd({})'.format(self.surface_name)

################################################################################

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


def compute_cfree(body, poses, obstacles=[]):
    cfree_poses = set()
    for pose in poses:
        pose.assign()
        if not any(pairwise_collision(body, obst) for obst in obstacles):
            cfree_poses.add(pose)
    return cfree_poses

################################################################################

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

# For a point, observation types
# outside cone, visible, occluded
# no detection, detection at point, detection elsewhere
# The two observation functions mimic how the examples are generated

def get_detection_fn(visible, p_fp=MODEL_P_FP, p_fn=MODEL_P_FN):
    # TODO: precompute visible here
    # TODO: mixture over ALL_SURFACES
    # Checking surfaces is important because incorrect surfaces may have similar relative poses
    assert p_fp == 0

    def fn(pose):
        # P(detect | s in visible)
        # This could depend on the position as well
        if pose in visible:
            return DDist({pose.support: 1. - p_fn, None: p_fn})
        return DeltaDist(None)
    return fn


def get_registration_fn(visible):
    # TODO: clip probabilities so doesn't become zero
    # TODO: nearby objects that might cause miss detections
    # TODO: add the observation as a particle

    def fn(pose, surface):
        # P(obs point | state detect)
        if surface is None:
            return DeltaDist(None)
        # Weight can be proportional weight in the event that the distribution can't be normalized
        x, y, yaw = base_values_from_pose(pose.get_reference_from_body())
        if DIM == 2:
            return ProductDistribution([
               GaussianDistribution(gmean=x, stdev=MODEL_POS_STD),
               GaussianDistribution(gmean=y, stdev=MODEL_POS_STD),
               #CUniformDist(-np.pi, +np.pi),
            ])
        else:
            return SE2Distribution(x, y, yaw, pos_std=MODEL_POS_STD, ori_std=MODEL_ORI_STD)
        # Could also mix with a uniform distribution over the space
        #if not visible[index]: uniform over the space
    return fn
