import numpy as np
import random
import math

from collections import namedtuple
from itertools import islice

from sklearn.neighbors import KernelDensity

from examples.discrete_belief.dist import UniformDist, DDist, GaussianDistribution, \
    ProductDistribution, CUniformDist, DeltaDist
#from examples.pybullet.pr2_belief.primitives import get_observation_fn

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.pr2_primitives import Pose as WorldPose
from pybullet_tools.utils import point_from_pose, Ray, draw_point, RED, batch_ray_collision, draw_ray, wait_for_user, \
    CIRCULAR_LIMITS, stable_z_on_aabb, Point, Pose, Euler, set_pose, get_pose, BodySaver, \
    LockRenderer, multiply, remove_all_debug, base_values_from_pose, GREEN
from src.stream import get_stable_gen, test_supported
from src.utils import OPEN_SURFACES, compute_surface_aabb, KINECT_DEPTH

KITCHEN_FROM_ZED_LEFT = (
    (1.0600011348724365, 1.529999017715454, 0.5699998736381531),
    (-0.10374931246042252, 0.9274755120277405, -0.19101102650165558, -0.30420398712158203))
CAMERA_MATRIX = np.array(
    [[ 532.569,    0.,     320.,   ],
     [   0.,     532.569,  240.,   ],
     [   0.,       0.,       1.,   ]])
Particle = namedtuple('Particle', ['sample', 'weight'])

P_FALSE_POSITIVE = 0.0
P_FALSE_NEGATIVE = 0.0 # 0.1
POSITION_STD = 0.01
ORIENTATION_STD = np.pi / 8

# Prior on the number of false detections to ensure correlated

# TODO: could do open world or closed world
# For open world, can sum independent probabilities

# For a point, observation types
# outside cone, visible, occluded
# no detection, detection at point, detection elsewhere

# https://github.com/tlpmit/hpn
# https://github.mit.edu/tlp/bhpn

# TODO: how to factor visibility observation costs such that the appropriate actions are selected
# i.e. what to move out of the way

# TODO: use a proper probabilistic programming library rather than dist.py

################################################################################

# https://github.com/caelan/pddlstream/tree/stable/examples/discrete_belief
# https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_online.py
# https://github.mit.edu/caelan/stripstream/blob/master/robotics/openrave/belief_tamp.py
# https://github.mit.edu/caelan/ss/blob/master/belief/belief_online.py
# https://github.com/caelan/pddlstream/blob/stable/examples/pybullet/pr2_belief/run.py

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

def compute_normalization(particles):
    return sum(particle.weight for particle in particles)

def create_belief(world, entity_name, surface_dist, n=100):
    # TODO: halton seqeunce
    particles = []
    handles = []
    placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)

    with BodySaver(world.get_body(entity_name)):
        while len(particles) < n:
            surface_name = surface_dist.draw()
            #print(len(particles), surface_name)
            rel_pose, = next(placement_gen(entity_name, surface_name), (None,))
            if rel_pose is None:
                continue
            world_pose = WorldPose(rel_pose.body, rel_pose.get_world_from_body(), support=surface_name)
            point = point_from_pose(world_pose.value)
            weight = 1.
            particle = Particle(world_pose, weight=weight)
            particles.append(particle)
            #pose.assign()
            #wait_for_user()
            handles.extend(draw_point(point, color=RED))
    return particles

################################################################################

def compute_detectable(particles, camera_pose):
    ray_indices = set()
    for index, particle in enumerate(particles):
        point = point_from_pose(particle.sample.value)
        if is_visible_point(CAMERA_MATRIX, KINECT_DEPTH, point, camera_pose=camera_pose):
            ray_indices.add(index)
    return ray_indices

def compute_visible(particles, camera_pose, draw=True):
    ray_indices = compute_detectable(particles, camera_pose)
    rays = []
    camera_point = point_from_pose(camera_pose)
    for i in ray_indices:
        point = point_from_pose(particles[i].sample.value)
        rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    if draw:
        with LockRenderer():
            handles = []
            for ray, result in zip(rays, ray_results):
                handles.extend(draw_ray(ray, result))
    visible_indices = {index for index, result in zip(ray_indices, ray_results)
                       if result.objectUniqueId == -1}
    return visible_indices

################################################################################

def compute_density(particles, std=0.01):
    weighted_points = [Particle(point_from_pose(particle.sample.value)[:2], particle.weight)
                       for particle in particles]
    points, weights = zip(*weighted_points)
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

def observe_scene(world, camera_pose):
    # Could use an UKF to propagate a GMM
    visible_entities = are_visible(world, camera_pose)
    observations = {}
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
        observations.setdefault(visible_name, []).append(observed_pose)
        #wait_for_user()
        #set_pose(body, observed_pose)
    return observations

################################################################################

def get_detection_fn(poses, visible, p_fp=P_FALSE_POSITIVE, p_fn=P_FALSE_NEGATIVE):

    def fn(index):
        # P(detect | s in visible)
        # This could depend on the position as well
        if index in visible:
            return DDist({True: 1 - p_fn, False: p_fn})
        return DDist({True: p_fp, False: 1 - p_fp})
    return fn

def get_registration_fn(poses, visible, pos_std=POSITION_STD):
    # use distance to z for placing on difference surfaces
    # TODO: clip probabilities so doesn't become zero
    # TODO: nearby objects that might cause misdetections

    def fn(index, detection):
        # P(obs point | state detect)
        if detection:
            # TODO: proportional weight in the event that no normalization
            x, y, _ = point_from_pose(poses[index].value)
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
        particles = create_belief(world, entity_name, surface_dist)
    #pose = random.choice(particles).sample
    # np.random.choice(elements, size=10, replace=True, p=probabilities)
    # random.choice(elements, k=10, weights=probabilities)
    #pose.assign()

    # TODO: really want a piecewise distribution or something
    # The two observations do mimic how the examples are generated though

    poses = [particle.sample for particle in particles]
    samples_from_surface = {}
    for index, pose in enumerate(poses):
        samples_from_surface.setdefault(pose.support, set()).add(index)


    dist = DDist({i: particles[i].weight for i in range(len(poses))}).normalize()
    detections = observe_scene(world, camera_pose)
    has_detection = entity_name in detections
    pose_estimate = None
    if entity_name in detections:
        pose_estimate = base_values_from_pose(detections[entity_name][0])
    # TODO: each pose itself is hashable
    print(has_detection, pose_estimate)
    #dist.conditionOnVar(index=1, has_detection=True)

    field_of_view_indices = compute_detectable(particles, camera_pose)
    visible_indices = compute_visible(particles, camera_pose)
    #print(len(field_of_view_indices), len(visible_indices))
    assert set(visible_indices) <= set(field_of_view_indices)
    #obs_fn = get_observation_fn(surface)
    wait_for_user()

    detection_fn = get_detection_fn(poses, visible_indices)
    registration_fn = get_registration_fn(poses, visible_indices)
    #dist.obsUpdate(detection_fn, has_detection)
    dist.obsUpdates([detection_fn, registration_fn], [has_detection, pose_estimate])
    #dist = bayesEvidence(dist, detection_fn, has_detection) # projects out b and computes joint
    #joint_dist = JDist(dist, detection_fn, registration_fn)
    print(dist)

    remove_all_debug()

    with LockRenderer():
        handles = []
        z_offset = Point(z=0.1)
        for index in dist.support():
            # TODO: draw weights using color, length, or thickness
            point = point_from_pose(poses[index].value)
            color = GREEN if index == dist.mode() else RED
            handles.extend(draw_point(point + z_offset, color=color, width=1))
    wait_for_user()

    remove_all_debug()
    #norm_constant = compute_normalization(particles)
    for surface_name in samples_from_surface:
        #surface_particles = samples_from_surface[surface_name]
        #print(surface_name, compute_normalization(surface_particles) / norm_constant)
        surface_indices = samples_from_surface[surface_name] & set(dist.support())
        surface_particles = [Particle(poses[index], dist.prob(index)) for index in surface_indices]
        density = compute_density(surface_particles)
        predictions = list(islice(density_generator(
            world, entity_name, surface_name, density), n))
        wait_for_user()

    return particles
