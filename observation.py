import numpy as np
import random

from collections import namedtuple
from itertools import islice

from sklearn.neighbors import KernelDensity

from examples.discrete_belief.dist import UniformDist, DDist, MixtureDist, gauss, GMU
#from examples.pybullet.pr2_belief.primitives import get_observation_fn

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import point_from_pose, Ray, draw_point, RED, batch_ray_collision, draw_ray, wait_for_user, \
    CIRCULAR_LIMITS, stable_z_on_aabb, Point, Pose, Euler, set_pose, get_pose
from stream import get_stable_gen, compute_surface_aabb, test_supported
from utils import SURFACES

KITCHEN_FROM_ZED_LEFT = (
    (1.0600011348724365, 1.529999017715454, 0.5699998736381531),
    (-0.10374931246042252, 0.9274755120277405, -0.19101102650165558, -0.30420398712158203))
CAMERA_MATRIX = np.array(
    [[ 532.569,    0.,     320.,   ],
     [   0.,     532.569,  240.,   ],
     [   0.,       0.,       1.,   ]])
DEPTH = 5.0
Particle = namedtuple('Particle', ['sample', 'weight'])

# Negative observations are easy. Just apply observation model
# Positive observations will give rise to a pose that is likely not within the samples
# Density estimation to recover the belief at a point
# https://scikit-learn.org/stable/modules/density.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde

# Discrete belief over surfaces. Particle filter for each surface
# Alternatively, just annotate poses with surfaces

# elements = [1.1, 2.2, 3.3]
# probabilities = [0.2, 0.5, 0.3]
# np.random.choice(elements, size=10, replace=True, p=probabilities)
# random.choice(elements, k=10, weights=probabilities)

# Prior on the number of false detections to ensure correlated

#from sklearn.mixture import GaussianMixture
# pip2 install -U --no-deps scikit-learn=0.20

# TODO: could do open world or closed world
# For open world, can sum independent probabilities


# For a point, observation types
# outside cone, visible, occluded
# no detection, detection at point, detection elsewhere

P_FALSE_POSITIVE = 0.0
P_FALSE_NEGATIVE = 0.0

# https://github.com/tlpmit/hpn
# https://github.mit.edu/tlp/bhpn

# TODO: how to factor visbility observation costs such that the appropriate actions are selected
# i.e. what to move out of the way

def are_visible(world, camera_pose):
    ray_names = []
    rays = []
    camera_point = point_from_pose(camera_pose)
    for name in world.movable:
        point = point_from_pose(get_pose(world.get_body(name)))
        if is_visible_point(CAMERA_MATRIX, DEPTH, point, camera_pose=camera_pose):
            ray_names.append(name)
            rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    return {name for name, result in zip(ray_names, ray_results)
            if result.objectUniqueId == world.get_body(name)}

def observe(world, camera_pose):
    # TODO: randomize robot's pose
    # Coarse pose estimation
    visible = are_visible(world, camera_pose)
    detections = {}
    for name in world.movable:
        body = world.get_body(name)
        pose = get_pose(body)
        if name in visible:
            if P_FALSE_NEGATIVE <= random.random():
                detections[name] = pose
        else:
            if P_FALSE_POSITIVE <= random.random():
                pass
            else:
                pass # TODO: sample from poses on table
                #detections[name] = pose
    # Would need to use UKF to handle GMM
    return detections

def get_observation_fn(p_look_fp=0, p_look_fn=0):
    # Observation: True/False for detection & numeric pose
    # use distance to z
    # TODO: clip probabilities so doesn't become zero
    def fn(is_visible, is_pose):
        # P(obs | s1=loc1, a=control_loc)
        #return MixtureDist()
        if is_visible and is_pose:
            return DDist({True: 1 - p_look_fn, False: p_look_fn})
        return DDist({True: p_look_fp, False: 1 - p_look_fp})
    return fn

def compute_normalization(particles):
    return sum(particle.weight for particle in particles)

def create_belief(world, entity_name, n=100):
    # TODO: halton seqeunce
    surfaces = SURFACES[1:2]
    surface_dist = UniformDist(surfaces)
    particles = []
    handles = []
    placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)
    # TODO: could just make the belief
    while len(particles) < n:
        surface_name = surface_dist.draw()
        print(len(particles), surface_name)
        pose, = next(placement_gen(entity_name, surface_name), (None,))
        if pose is None:
            continue
        point = point_from_pose(pose.value)
        weight = 1.
        particle = Particle(pose, weight=weight)
        particles.append(particle)
        #pose.assign()
        #wait_for_user()
        handles.extend(draw_point(point, color=RED))
    return particles

def compute_visible(particles, camera_pose):
    ray_indices = []
    rays = []
    camera_point = point_from_pose(camera_pose)
    for i, particle in enumerate(particles):
        point = point_from_pose(particle.sample.value)
        if is_visible_point(CAMERA_MATRIX, DEPTH, point, camera_pose=camera_pose):
            ray_indices.append(i)
            rays.append(Ray(camera_point, point))
    ray_results = batch_ray_collision(rays)
    handles = []
    for ray, result in zip(rays, ray_results):
        handles.extend(draw_ray(ray, result))
    wait_for_user()
    visible_indices = {index for index, result in zip(ray_indices, ray_results)
                       if result.objectUniqueId == -1}
    return visible_indices

def pose_generator(world, entity_name, surface_name, density):
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

def compute_density(particles):
    weighted_points = [Particle(point_from_pose(particle.sample.value)[:2], particle.weight)
                       for particle in particles]
    points, weights = zip(*weighted_points)

    std = 0.05  # meters
    density = KernelDensity(bandwidth=std, algorithm='auto',
                            kernel='gaussian', metric="euclidean", atol=0, rtol=0,
                            breadth_first=True, leaf_size=40, metric_params=None)
    density.fit(X=points, sample_weight=weights)
    # from scipy.stats.kde import gaussian_kde
    # density = gaussian_kde(points, weights=weights) # No weights in my scipy version
    return density

def test_observation(world, entity_name, camera_pose, n=100):
    # TODO: could just make the belief
    particles = create_belief(world, entity_name)
    pose = random.choice(particles).sample
    pose.assign()

    visible_entities = are_visible(world, camera_pose)
    print(visible_entities)


    visible_indices = compute_visible(particles, camera_pose)
    #obs_fn = get_observation_fn(surface)
    wait_for_user()

    samples_from_surface = {}
    for particle in particles:
        samples_from_surface.setdefault(particle.sample.support, []).append(particle)

    norm_constant = compute_normalization(particles)
    for surface_name in samples_from_surface:
        surface_particles = samples_from_surface[surface_name]
        weight = compute_normalization(surface_particles)
        print(surface_name, weight / norm_constant)
        density = compute_density(particles)
        predictions = list(islice(pose_generator(
            world, entity_name, surface_name, density), n))
        wait_for_user()

    wait_for_user()
    return particles