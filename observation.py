import numpy as np
from collections import namedtuple

from sklearn.neighbors import KernelDensity

from examples.discrete_belief.dist import UniformDist
from pybullet_tools.pr2_utils import is_visible_point
from pybullet_tools.utils import point_from_pose, Ray, draw_point, RED, batch_ray_collision, draw_ray, wait_for_user, \
    CIRCULAR_LIMITS, stable_z_on_aabb, Point, Pose, Euler, set_pose
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


def compute_normalization(particles):
    return sum(particle.weight for particle in particles)


def test(world, entity_name, camera_pose, n=100):
    entity_body = world.get_body(entity_name)
    surfaces = SURFACES
    surface_dist = UniformDist(surfaces)
    particles = []
    handles = []
    placement_gen = get_stable_gen(world, learned=True, pos_scale=1e-3, rot_scale=1e-2)

    # TODO: could just make the belief

    rays = []
    camera_point = point_from_pose(camera_pose)
    while len(particles) < n:
        surface_name = surface_dist.draw()
        print(len(particles), surface_name)
        pose, = next(placement_gen(entity_name, surface_name), (None,))
        if pose is None:
            continue
        point = point_from_pose(pose.value)


        if not is_visible_point(CAMERA_MATRIX, DEPTH, point, camera_pose=camera_pose):
            continue

        rays.append(Ray(camera_point, point))

        weight = 1.
        particle = Particle(pose, weight=weight)
        particles.append(particle)
        #pose.assign()
        #wait_for_user()
        handles.extend(draw_point(point, color=RED))

    ray_results = batch_ray_collision(rays)
    for ray, result in zip(rays, ray_results):
        draw_ray(ray, result)
    wait_for_user()




    samples_from_surface = {}
    for particle in particles:
        samples_from_surface.setdefault(particle.sample.support, []).append(particle)

    norm_constant = compute_normalization(particles)
    for surface_name in samples_from_surface:
        surface_particles = samples_from_surface[surface_name]
        weight = compute_normalization(surface_particles)

        # Pose

        weighted_points = [Particle(point_from_pose(particle.sample.value)[:2], particle.weight)
                           for particle in surface_particles]
        points, weights = zip(*weighted_points)

        std = 0.05 # meters
        density = KernelDensity(bandwidth=std, algorithm='auto',
                                kernel='gaussian', metric="euclidean", atol=0, rtol=0,
                                breadth_first=True, leaf_size=40, metric_params=None)
        density.fit(X=points, sample_weight=weights)

        theta = np.random.uniform(*CIRCULAR_LIMITS)
        surface_aabb = compute_surface_aabb(world, surface_name)
        z = stable_z_on_aabb(entity_body, surface_aabb)
        print(z, theta)

        predictions = []

        while len(predictions) < n:
            [sample] = density.sample(n_samples=1)
            [score] = density.score_samples([sample])
            prob = np.exp(-score)
            print(prob)
            x, y = sample
            point = Point(x, y, z)
            pose = Pose(point, Euler(yaw=theta))
            set_pose(entity_body, pose)
            if test_supported(world, entity_body, surface_name):
                predictions.append(pose)
                handles.extend(draw_point(point, color=RED))
        wait_for_user()

        #from scipy.stats.kde import gaussian_kde
        #density = gaussian_kde(points, weights=weights) # No weights in my scipy version
        #print(density)

        #test_supported


        print(surface_name, weight / norm_constant)

    wait_for_user()
    return particles