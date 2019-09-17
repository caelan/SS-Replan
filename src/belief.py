from __future__ import print_function

import time
import math
import copy

from pddlstream.utils import str_from_object
from examples.discrete_belief.dist import UniformDist, DeltaDist
#from examples.discrete_belief.run import continue_mdp_cost
#from examples.pybullet.pr2_belief.primitives import get_observation_fn
#from examples.pybullet.pr2_belief.problems import BeliefState, BeliefTask

#from examples.discrete_belief.run import geometric_cost
from pybullet_tools.utils import BodySaver, joint_from_name, LockRenderer, spaced_colors, WorldSaver, \
    pairwise_collision, elapsed_time, randomize, remove_handles, wait_for_duration, wait_for_user, \
    get_joint_positions, get_joint_name, get_joint_position, GREEN
from src.command import State, TIN_OBJECTS
from src.inference import NUM_PARTICLES, PoseDist
from src.observe import fix_detections, relative_detections, ELSEWHERE
from src.stream import get_stable_gen
from src.utils import create_relative_pose, RelPose, FConf, are_confs_close, type_from_name

# TODO: prior on the number of false detections to ensure correlated
# TODO: could do open world or closed world. For open world, can sum independent probabilities
# TODO: use a proper probabilistic programming library rather than dist.py
# TODO: the really can just be viewed as a change in the kernel applied

# Detect preconditions and cost
# * Most general would be conditioning the success prob on the full state via a cost
# * Does not admit factoring though
# * Instead, produce a detection for a subset of the region
# * Preconditions involving the likelihood something is interfering

# https://github.com/tlpmit/hpn
# https://github.mit.edu/tlp/bhpn
# https://github.com/caelan/pddlstream/tree/stable/examples/discrete_belief
# https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_online.py
# https://github.mit.edu/caelan/stripstream/blob/master/robotics/openrave/belief_tamp.py
# https://github.mit.edu/caelan/ss/blob/master/belief/belief_online.py
# https://github.com/caelan/pddlstream/blob/stable/examples/pybullet/pr2_belief/run.py

MIN_GRASP_WIDTH = 0.005
REPAIR_DETECTIONS = True
STOCHASTIC_PLACE = False

################################################################################

# TODO: point estimates and confidence intervals/regions
# TODO: mixture between discrete and growing Mixture of Gaussian
# TODO: belief fluents

ARM_TOLERANCE = math.radians(2) # Can afford to be larger (just move to new initial conf)
GRIPPER_TOLERANCE = 1e-2

class Belief(object):
    def __init__(self, world, pose_dists={}, grasped=None):
        self.world = world
        self.pose_dists = pose_dists
        self.grasped = grasped # grasped or holding?
        #colors = spaced_colors(len(self.objects))
        colors = [GREEN]*len(self.objects)
        self.color_from_name = dict(zip(self.objects, colors))
        self.observations = []
        self.handles = []

        # TODO: store state history
        self.base_conf = None
        self.arm_conf = None
        self.gripper_conf = None
        self.door_confs = {}
        self.pressed = set()
        self.cooked = set()
        self.liquid = set()
    def update_state(self):
        # TODO: apply this directly from observations
        # No use applying this to base confs
        self.base_conf = FConf(self.world.robot, self.world.base_joints, init=True)
        arm_conf = FConf(self.world.robot, self.world.arm_joints, init=True)
        if (self.arm_conf is None) or not are_confs_close(arm_conf, self.arm_conf, tol=ARM_TOLERANCE):
            self.arm_conf = arm_conf
        else:
            print('At anticipated arm conf')
        gripper_conf = FConf(self.world.robot, self.world.gripper_joints, init=True)
        if (self.gripper_conf is None) or not are_confs_close(gripper_conf, self.gripper_conf, tol=GRIPPER_TOLERANCE):
            self.gripper_conf = gripper_conf
        else:
            print('At anticipated gripper conf')

        # TODO: do I still need to test if the current values are equal to the last ones?
        for joint in self.world.kitchen_joints:
            name = get_joint_name(self.world.kitchen, joint)
            position = get_joint_position(self.world.kitchen, joint)
            self.update_door_conf(name, position)
            self.update_door_conf(name, position)
        #wait_for_user()
        return self.check_consistent()
    def update_door_conf(self, name, position):
        joint = joint_from_name(self.world.kitchen, name)
        conf = FConf(self.world.kitchen, [joint], [position], init=True)
        if (name not in self.door_confs) or not are_confs_close(conf, self.door_confs[name], tol=1e-3):
            # TODO: different threshold for drawers and doors
            self.door_confs[name] = conf
        else:
            print('At anticipated conf for door {}'.format(name))
        return self.door_confs[name]
    @property
    def holding(self):
        if self.grasped is None:
            return None
        return self.grasped.body_name
    @property
    def placed(self):
        return sorted(set(self.pose_dists.keys()))
    @property
    def objects(self):
        objects = set(self.placed)
        if self.holding is not None:
            objects.add(self.holding)
        return sorted(objects)
    def is_gripper_closed(self):
        # TODO: base this on the object type
        if self.holding is not None:
            obj_type = type_from_name(self.holding)
            if obj_type not in TIN_OBJECTS:
                return False
        # each joint in [0.00, 0.04] (units coincide with meters on the physical gripper)
        current_gq = get_joint_positions(self.world.robot, self.world.gripper_joints)
        gripper_width = sum(current_gq)
        return gripper_width <= MIN_GRASP_WIDTH
    def check_consistent(self):
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/d1e6024c5c13df7edeab3a271b745e656a794b02/control_tools/execution.py#L163
        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/pr2_controller.py#L93
        # https://github.mit.edu/caelan/mudfish/blob/master/scripts/planner.py#L346
        if (self.grasped is not None) and self.is_gripper_closed():
            # TODO: need to add the grasped object back into the dist
            self.grasped = None
            print('Inconsistent belief!')
            return False
        return True
    def update(self, detections, n_samples=25):
        start_time = time.time()
        self.observations.append(detections)
        # Processing detected first
        # Could simply sample from the set of worlds and update
        # Would need to sample many worlds with name at different poses
        # Instead, let the moving object take on different poses
        with LockRenderer():
            with WorldSaver():
                if REPAIR_DETECTIONS:
                    detections = fix_detections(self, detections) # TODO: skip if in sim
                detections = relative_detections(self, detections)
                order = [name for name in detections]  # Detected
                order.extend(set(self.pose_dists) - set(order))  # Not detected
                for name in order:
                    self.pose_dists[name] = self.pose_dists[name].update(
                        self, detections, n_samples=n_samples)
        self.update_state()
        print('Update time: {:.3f} sec for {} objects and {} samples'.format(
            elapsed_time(start_time), len(order), n_samples))
        return self

    def sample(self, discrete=True):
        # TODO: timeout if unable to find
        while True:
            poses = {}
            for name, pose_dist in randomize(self.pose_dists.items()):
                body = self.world.get_body(name)
                pose = pose_dist.sample_discrete() if discrete else pose_dist.sample()
                pose.assign()
                if any(pairwise_collision(body, self.world.get_body(other)) for other in poses):
                    break
                poses[name] = pose
            else:
                return poses

    def sample_state(self, **kwargs):
        pose_from_name = self.sample(**kwargs)
        world_saver = WorldSaver()
        attachments = []
        for pose in pose_from_name.values():
            attachments.extend(pose.confs)
        if self.grasped is not None:
            attachments.append(self.grasped.get_attachment())
        return State(self.world, savers=[world_saver], attachments=attachments)
    #def resample(self):
    #    for pose_dist in self.pose_dists:
    #        pose_dist.resample() # Need to update distributions
    def dump(self):
        print(self)
        for i, name in enumerate(sorted(self.pose_dists)):
            #self.pose_dists[name].dump()
            print(i, name, self.pose_dists[name])
    def draw(self, **kwargs):
        with LockRenderer(True):
            remove_handles(self.handles)
            self.handles = []
            with WorldSaver():
                for name, pose_dist in self.pose_dists.items():
                    self.handles.extend(pose_dist.draw(
                            color=self.color_from_name[name], **kwargs))
    def __repr__(self):
        return '{}(holding={}, placed={})'.format(self.__class__.__name__, self.holding, str_from_object(
            {name: self.pose_dists[name].surface_dist for name in self.placed}))

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
        belief = Belief(world, pose_dists={
            name: create_observable_pose_dist(world, name)
            for name in world.movable}, **kwargs)
        belief.task = world.task
        return belief

def create_surface_pose_dist(world, obj_name, surface_dist, n=NUM_PARTICLES):
    # TODO: likely easier to just make a null surface below ground
    placement_gen = get_stable_gen(world, max_attempts=100, learned=True,
                                   pos_scale=1e-3, rot_scale=1e-2)
    poses = []
    with LockRenderer():
        with BodySaver(world.get_body(obj_name)):
            while len(poses) < n:
                surface_name = surface_dist.sample()
                assert surface_name is not ELSEWHERE
                result = next(placement_gen(obj_name, surface_name), None)
                if result is None:
                    surface_dist = surface_dist.condition(lambda s: s != surface_name)
                else:
                    (rel_pose,) = result
                    rel_pose.init = True
                    poses.append(rel_pose)
    return PoseDist(world, obj_name, UniformDist(poses))

def create_surface_belief(world, surface_dists, **kwargs):
    with WorldSaver():
        belief = Belief(world, pose_dists={
            name: create_surface_pose_dist(world, name, surface_dist)
            for name, surface_dist in surface_dists.items()}, **kwargs)
        belief.task = world.task
        return belief

################################################################################

def delocalize_belief(belief, o, rp):
    dist = UniformDist([rp, copy.copy(rp)])
    belief.pose_dists[o] = PoseDist(belief.world, o, dist)
    return dist

def transition_belief_update(belief, plan):
    if plan is None:
        return False
    success = True
    for action, params in plan:
        if action in ['move_base', 'calibrate', 'detect']:
            pass
        elif action == 'press-on':
            s, k, o, bq, aq, gq, at = params
            belief.pressed.add(k)
            belief.cooked.add(o)
            for bowl, liquid in belief.liquid:
                if bowl == o:
                    belief.cooked.add(liquid)
        elif action == 'press-off':
            s, k, o, bq, aq, gq, at = params
            belief.pressed.remove(k)
        elif action == 'move_arm':
            bq, aq1, aq2, at = params
            belief.arm_conf = aq2
        elif action == 'move_gripper':
            gq1, gq2, gt = params
            belief.gripper_conf = gq2
        elif action == 'pull':
            j, a1, a2, o, wp1, wp2, bq, aq1, aq2, gq, at = params
            belief.door_confs[j] = a2
            belief.arm_conf = aq2
        elif action == 'pour':
            bowl, wp, cup, g, liquid, bq, aq, at = params
            belief.liquid.discard((cup, liquid))
            belief.liquid.add((bowl, liquid))
        elif action == 'pick':
            o, p, g, rp = params[:4]
            obj_type = type_from_name(o)
            if (obj_type not in TIN_OBJECTS) or not belief.is_gripper_closed():
                del belief.pose_dists[o]
                belief.grasped = g
                # TODO: open gripper afterwards to ensure not in hand
            else:
                delocalize_belief(belief, o, rp)
                print('Failed to grasp! Delocalizing belief')
                success = False
                break
        elif action == 'place':
            o, p, g, rp = params[:4]
            belief.grasped = None
            if STOCHASTIC_PLACE and belief.world.is_real():
                delocalize_belief(belief, o, rp)
            else:
                belief.pose_dists[o] = PoseDist(belief.world, o, DeltaDist(rp))
        elif action == 'cook':
            pass
        else:
            raise NotImplementedError(action)
    # TODO: replan after every action
    return success
