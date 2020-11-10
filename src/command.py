from __future__ import print_function

import math
import numpy as np
import time

from pybullet_tools.utils import get_moving_links, set_joint_positions, create_attachment, \
    wait_for_duration, flatten_links, remove_handles, \
    batch_ray_collision, draw_ray, wait_for_user, WorldSaver, adjust_path, waypoints_from_path
from pybullet_tools.retime import interpolate_path, decompose_into_paths
from src.utils import create_surface_attachment, SPAM, TOMATO_SOUP, MUSTARD, SUGAR, CHEEZIT, DEBUG

DEFAULT_TIME_STEP = 0.02
DEFAULT_SLEEP = 0.5
FORCE = 50 # 20 | 50 | 100

TIN_EFFORT = 60
PLASTIC_EFFORT = 50
CARDBOARD_EFFORT = 60

TIN_OBJECTS = [SPAM]

EFFORT_FROM_OBJECT = {
    SPAM: TIN_EFFORT,
    TOMATO_SOUP: TIN_EFFORT,
    MUSTARD: PLASTIC_EFFORT,
    SUGAR: CARDBOARD_EFFORT,
    CHEEZIT: CARDBOARD_EFFORT,
}
# TODO: grasps per object

################################################################################

class State(object):
    # TODO: rename to be world state?
    def __init__(self, world, savers=[], attachments=[]):
        # a part of the state separate from PyBullet
        self.world = world
        self.savers = tuple(savers)
        self.attachments = {attachment.child: attachment for attachment in attachments}
    @property
    def bodies(self):
        raise NotImplementedError()
        #return {saver.body for saver in self.savers} | set(self.attachments)
    def derive(self):
        for attachment in self.attachments.values():
            # Derived values
            # TODO: topological sort
            attachment.assign()
    def copy(self):
        return State(self.world, self.savers, self.attachments.values())
        #return copy.deepcopy(self)
    def assign(self):
        for saver in self.savers:
            saver.restore()
        self.derive()
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, list(self.savers), self.attachments)


def create_state(world):
    # TODO: support initially holding
    # TODO: would be better to explicitly keep the state around
    # world.initial_saver.restore()
    world_saver = WorldSaver()
    attachments = []
    for obj_name in world.movable:
        surface_name = world.get_supporting(obj_name)
        if surface_name is not None:
            attachments.append(create_surface_attachment(world, obj_name, surface_name))
    return State(world, savers=[world_saver], attachments=attachments)

################################################################################

class Command(object):
    def __init__(self, world):
        self.world = world
    #@property
    #def robot(self):
    #    return self.world.robot
    @property
    def bodies(self):
        raise NotImplementedError()
    @property
    def cost(self):
        raise NotImplementedError()
    def reverse(self):
        raise NotImplementedError()
    def iterate(self, state):
        raise NotImplementedError()
    def simulate(self, state, time_per_step=DEFAULT_TIME_STEP, **kwargs):
        for j, _ in enumerate(self.iterate(state)):
            state.derive()
            if j != 0:
                wait_for_duration(time_per_step)
    def execute(self, interface):
        raise NotImplementedError()

class Sequence(object):
    def __init__(self, context, commands=[], name=None):
        self.context = context
        self.commands = tuple(commands)
        self.name = self.__class__.__name__.lower() if name is None else name
    @property
    def bodies(self):
        bodies = set(self.context.bodies)
        for command in self.commands:
            bodies.update(command.bodies)
        return bodies
    @property
    def cost(self):
        return sum([0] + [command.cost for command in self.commands])
    def reverse(self):
        return Sequence(self.context, [command.reverse() for command in reversed(self.commands)], name=self.name)
    def __repr__(self):
        #return '[{}]'.format('->'.join(map(repr, self.commands)))
        return '{}({})'.format(self.name, len(self.commands))

################################################################################

class Trajectory(Command):
    def __init__(self, world, robot, joints, path, speed=1.0):
        super(Trajectory, self).__init__(world)
        self.robot = robot
        self.joints = tuple(joints)
        self.path = tuple(path)
        self.speed = speed
    @property
    def bodies(self):
        # TODO: decompose into dependents and moving?
        return flatten_links(self.robot, get_moving_links(self.robot, self.joints))
    @property
    def cost(self):
        return len(self.path)
    def reverse(self):
        return self.__class__(self.world, self.robot, self.joints, self.path[::-1])
    def iterate(self, state):
        #time_parameterization(self.robot, self.joints, self.path)
        for positions in self.path:
            set_joint_positions(self.robot, self.joints, positions)
            yield
    def simulate(self, state, real_per_sim=1, time_step=1./60, **kwargs):

        path = list(self.path)
        path = adjust_path(self.robot, self.joints, path)
        path = waypoints_from_path(path)
        if len(path) <= 1:
            return True
        for joints, path in decompose_into_paths(self.joints, path):
            positions_curve = interpolate_path(self.robot, joints, path)
            print('Following {} {}-DOF waypoints in {:.3f} seconds'.format(len(path), len(joints), positions_curve.x[-1]))
            for t in np.arange(positions_curve.x[0], positions_curve.x[-1], step=time_step):
                positions = positions_curve(t)
                set_joint_positions(self.robot, joints, positions)
                state.derive()
                wait_for_duration(real_per_sim*time_step)
        return True
    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.joints), len(self.path))

################################################################################

class ApproachTrajectory(Trajectory):
    def __init__(self, objects=[], *args, **kwargs):
        super(ApproachTrajectory, self).__init__(*args, **kwargs)
        assert self.joints == self.world.arm_joints
        self.speed = 0.25
        self.objects = set(objects)
    @property
    def bodies(self):
        bodies = set(super(ApproachTrajectory, self).bodies) # TODO: rename to bodies
        for name in self.objects:
            bodies.update(flatten_links(self.world.get_body(name)))
        return bodies
    def reverse(self):
        return self.__class__(self.objects, self.world, self.robot, self.joints, self.path[::-1])


class DoorTrajectory(Command):  # TODO: extend Trajectory
    def __init__(self, world, robot, robot_joints, robot_path,
                 door, door_joints, door_path):
        super(DoorTrajectory, self).__init__(world)
        self.robot = robot
        self.robot_joints = tuple(robot_joints)
        self.robot_path = tuple(robot_path)
        self.door = door
        self.door_joints = tuple(door_joints)
        self.door_path = tuple(door_path)
        self.do_pull = (door_path[0][0] < door_path[-1][0])
        assert len(self.robot_path) == len(self.door_path)
    @property
    def joints(self):
        return self.robot_joints
    @property
    def path(self):
        return self.robot_path
    @property
    def bodies(self):
        return flatten_links(self.robot, get_moving_links(self.robot, self.robot_joints)) | \
               flatten_links(self.world.kitchen, get_moving_links(self.world.kitchen, self.door_joints))
    @property
    def cost(self):
        return len(self.path)
    def reverse(self):
        return self.__class__(self.world, self.robot, self.robot_joints, self.robot_path[::-1],
                              self.door, self.door_joints, self.door_path[::-1])
    def iterate(self, state):
        for robot_conf, door_conf in zip(self.robot_path, self.door_path):
            set_joint_positions(self.robot, self.robot_joints, robot_conf)
            set_joint_positions(self.door, self.door_joints, door_conf)
            yield
    def simulate(self, state, **kwargs):
        # TODO: linearly interpolate for drawer
        # TODO: interpolate drawer and robot individually
        # TODO: find drawer joint angle that minimizes deviation from transform
        super(DoorTrajectory, self).simulate(state, time_per_step=2*DEFAULT_TIME_STEP)
    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.robot_joints) + len(self.door_joints),
                                  len(self.robot_path))

################################################################################s

class Attach(Command):
    def __init__(self, world, robot, link, body):
        # TODO: names or bodies?
        super(Attach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body
    @property
    def bodies(self):
        return set()
        #return {self.robot, self.body}
    @property
    def cost(self):
        return 0
    def reverse(self):
        return Detach(self.world, self.robot, self.link, self.body)
    def attach(self):
        return create_attachment(self.robot, self.link, self.body)
    def iterate(self, state):
        state.attachments[self.body] = self.attach()
        yield
    def execute(self, interface):
        return True
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

class AttachGripper(Attach):
    def __init__(self, world, body, grasp=None):
        super(AttachGripper, self).__init__(world, world.robot, world.tool_link, body)
        self.grasp = grasp

#class AttachSurface(Attach):
#    def __init__(self, world, obj_name, surface_name):
#        body = world.get_body(obj_name)
#        surface = surface_from_name(surface_name)
#        surface_link = link_from_name(world.kitchen, surface.link)
#        super(AttachSurface, self).__init__(world, world.kitchen, surface_link, body)
#        self.obj_name = obj_name
#        self.surface_name = surface_name

class Detach(Command):
    def __init__(self, world, robot, link, body):
        super(Detach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body
    @property
    def bodies(self):
        return set()
        #return {self.robot, self.body}
    @property
    def cost(self):
        return 0
    def reverse(self):
        return Attach(self.world, self.robot, self.link, self.body)
    def iterate(self, state):
        assert self.body in state.attachments
        del state.attachments[self.body]
        yield
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

################################################################################s

class Detect(Command):
    duration = 2.0
    def __init__(self, world, camera, name, pose, rays):
        super(Detect, self).__init__(world)
        self.camera = camera
        self.name = name
        self.pose = pose # Object pose
        self.rays = tuple(rays)
        # TODO: could instead use cones for full detection
    # TODO: bodies?
    @property
    def cost(self):
        return 0
    @property
    def surface_name(self):
        return self.pose.support
    def ray_collision(self):
        return batch_ray_collision(self.rays)
    def compute_occluding(self):
        # TODO: compute as a fraction of the rays
        return {(result.objectUniqueId, frozenset([result.linkIndex]))
                for result in self.ray_collision() if result.objectUniqueId != -1}
    def draw(self):
        handles = []
        for ray, result in zip(self.rays, self.ray_collision()):
            handles.extend(draw_ray(ray, result))
        return handles
    def iterate(self, state):
        handles = self.draw() if DEBUG else []
        steps = int(math.ceil(self.duration / 0.02))
        for _ in range(steps):
            yield
        remove_handles(handles)
    def execute(self, interface):
        return True
    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__, self.camera, self.name, self.surface_name)

class Wait(Command):
    def __init__(self, world, steps=1, duration=1.0):
        super(Wait, self).__init__(world)
        self.steps = steps
        self.duration = duration
    @property
    def bodies(self):
        return set()
    @property
    def cost(self):
        return 0
    def reverse(self):
        return self
    def iterate(self, state):
        for _ in range(self.steps+1):
            yield
    def simulate(self, state, **kwargs):
        wait_for_duration(self.duration)
    def execute(self, interface):
        time.sleep(self.duration)
        #import rospy
        #rospy.sleep(self.duration)
        return True
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.steps)

# TODO: cook that includes a wait

################################################################################s

def iterate_commands(state, commands, time_step=DEFAULT_TIME_STEP, pause=False):
    if commands is None:
        return False
    for i, command in enumerate(commands):
        print('\nCommand {:2}/{:2}: {}'.format(i + 1, len(commands), command))
        # TODO: skip to end
        # TODO: downsample
        for j, _ in enumerate(command.iterate(state)):
            state.derive()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_for_user('Command {:2}/{:2} | step {:2} | Next?'.format(i + 1, len(commands), j))
            elif time_step == 0:
                pass
            else:
                wait_for_duration(time_step)
        if pause:
            wait_for_user('Continue?')
    return True

def simulate_commands(state, commands, **kwargs):
    if commands is None:
        return False
    # TODO: simulate commands simultaneously
    for i, command in enumerate(commands):
        print('\nCommand {:2}/{:2}: {}'.format(i + 1, len(commands), command))
        command.simulate(state, **kwargs)
    return True

def execute_commands(interface, commands):
    if commands is None:
        return False
    for command in commands:
        success = command.execute(interface)
        if success:
            print('Successfully executed command', command)
        else:
            print('Failed to execute command', command)
            return False
    return True
