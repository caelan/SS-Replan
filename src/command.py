import math
import numpy as np
import time

from isaac_bridge.manager import SimulationManager

from pybullet_tools.utils import get_moving_links, set_joint_positions, create_attachment, \
    wait_for_duration, user_input, flatten_links, remove_handles, \
    get_joint_limits, batch_ray_collision, draw_ray
from src.base import follow_base_trajectory
from src.execution import moveit_control, \
    franka_open_gripper, franka_close_gripper, franka_control
from src.issac import update_robot, update_isaac_robot, update_observer

DEFAULT_SLEEP = 0.5
FORCE = 50 # 20 | 50 | 100
# TODO: force per object

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/isaac_bridge/launch/carter_localization_priors.launch#L6
# /isaac/odometry is zero
# /isaac/pose_status is 33.1
CARTER_X = 33.1
CARTER_Y = 7.789

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

    def reverse(self):
        raise NotImplementedError()

    def iterate(self, state):
        raise NotImplementedError()

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

    def reverse(self):
        return self.__class__(self.world, self.robot, self.joints, self.path[::-1])

    def iterate(self, state):
        #time_parameterization(self.robot, self.joints, self.path)
        for positions in self.path:
            set_joint_positions(self.robot, self.joints, positions)
            yield

    def execute_base(self, interface):
        assert self.joints == self.world.base_joints
        # assert not moveit.use_lula
        domain = interface.domain
        carter = domain.carter
        if carter is None:
            follow_base_trajectory(self.world, self.path, interface.moveit, interface.observer)
        elif isinstance(carter, SimulationManager):
            sim_manager = carter
            interface.pause_simulation()
            set_joint_positions(self.robot, self.joints, self.path[-1])
            update_isaac_robot(interface.observer, sim_manager, self.world)
            time.sleep(DEFAULT_SLEEP)
            interface.resume_simulation()
            # TODO: teleport attached
        else:
            world_state = domain.root
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter_sim.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/2cb8df9ac14b56a5955251cf4325369172c2ba72/packages/isaac_bridge/src/isaac_bridge/carter.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/scripts/move_carter.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_predicates.py#L164
            # TODO: ensure that I transform into the correct base units
            # The base uses its own motion planner
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/fixed_base_suppressor.py
            world_state[domain.robot].suppress_fixed_bases()
            carter.move_to_async(self.path[-1])
            # for conf in waypoints_from_path(self.path):
            #    carter.move_to_safe(conf)
            #    #carter.move_to_async(conf)
            #    #carter.simple_move(cmd)
            # carter.simple_stop()
            world_state[domain.robot].unsuppress_fixed_bases()
            # carter.pub_disable_deadman_switch.publish(True)

    def execute_gripper(self, interface):
        assert self.joints == self.world.gripper_joints
        position = self.path[-1][0]
        # move_gripper_action(position)
        joint = self.joints[0]
        average = np.average(get_joint_limits(self.robot, joint))
        if position < average:
            franka_close_gripper(interface)
        else:
            franka_open_gripper(interface)

    def execute(self, interface):
        # TODO: ensure the same joint name order
        observer = interface.observer
        if self.joints == self.world.base_joints:
            self.execute_base(interface)
        elif self.joints == self.world.gripper_joints:
            self.execute_gripper(interface)
        else:
            franka_control(self.robot, self.joints, self.path, interface)
        #status = joint_state_control(self.robot, self.joints, self.path, domain, moveit, observer)
        time.sleep(DEFAULT_SLEEP)
        #return status

        if self.joints == self.world.arm_joints:
            #world_state = observer.current_state
            world_state = interface.update_state()
            robot_entity = world_state[interface.domain.robot]
            print('Error:', (np.array(robot_entity.q) - np.array(self.path[-1])).round(5))
            update_robot(self.world, interface.domain, observer)
            #wait_for_user('Continue?')
    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.joints), len(self.path))

class DoorTrajectory(Command):
    def __init__(self, world, robot, robot_joints, robot_path,
                 door, door_joints, door_path):
        super(DoorTrajectory, self).__init__(world)
        self.robot = robot
        self.robot_joints = tuple(robot_joints)
        self.robot_path = tuple(robot_path)
        self.door = door
        self.door_joints = tuple(door_joints)
        self.door_path = tuple(door_path)
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

    def reverse(self):
        return self.__class__(self.world, self.robot, self.robot_joints, self.robot_path[::-1],
                              self.door, self.door_joints, self.door_path[::-1])

    def iterate(self, state):
        for robot_conf, door_conf in zip(self.robot_path, self.door_path):
            set_joint_positions(self.robot, self.robot_joints, robot_conf)
            set_joint_positions(self.door, self.door_joints, door_conf)
            yield

    def execute(self, interface):
        #update_robot(self.world, domain, observer, observer.observe())
        #wait_for_user()
        franka_close_gripper(interface)
        time.sleep(DEFAULT_SLEEP)

        franka_control(self.robot, self.joints, self.path, interface)
        time.sleep(DEFAULT_SLEEP)

        franka_open_gripper(interface)
        time.sleep(DEFAULT_SLEEP)

        #close_gripper(self.robot, moveit)
        #status = joint_state_control(self.robot, self.robot_joints, self.robot_path,
        #                             domain, moveit, observer)
        #open_gripper(self.robot, moveit)
        #return status

    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.robot_joints) + len(self.door_joints),
                                  len(self.robot_path))

################################################################################s

class Attach(Command):
    def __init__(self, world, robot, link, body, grasp=None):
        # TODO: names or bodies?
        super(Attach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body
        self.grasp = grasp

    @property
    def bodies(self):
        return set()
        #return {self.robot, self.body}

    def reverse(self):
        return Detach(self.world, self.robot, self.link, self.body)

    def attach(self):
        return create_attachment(self.robot, self.link, self.body)

    def iterate(self, state):
        state.attachments[self.body] = self.attach()
        yield

    def execute(self, interface):
        if self.world.robot != self.robot:
            return
        franka_close_gripper(interface)
        #return close_gripper(self.robot, moveit)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

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

    def reverse(self):
        return Attach(self.world, self.robot, self.link, self.body)

    def iterate(self, state):
        assert self.body in state.attachments
        del state.attachments[self.body]
        yield

    def execute(self, interface):
        if self.world.robot != self.robot:
            return
        franka_open_gripper(interface)
        #return open_gripper(self.robot, moveit)

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

    @property
    def surface_name(self):
        return self.pose.support

    def ray_collision(self):
        return batch_ray_collision(self.rays)

    def compute_occluding(self):
        return {(result.objectUniqueId, frozenset([result.linkIndex]))
                for result in self.ray_collision() if result.objectUniqueId != -1}

    def draw(self):
        handles = []
        for ray, result in zip(self.rays, self.ray_collision()):
            handles.extend(draw_ray(ray, result))
        return handles

    def iterate(self, state):
        handles = self.draw()
        steps = int(math.ceil(self.duration / 0.02))
        for _ in range(steps):
            yield
        remove_handles(handles)

    def execute(self, interface):
        pass

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__, self.camera, self.name, self.surface_name)

class Wait(Command):
    def __init__(self, world, steps):
        super(Wait, self).__init__(world)
        self.steps = steps

    @property
    def bodies(self):
        return set()

    def reverse(self):
        return self

    def iterate(self, state):
        for _ in range(self.steps):
            yield

    def execute(self, interface):
        pass

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.steps)

# TODO: cook that includes a wait

################################################################################s

def iterate_commands(state, commands, time_step=None):
    if not commands:
        return
    for i, command in enumerate(commands):
        print('\nCommand {:2}: {}'.format(i, command))
        # TODO: skip to end
        # TODO: downsample
        for j, _ in enumerate(command.iterate(state)):
            state.derive()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                user_input('Command {:2} | step {:2} | Next?'.format(i, j))
            else:
                wait_for_duration(time_step)

def execute_commands(intereface, commands):
    if not commands:
        return
    for command in commands:
        command.execute(intereface)
