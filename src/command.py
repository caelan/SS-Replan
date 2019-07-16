from src.execution import joint_state_control, open_gripper, close_gripper, moveit_control, follow_base_trajectory
from pybullet_tools.utils import get_moving_links, set_joint_positions, create_attachment, \
    wait_for_duration, user_input, wait_for_user, flatten_links, \
    get_max_limit, get_joint_limits, waypoints_from_path
from src.issac import update_robot, update_isaac_robot

from isaac_bridge.manager import SimulationManager

import numpy as np
import time

MOVEIT = True
DEFAULT_SLEEP = 1.0
FORCE = 100

class State(object):
    def __init__(self, savers=[], attachments=[]):
        # a part of the state separate from pybullet
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
    def assign(self):
        for saver in self.savers:
            saver.restore()
        self.derive()
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, list(self.savers), self.attachments)
    # TODO: copy?

################################################################################

class Command(object):
    def __init__(self, world):
        self.world = world

    @property
    def bodies(self):
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    def iterate(self, world, state):
        raise NotImplementedError()

    def execute(self, domain, moveit, observer):
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

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/isaac_bridge/launch/carter_localization_priors.launch#L6

CARTER_X = 33.1
CARTER_Y = 7.789

class Trajectory(Command):
    def __init__(self, world, robot, joints, path):
        super(Trajectory, self).__init__(world)
        self.robot = robot
        self.joints = tuple(joints)
        self.path = tuple(path)

    @property
    def bodies(self):
        # TODO: decompose into dependents and moving?
        return flatten_links(self.robot, get_moving_links(self.robot, self.joints))

    def reverse(self):
        return self.__class__(self.world, self.robot, self.joints, self.path[::-1])

    def iterate(self, world, state):
        for positions in self.path:
            set_joint_positions(self.robot, self.joints, positions)
            yield

    def execute(self, domain, moveit, observer): # TODO: actor
        # TODO: ensure the same joint names
        if self.joints == self.world.base_joints:
            #assert not moveit.use_lula
            robot_entity = domain.get_robot()
            #robot_entity = world_state.entities[domain.robot]
            carter = robot_entity.carter_interface
            if carter is None:
                follow_base_trajectory(self.world, self.path, moveit, observer)
            elif isinstance(carter, SimulationManager):
                set_joint_positions(self.robot, self.joints, self.path[-1])
                update_isaac_robot(observer, carter, self.world)
                time.sleep(DEFAULT_SLEEP)
            else:
                world_state = domain.root
                # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/scripts/move_carter.py
                # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py
                # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_predicates.py#L164
                # TODO: ensure that I transform into the correct base units
                # The base uses its own motion planner
                world_state[domain.robot].suppress_fixed_bases()
                carter.move_to_async(self.path[-1])
                #for conf in waypoints_from_path(self.path):
                #    carter.move_to_safe(conf)
                #    #carter.move_to_async(conf)
                #    #carter.simple_move(cmd)
                #carter.simple_stop()
                world_state[domain.robot].unsuppress_fixed_bases()
                #carter.pub_disable_deadman_switch.publish(True)
            return

        if MOVEIT:
            if self.joints == self.world.gripper_joints:
                joint = self.joints[0]
                average = np.average(get_joint_limits(self.robot, joint))
                position = self.path[-1][0]
                if position < average:
                    moveit.close_gripper(force=FORCE)
                else:
                    moveit.open_gripper()
            else:
                moveit_control(self.robot, self.joints, self.path, moveit, observer)
            time.sleep(DEFAULT_SLEEP)
        else:
            status = joint_state_control(self.robot, self.joints, self.path, domain, moveit, observer)
        time.sleep(DEFAULT_SLEEP)
        #return status

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
    def bodies(self):
        return flatten_links(self.robot, get_moving_links(self.robot, self.robot_joints)) | \
               flatten_links(self.world.kitchen, get_moving_links(self.world.kitchen, self.door_joints))

    def reverse(self):
        return self.__class__(self.world, self.robot, self.robot_joints, self.robot_path[::-1],
                              self.door, self.door_joints, self.door_path[::-1])

    def iterate(self, world, state):
        for robot_conf, door_conf in zip(self.robot_path, self.door_path):
            set_joint_positions(self.robot, self.robot_joints, robot_conf)
            set_joint_positions(self.door, self.door_joints, door_conf)
            yield

    def execute(self, domain, moveit, observer):
        #update_robot(self.world, domain, observer, observer.observe())
        #wait_for_user()
        if MOVEIT:
            moveit.close_gripper() #force=FORCE)
            time.sleep(DEFAULT_SLEEP)
            moveit_control(self.robot, self.robot_joints, self.robot_path, moveit, observer)
            time.sleep(DEFAULT_SLEEP)
            moveit.open_gripper()
            time.sleep(DEFAULT_SLEEP)
        else:
            close_gripper(self.robot, moveit)
            status = joint_state_control(self.robot, self.robot_joints, self.robot_path,
                                         domain, moveit, observer)
            open_gripper(self.robot, moveit)
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

    def iterate(self, world, state):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        yield

    def execute(self, domain, moveit, observer):
        if self.world.robot != self.robot:
            return
        if MOVEIT:
            moveit.close_gripper(wait=True, sleep=0., speed=0.03)
        else:
            return close_gripper(self.robot, moveit)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

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

    def iterate(self, world, state):
        assert self.body in state.attachments
        del state.attachments[self.body]
        yield

    def execute(self, domain, moveit, observer):
        if self.world.robot != self.robot:
            return
        if MOVEIT:
            moveit.open_gripper()
        else:
            return open_gripper(self.robot, moveit)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

class Wait(Command):
    def __init__(self, world, steps):
        super(Wait, self).__init__(world)
        self.steps = steps

    @property
    def bodies(self):
        return set()

    def reverse(self):
        return self

    def iterate(self, world, state):
        for _ in range(self.steps):
            yield

    def execute(self, domain, moveit, observer):
        pass

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.steps)

# TODO: cook that includes a wait

################################################################################s

def execute_plan(world, state, commands, time_step=None):
    for i, command in enumerate(commands):
        print('\nCommand {:2}: {}'.format(i, command))
        # TODO: skip to end
        # TODO: downsample
        for j, _ in enumerate(command.iterate(world, state)):
            state.derive()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                user_input('Command {:2} | step {:2} | Next?'.format(i, j))
            else:
                wait_for_duration(time_step)