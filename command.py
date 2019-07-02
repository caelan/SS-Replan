from pybullet_tools.utils import set_joint_positions, create_attachment, get_joint_name, \
    wait_for_duration, user_input, get_distance_fn
from utils import get_descendant_obstacles
from issac import update_robot, ISSAC_REFERENCE_FRAME

import time
import numpy as np

class State(object):
    def __init__(self, savers=[], attachments=[]):
        # a part of the state separate from pybullet
        self.savers = tuple(savers)
        self.attachments = {attachment.child: attachment for attachment in attachments}
    @property
    def bodies(self):
        return {saver.body for saver in self.savers} | set(self.attachments)
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

    def execute(self, domain, world_state, observer):
        raise NotImplementedError()

class Sequence(object):
    def __init__(self, context, commands=[]):
        self.context = context
        self.commands = tuple(commands)
    @property
    def bodies(self):
        bodies = set(self.context.bodies)
        for command in self.commands:
            bodies.update(command.bodies)
        return bodies
    def reverse(self):
        return Sequence(self.context, [command.reverse() for command in reversed(self.commands)])
    def __repr__(self):
        #return '[{}]'.format('->'.join(map(repr, self.commands)))
        return '{}({})'.format(self.__class__.__name__, len(self.commands))

################################################################################

class Trajectory(Command):
    def __init__(self, world, robot, joints, path):
        super(Trajectory, self).__init__(world)
        self.robot = robot
        self.joints = tuple(joints)
        self.path = tuple(path)

    @property
    def bodies(self):
        return {self.robot}

    def reverse(self):
        return self.__class__(self.world, self.robot, self.joints, self.path[::-1])

    def iterate(self, world, state):
        for positions in self.path:
            set_joint_positions(self.robot, self.joints, positions)
            yield

    def execute(self, domain, moveit, observer): # TODO: actor
        robot_entity = domain.get_robot()
        if len(robot_entity.joints) != len(self.joints):
            # TODO: ensure same joint names
            # TODO: allow partial gripper closures
            return

        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/interpolator.py
        # Only position, time_from_start, and velocity are used
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from sensor_msgs.msg import JointState
        import rospy
        joint_names = [get_joint_name(self.robot, joint).encode('ascii') #,'ignore')
                                      for joint in self.joints]

        plan = RobotTrajectory()
        plan.joint_trajectory.header.frame_id = ISSAC_REFERENCE_FRAME
        plan.joint_trajectory.header.stamp = rospy.Time(0)
        plan.joint_trajectory.joint_names = joint_names
        #speed = 0.1
        distance_fn = get_distance_fn(self.robot, self.joints)
        #distances = [0] + [distance_fn(*pair) for pair in zip(self.path[:-1], self.path[1:])]
        #time_from_starts = np.cumsum(distances) / speed
        #print(time_from_starts)

        #robot = domain.get_robot()
        #for i in range(1, len(self.path)):
        for target_conf in self.path:

            #point = JointTrajectoryPoint()
            #point.positions = list(self.path[i])
            # Don't need velocities, accelerations, or efforts
            #vector = np.array(self.path[i]) - np.array(self.path[i-1])
            #duration = (time_from_starts[i] - time_from_starts[i-1])
            #point.velocities = list(vector / duration)
            #point.accelerations = list(np.ones(len(self.joints)))
            #point.effort = list(np.ones(len(self.joints)))
            #point.time_from_start = rospy.Duration(time_from_starts[i])
            #plan.joint_trajectory.points.append(point)

            joint_state = JointState(name=joint_names, position=list(target_conf))
            ee_frame = moveit.forward_kinematics(joint_state.position)
            moveit.visualizer.send(ee_frame)
            moveit.joint_cmd_pub.publish(joint_state)
            rate = rospy.Rate(1000)
            start_time = rospy.Time.now()
            while not rospy.is_shutdown() and ((rospy.Time.now() - start_time).to_sec() < 2):
                world_state = observer.observe()
                robot_entity = world_state.entities[domain.robot]
                error = distance_fn(target_conf, robot_entity.q)
                if error < 0.01:
                    break
                rate.sleep()
            else:
                print('Failed to reach set point')

        #moveit.execute(plan, required_orig_err=0.05, timeout=20.0,
        #               publish_display_trajectory=True)

        #for name, entity in world_state.entities.items():
        #    if entity.controllable_object is not None:
        #        #entity.controllable_object.unsuppress()
        #        entity.controllable_object.suppress() # Propagate to parents?
        #    # entity.set_detached()
        #    #domain.attachments[actor] = goal
        #franka = robot_entity.robot
        #for i, positions in enumerate(self.path):
        #    print('{}/{}'.format(i, len(self.path)))
        #    timeout = 10.0 if i == len(positions)-1 else 2.0
        #    franka.end_effector.go_config(positions, err_thresh=0.05,
        #        wait_for_target=True, wait_time=timeout, verbose=True) # TODO: go_guided/go_long_range
        #    update_robot(self.world, domain, observer.observe())
        #    #wait_for_duration(1e-3)
        #    # TODO: attachments
        time.sleep(1.0)
        # TODO: return status

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
        return {self.robot} | get_descendant_obstacles(self.world.kitchen, self.door_joints[0])

    def reverse(self):
        return self.__class__(self.world, self.robot, self.robot_joints, self.robot_path[::-1],
                              self.door, self.door_joints, self.door_path[::-1])

    def iterate(self, world, state):
        for robot_conf, door_conf in zip(self.robot_path, self.door_path):
            set_joint_positions(self.robot, self.robot_joints, robot_conf)
            set_joint_positions(self.door, self.door_joints, door_conf)
            yield

    def execute(self, domain, moveit, observer):
        raise NotImplementedError()
        robot_entity = domain.get_robot()
        franka = robot_entity.robot
        for positions in self.robot_path:
            franka.end_effector.go_config(positions)
        time.sleep(1.0)
        # TODO: return status

    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.robot_joints) + len(self.door_joints),
                                  len(self.robot_path))

class Attach(Command):
    def __init__(self, world, robot, link, body):
        # TODO: names or bodies?
        super(Attach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body

    @property
    def bodies(self):
        return {self.robot, self.body}

    def reverse(self):
        return Detach(self.world, self.robot, self.link, self.body)

    def iterate(self, world, state):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        yield

    def execute(self, domain, moveit, observer):
        # controllable_object is not needed for joint positions
        moveit.close_gripper(controllable_object=None, speed=0.1, force=40., sleep=0.2, wait=True)
        # TODO: attach_obj
        #robot_entity = domain.get_robot()
        #franka = robot_entity.robot
        #gripper = franka.end_effector.gripper
        #gripper.close(attach_obj=None, speed=.2, force=40., actuate_gripper=True, wait=True)
        #update_robot(self.world, domain, observer, observer.observe())
        #time.sleep(1.0)

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
        return {self.robot, self.body}

    def reverse(self):
        return Attach(self.world, self.robot, self.link, self.body)

    def iterate(self, world, state):
        assert self.body in state.attachments
        del state.attachments[self.body]
        yield

    def execute(self, domain, moveit, observer):
        moveit.open_gripper(speed=0.1, sleep=0.2, wait=True)
        #robot_entity = domain.get_robot()
        #franka = robot_entity.robot
        #gripper = franka.end_effector.gripper
        #gripper.open(speed=.2, actuate_gripper=True, wait=True)
        #update_robot(self.world, domain, observer.observe())
        #time.sleep(1.0)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

class Wait(Command):
    def __init__(self, world, steps):
        super(Wait, self).__init__(world)
        self.steps = steps

    @property
    def bodies(self):
        return {}

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