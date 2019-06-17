from pybullet_tools.utils import set_joint_positions, create_attachment

class State(object):
    def __init__(self, savers=[], attachments={}):
        # a part of the state separate from pybullet
        self.savers = tuple(savers)
        self.attachments = dict(attachments)
    def derive(self):
        for attachment in self.attachments.values():
            # Derived values
            # TODO: topological sort
            attachment.assign()
    def assign(self):
        for saver in self.savers:
            saver.restore()
        self.derive()
    # TODO: copy?

class Command(object):
    def __init__(self, world):
        self.world = world

    def reverse(self):
        raise NotImplementedError()

    def iterate(self, world, state):
        raise NotImplementedError()

class Sequence(object):
    def __init__(self, context, commands=[]):
        self.context = context
        self.commands = tuple(commands)
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

    def reverse(self):
        return Trajectory(self.world, self.robot, self.joints, self.path[::-1])

    def iterate(self, world, state):
        for positions in self.path:
            set_joint_positions(world.robot, self.joints, positions)
            yield positions

    def __repr__(self):
        return '{}({}x{})'.format(self.__class__.__name__, len(self.joints), len(self.path))


class Attach(Command):
    def __init__(self, world, robot, link, body):
        # TODO: names or bodies?
        super(Attach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body

    def reverse(self):
        return Detach(self.world, self.robot, self.link, self.body)

    def iterate(self, world, state):
        state.attachments[self.robot, self.link, self.body] = \
            create_attachment(self.robot, self.link, self.body)
        yield

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

class Detach(Command):
    def __init__(self, world, robot, link, body):
        super(Detach, self).__init__(world)
        self.robot = robot
        self.link = link
        self.body = body

    def reverse(self):
        return Attach(self.world, self.robot, self.link, self.body)

    def iterate(self, world, state):
        assert (self.robot, self.link, self.body) in state.attachments
        del state.attachments[self.robot, self.link, self.body]
        yield

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.world.get_name(self.body))

class Wait(Command):
    def __init__(self, world, steps):
        super(Wait, self).__init__(world)
        self.steps = steps

    def reverse(self):
        return self

    def iterate(self, world, state):
        for _ in range(self.steps):
            yield

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.steps)

# TODO: cook that includes a wait
