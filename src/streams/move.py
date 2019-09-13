import numpy as np

from pybullet_tools.utils import BodySaver, plan_nonholonomic_motion, set_renderer, wait_for_user, plan_joint_motion, \
    get_extend_fn
from src.command import Sequence, State, Trajectory
from src.inference import SurfaceDist
from src.stream import PAUSE_MOTION_FAILURES, ARM_RESOLUTION, SELF_COLLISIONS, GRIPPER_RESOLUTION
from src.utils import get_link_obstacles, FConf


def parse_fluents(world, fluents):
    attachments = []
    obstacles = set()
    for fluent in fluents:
        predicate, args = fluent[0], fluent[1:]
        if predicate in {p.lower() for p in ['AtBConf', 'AtAConf', 'AtGConf']}:
            q, = args
            q.assign()
        elif predicate == 'AtAngle'.lower():
            raise RuntimeError()
            # j, a = args
            # a.assign()
            # obstacles.update(get_descendant_obstacles(a.body, a.joints[0]))
        elif predicate in {p.lower() for p in ['AtPose', 'AtWorldPose']}:
            b, p = args
            if isinstance(p, SurfaceDist):
                continue
            p.assign()
            obstacles.update(get_link_obstacles(world, b))
        elif predicate == 'AtGrasp'.lower():
            b, g = args
            if b is not None:
                attachments.append(g.get_attachment())
                attachments[-1].assign()
        else:
            raise NotImplementedError(predicate)
    return attachments, obstacles

################################################################################

def get_base_motion_fn(world, teleport_base=False, collisions=True, teleport=False,
                       restarts=4, iterations=75, smooth=100):
    # TODO: lazy planning on a common base roadmap

    def fn(bq1, bq2, aq, fluents=[]):
        #if bq1 == bq2:
        #    return None
        bq1.assign()
        aq.assign()
        attachments, obstacles = parse_fluents(world, fluents)
        if not collisions:
            obstacles = set()
        obstacles.update(world.static_obstacles)
        robot_saver = BodySaver(world.robot)
        if (bq1 == bq2) or teleport_base or teleport:
            path = [bq1.values, bq2.values]
        else:
            # It's important that the extend function is reversible to avoid getting trapped
            path = plan_nonholonomic_motion(world.robot, bq2.joints, bq2.values, attachments=attachments,
                                            obstacles=obstacles, custom_limits=world.custom_limits,
                                            reversible=True, self_collisions=False,
                                            restarts=restarts, iterations=iterations, smooth=smooth)
            if path is None:
                print('Failed to find a base motion plan!')
                if PAUSE_MOTION_FAILURES:
                    set_renderer(enable=True)
                    #print(fluents)
                    for bq in [bq1, bq2]:
                        bq.assign()
                        wait_for_user()
                    set_renderer(enable=False)
                return None
        # TODO: could actually plan with all joints as long as we return to the same config
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            Trajectory(world, world.robot, world.base_joints, path),
        ], name='base')
        return (cmd,)
    return fn


def get_reachability_test(world, **kwargs):
    base_motion_fn = get_base_motion_fn(world, restarts=2, iterations=50, smooth=0, **kwargs)
    bq0 = FConf(world.robot, world.base_joints)
    # TODO: can check for arm motions as well

    def test(bq):
        aq = world.carry_conf
        outputs = base_motion_fn(aq, bq0, bq, fluents=[])
        return outputs is not None
    return test


def get_arm_motion_gen(world, collisions=True, teleport=False):
    resolutions = ARM_RESOLUTION * np.ones(len(world.arm_joints))

    def fn(bq, aq1, aq2, fluents=[]):
        #if aq1 == aq2:
        #    return None
        bq.assign()
        aq1.assign()
        attachments, obstacles = parse_fluents(world, fluents)
        if not collisions:
            obstacles = set()
        obstacles.update(world.static_obstacles)
        robot_saver = BodySaver(world.robot)
        if teleport:
            path = [aq1.values, aq2.values]
        else:
            path = plan_joint_motion(world.robot, aq2.joints, aq2.values,
                                     attachments=attachments, obstacles=obstacles,
                                     self_collisions=SELF_COLLISIONS,
                                     disabled_collisions=world.disabled_collisions,
                                     custom_limits=world.custom_limits, resolutions=resolutions,
                                     restarts=2, iterations=50, smooth=50)
            if path is None:
                print('Failed to find an arm motion plan!')
                if PAUSE_MOTION_FAILURES:
                    set_renderer(enable=True)
                    #print(fluents)
                    for bq in [aq1, aq2]:
                        bq.assign()
                        wait_for_user()
                    set_renderer(enable=False)
                return None
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            Trajectory(world, world.robot, world.arm_joints, path),
        ], name='arm')
        return (cmd,)
    return fn


def get_gripper_motion_gen(world, teleport=False, **kwargs):
    resolutions = GRIPPER_RESOLUTION * np.ones(len(world.gripper_joints))

    def fn(gq1, gq2):
        #if gq1 == gq2:
        #    return None
        if teleport:
            path = [gq1.values, gq2.values]
        else:
            extend_fn = get_extend_fn(gq2.body, gq2.joints, resolutions=resolutions)
            path = [gq1.values] + list(extend_fn(gq1.values, gq2.values))
        cmd = Sequence(State(world), commands=[
            Trajectory(world, gq2.body, gq2.joints, path),
        ], name='gripper')
        return (cmd,)
    return fn