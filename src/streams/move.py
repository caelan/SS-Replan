import numpy as np

from pybullet_tools.utils import BodySaver, plan_nonholonomic_motion, set_renderer, wait_for_user, plan_joint_motion, \
    get_extend_fn, child_link_from_joint
from src.command import Sequence, State, Trajectory
from src.inference import SurfaceDist
from src.stream import ARM_RESOLUTION, SELF_COLLISIONS, GRIPPER_RESOLUTION
from src.utils import get_link_obstacles, FConf, get_descendant_obstacles

PAUSE_MOTION_FAILURES = False

def parse_fluents(world, fluents):
    obstacles = set()
    for fluent in fluents:
        predicate, args = fluent[0], fluent[1:]
        if predicate in {p.lower() for p in ['AtBConf', 'AtAConf', 'AtGConf']}:
            q, = args
            q.assign()
        elif predicate == 'AtAngle'.lower():
            j, a = args
            a.assign()
            link = child_link_from_joint(a.joints[0])
            obstacles.update(get_descendant_obstacles(a.body, link))
        elif predicate in 'AtWorldPose'.lower():
            # TODO: conditional effects are not being correctly updated in pddlstream
            #b, p = args
            #if isinstance(p, SurfaceDist):
            #    continue
            #p.assign()
            #obstacles.update(get_link_obstacles(world, b))
            raise RuntimeError()
        elif predicate in 'AtRelPose'.lower():
            pass
        elif predicate == 'AtGrasp'.lower():
            pass
        else:
            raise NotImplementedError(predicate)

    attachments = []
    for fluent in fluents:
        predicate, args = fluent[0], fluent[1:]
        if predicate in {p.lower() for p in ['AtBConf', 'AtAConf', 'AtGConf']}:
            pass
        elif predicate == 'AtAngle'.lower():
            pass
        elif predicate in 'AtWorldPose'.lower():
            raise RuntimeError()
        elif predicate in 'AtRelPose'.lower():
            o1, rp, o2 = args
            if isinstance(rp, SurfaceDist):
                continue
            rp.assign()
            obstacles.update(get_link_obstacles(world, o1))
        elif predicate == 'AtGrasp'.lower():
            o, g = args
            if o is not None:
                attachments.append(g.get_attachment())
                attachments[-1].assign()
        else:
            raise NotImplementedError(predicate)
    return attachments, obstacles

################################################################################

# TODO: more efficient collision checking

def get_base_motion_fn(world, teleport_base=False, collisions=True, teleport=False,
                       restarts=4, iterations=75, smooth=100):
    # TODO: lazy planning on a common base roadmap

    def fn(bq1, bq2, aq, fluents=[]):
        #if bq1 == bq2:
        #    return None
        aq.assign()
        attachments, obstacles = parse_fluents(world, fluents)
        obstacles.update(world.static_obstacles)
        if not collisions:
            obstacles = set()

        start_path, end_path = [], []
        if hasattr(bq1, 'nearby_bq'):
            bq1.assign()
            start_path = plan_nonholonomic_motion(world.robot, bq2.joints, bq1.nearby_bq.values, attachments=attachments,
                                            obstacles=obstacles, custom_limits=world.custom_limits,
                                            reversible=True, self_collisions=False, restarts=-1)
            if start_path is None:
                print('Failed to find nearby base conf!')
                return
            bq1 = bq1.nearby_bq
        if hasattr(bq2, 'nearby_bq'):
            bq2.nearby_bq.assign()
            end_path = plan_nonholonomic_motion(world.robot, bq2.joints, bq2.values, attachments=attachments,
                                            obstacles=obstacles, custom_limits=world.custom_limits,
                                            reversible=True, self_collisions=False, restarts=-1)
            if end_path is None:
                print('Failed to find nearby base conf!')
                return
            bq2 = bq2.nearby_bq

        bq1.assign()
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
                print('Failed to find an arm motion plan for {}->{}'.format(bq1, bq2))
                if PAUSE_MOTION_FAILURES:
                    set_renderer(enable=True)
                    print(fluents)
                    for bq in [bq1, bq2]:
                        bq.assign()
                        wait_for_user()
                    set_renderer(enable=False)
                return None

        # TODO: could actually plan with all joints as long as we return to the same config
        cmd = Sequence(State(world, savers=[robot_saver]), commands=[
            Trajectory(world, world.robot, world.base_joints, path)
            for path in [start_path, path, end_path] if path], name='base')
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
        obstacles.update(world.static_obstacles)
        if not collisions:
            obstacles = set()
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
                print('Failed to find an arm motion plan for {}->{}'.format(aq1, aq2))
                if PAUSE_MOTION_FAILURES:
                    set_renderer(enable=True)
                    print(fluents)
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
