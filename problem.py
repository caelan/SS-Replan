from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, And, Equal, PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf, Pose
from pybullet_tools.utils import get_joint_name, is_placed_on_aabb
from utils import STOVES, GRASP_TYPES, ALL_SURFACES, CABINET_JOINTS
from stream import get_stable_gen, get_grasp_gen, get_pick_gen, \
    get_motion_gen, base_cost_fn, get_pull_gen, compute_surface_aabb, get_door_test, CLOSED, DOOR_STATUSES, \
    get_cfree_traj_pose_test, get_cfree_traj_angle_test, get_cfree_pose_pose_test, get_cfree_approach_pose_test, \
    get_cfree_approach_angle_test, get_calibrate_gen


def existential_quantification(goal_literals):
    # TODO: merge with pddlstream-experiments
    goal_formula = []
    for literal in goal_literals:
        parameters = [a for a in get_args(literal) if is_parameter(a)]
        if parameters:
            type_literals = [('Type', p, get_parameter_name(p)) for p in parameters]
            goal_formula.append(Exists(parameters, And(literal, *type_literals)))
        else:
            goal_formula.append(literal)
    return And(*goal_formula)

################################################################################

def pdddlstream_from_problem(world, noisy_base=False, **kwargs):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {
        '@stove': 'stove',
    }
    # TODO: operate on initi

    initial_bq = Conf(world.robot, world.base_joints)
    initial_aq = Conf(world.robot, world.arm_joints)
    init = [
        ('BConf', initial_bq),
        ('AtBConf', initial_bq),
        ('AConf', initial_aq),
        ('AtAConf', initial_aq),
        ('HandEmpty',),
        ('CanMove',),

        Equal(('CalibrateCost',), 1),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('PullCost',), 1),
        Equal(('CookCost',), 1),
    ] + [('Type', name, 'stove') for name in STOVES] + \
           [('Status', status) for status in DOOR_STATUSES]
    if noisy_base:
        init.append(('NoisyBase',))

    goal_block = list(world.movable)[0]
    goal_surface = CABINET_JOINTS[0]
    goal_on = {
        goal_block: goal_surface,
    }

    goal_literals = [
        #('Holding', goal_block),
        #('Cooked', goal_block),
        ('AtBConf', initial_bq),
    ]

    for name in world.movable:
        body = world.get_body(name)
        [surface] = [surface for surface in ALL_SURFACES
                     if is_placed_on_aabb(body, compute_surface_aabb(world, surface))]
        pose = Pose(body, support=surface, init=True)
        init += [
            ('Graspable', name),
            ('Pose', name, pose),
            ('Supported', name, pose, surface),
            ('Stackable', name, None),
            ('AtPose', name, pose),
        ] # + [('Stackable', name, surface) for surface in STOVES + [None]]
    #for body, ty in problem.body_types:
    #    init += [('Type', body, ty)]

    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        initial_conf = Conf(world.kitchen, [joint])
        open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])
        closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
        init.append(('AtAngle', joint_name, initial_conf))
        goal_literals.append(('DoorStatus', joint_name, CLOSED))
        for conf in [initial_conf, open_conf, closed_conf]:
            init.append(('Angle', joint_name, conf))

    #if problem.goal_conf is not None:
    #    goal_conf = Conf(robot, get_group_joints(robot, 'base'), problem.goal_conf)
    #    init += [('BConf', goal_conf)]
    #    goal_literals += [('AtBConf', goal_conf)]

    #bodies_from_type = get_bodies_from_type(problem)
    for ty, s in goal_on.items():
        #bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]
        #init += [('Stackable', b, s) for b in bodies]
        init += [('Stackable', ty, s)]
        goal_literals += [('On', ty, s)]
    #goal_literals += [('Holding', a, b) for a, b in problem.goal_holding] + \
    #                 [('Cleaned', b) for b in problem.goal_cleaned] + \
    #                 [('Cooked', b) for b in problem.goal_cooked]

    goal_formula = existential_quantification(goal_literals)

    stream_map = {
        'test-door': from_test(get_door_test(world)),
        'sample-pose': from_gen_fn(get_stable_gen(world, **kwargs)),
        'sample-grasp': from_gen_fn(get_grasp_gen(world, grasp_types=GRASP_TYPES)),
        'inverse-kinematics': from_gen_fn(get_pick_gen(world, **kwargs)),
        'plan-pull': from_gen_fn(get_pull_gen(world, **kwargs)),
        'plan-base-motion': from_fn(get_motion_gen(world, **kwargs)),
        'plan-calibrate-motion': from_fn(get_calibrate_gen(world, **kwargs)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(**kwargs)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(world, **kwargs)),
        'test-cfree-approach-angle': from_test(get_cfree_approach_angle_test(world, **kwargs)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(world, **kwargs)),
        'test-cfree-traj-angle': from_test(get_cfree_traj_angle_test(world, **kwargs)),

        # 'MoveCost': move_cost_fn,
        'Distance': base_cost_fn,
    }
    #stream_map = DEBUG

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
