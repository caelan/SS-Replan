from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, And, Equal, PDDLProblem
from pddlstream.language.generator import from_gen_fn, from_fn
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf, Pose
from pybullet_tools.utils import get_joint_name
from utils import STOVES, GRASP_TYPES
from stream import get_stable_gen, get_grasp_gen, get_pick_gen, get_motion_gen, distance_fn, get_pull_gen


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

def pdddlstream_from_problem(world, **kwargs):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {
        '@stove': 'stove',
    }

    initial_bq = Conf(world.robot, world.base_joints)
    initial_aq = Conf(world.robot, world.arm_joints)
    init = [
        ('BConf', initial_bq),
        ('AtBConf', initial_bq),
        ('AConf', initial_aq),
        ('AtAConf', initial_aq),
        ('HandEmpty',),
        ('CanMove',),

        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('PullCost',), 1),
        Equal(('CookCost',), 1),
    ] + [('Type', name, 'stove') for name in STOVES]

    for name in world.movable:
        pose = Pose(world.get_body(name), init=True)  # TODO: supported here
        init += [
            ('Graspable', name),
            ('Pose', name, pose),
            ('AtPose', name, pose),
        ] + [('Stackable', name, surface) for surface in STOVES + [None]]
        #for surface in problem.surfaces:
        #    if is_placement(body, surface):
        #        init += [('Supported', body, pose, surface)]
    #for body, ty in problem.body_types:
    #    init += [('Type', body, ty)]

    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        initial_conf = Conf(world.kitchen, [joint])
        open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])
        closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
        init.extend([
            ('Conf', joint_name, initial_conf),
            ('AtConf', joint_name, initial_conf),
            ('Conf', joint_name, open_conf),
            ('OpenConf', joint_name, open_conf),
            ('Conf', joint_name, closed_conf),
            ('ClosedConf', joint_name, closed_conf),
        ])

    block = list(world.movable)[0]
    joint = 'chewie_door_left_joint' # baker_joint | chewie_door_left_joint | hitman_drawer_top_joint
    goal_literals = [
        #('Open', joint),
        ('Holding', block),
        #('Cooked', block),
        ('AtBConf', initial_bq),
    ]
    #if problem.goal_conf is not None:
    #    goal_conf = Conf(robot, get_group_joints(robot, 'base'), problem.goal_conf)
    #    init += [('BConf', goal_conf)]
    #    goal_literals += [('AtBConf', goal_conf)]

    #bodies_from_type = get_bodies_from_type(problem)
    #for ty, s in problem.goal_on:
    #    bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]
    #    init += [('Stackable', b, s) for b in bodies]
    #    goal_literals += [('On', ty, s)]
    #goal_literals += [('Holding', a, b) for a, b in problem.goal_holding] + \
    #                 [('Cleaned', b) for b in problem.goal_cleaned] + \
    #                 [('Cooked', b) for b in problem.goal_cooked]

    goal_formula = existential_quantification(goal_literals)

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(world, **kwargs)),
        'sample-grasp': from_gen_fn(get_grasp_gen(world, grasp_types=GRASP_TYPES)),
        'inverse-kinematics': from_gen_fn(get_pick_gen(world, **kwargs)),
        'plan-pull': from_gen_fn(get_pull_gen(world, **kwargs)),
        'plan-base-motion': from_fn(get_motion_gen(world, **kwargs)),

        #'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
        #'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
        #'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem, collisions=collisions)),
        # 'test-cfree-traj-grasp-pose': from_test(get_cfree_traj_grasp_pose_test(problem, collisions=collisions)),

        # 'MoveCost': move_cost_fn,
        'Distance': distance_fn,
    }
    #stream_map = DEBUG

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
