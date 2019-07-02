from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, \
    And, Equal, PDDLProblem
from pddlstream.language.stream import DEBUG
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import get_joint_name, is_placed_on_aabb, create_attachment, \
    child_link_from_joint, get_link_name, parent_joint_from_link, is_fixed
from utils import STOVES, GRASP_TYPES, ALL_SURFACES, CABINETS, DRAWERS, \
    get_surface, COUNTERS
from stream import get_stable_gen, get_grasp_gen, get_pick_gen, \
    get_motion_gen, base_cost_fn, get_pull_gen, compute_surface_aabb, get_door_test, CLOSED, DOOR_STATUSES, \
    get_cfree_traj_pose_test, get_cfree_pose_pose_test, get_cfree_approach_pose_test, \
    get_calibrate_gen, get_pick_ik_fn, \
    get_fixed_pull_gen, get_compute_angle_kin, get_compute_pose_kin, \
    link_from_name, get_link_pose, RelPose, wait_for_user


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

def pdddlstream_from_problem(world, close_doors=False, return_home=False,
                             movable_base=True, fixed_base=False,
                             noisy_base=False, debug=False, **kwargs):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {
        '@world': 'world',
        '@gripper': 'gripper',
        '@stove': 'stove',
    }

    init_bq = Conf(world.robot, world.base_joints)
    init_aq = Conf(world.robot, world.arm_joints)
    init = [
        ('BConf', init_bq),
        ('AtBConf', init_bq),
        ('AConf', init_aq),
        ('AtAConf', init_aq),
        ('HandEmpty',),
        ('CanMove',), # TODO: could always remove this

        Equal(('CalibrateCost',), 1),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('PullCost',), 1),
        Equal(('CookCost',), 1),
    ] + [('Type', obj_name, 'stove') for obj_name in STOVES] + \
           [('Status', status) for status in DOOR_STATUSES]
    if movable_base:
        init.append(('MovableBase',))
    if fixed_base:
        init.append(('InitBConf', init_bq))
    if noisy_base:
        init.append(('NoisyBase',))

    if fixed_base:
        world.carry_conf = init_aq


    compute_pose_kin = get_compute_pose_kin(world)
    compute_angle_kin = get_compute_angle_kin(world)

    goal_block = list(world.movable)[0]
    #goal_surface = CABINETS[0]
    goal_surface = DRAWERS[1]
    #goal_surface = COUNTERS[0]
    goal_on = {
        goal_block: goal_surface,
    }

    goal_literals = [
        #('Holding', goal_block),
        #('Cooked', goal_block),
    ]
    if return_home:
        goal_literals.append(('AtBConf', init_bq))

    surface_poses = {}
    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        link = child_link_from_joint(joint)
        link_name = get_link_name(world.kitchen, link) # Relies on the fact that drawers have identical surface and link names
        init_conf = Conf(world.kitchen, [joint], init=True)
        open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])
        closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
        for conf in [init_conf, open_conf, closed_conf]:
            pose, = compute_angle_kin(link_name, joint_name, conf)
            init.extend([
                ('Angle', joint_name, conf),
                ('Movable', link_name),
                ('AngleKin', link_name, pose, joint_name, conf),
                ('WorldPose', link_name, pose),
            ])
            if conf == init_conf:
                surface_poses[link_name] = pose
                init.extend([
                    ('AtAngle', joint_name, conf),
                    ('AtWorldPose', link_name, pose),
                ])
        if close_doors:
            goal_literals.append(('DoorStatus', joint_name, CLOSED))

    for surface_name in ALL_SURFACES:
        surface = get_surface(surface_name)
        surface_link = link_from_name(world.kitchen, surface.link)
        parent_joint = parent_joint_from_link(surface_link)
        if parent_joint in world.kitchen_joints:
            assert surface_name in surface_poses
            #joint_name = get_joint_name(world.kitchen, parent_joint)
            #init.append(('Connected', surface_name, joint_name))
        else:
            pose = RelPose(world.kitchen, surface_link, init=True)
            surface_poses[surface_name] = pose
            init += [
                #('RelPose', surface_name, pose, 'world'),
                ('WorldPose', surface_name, pose),
                #('AtRelPose', surface_name, pose, 'world'),
                ('AtWorldPose', surface_name, pose),
            ]

    for obj_name in world.movable:
        # TODO: raise above surface and simulate to exploit physics
        body = world.get_body(obj_name)
        supporting = [surface for surface in ALL_SURFACES if is_placed_on_aabb(
            body, compute_surface_aabb(world, surface),
            above_epsilon=1e-2, below_epsilon=5e-2)]
        if len(supporting) != 1:
            print('{} is not supported by a single surface ({})!'.format(obj_name, supporting))
            continue
            #raise RuntimeError(obj_name, supporting)
        [surface_name] = supporting
        surface = get_surface(surface_name)
        surface_link = link_from_name(world.kitchen, surface.link)
        attachment = create_attachment(world.kitchen, surface_link, body)
        world.initial_attachments[body] = attachment # TODO: init state instead
        rel_pose = RelPose(body, reference_body=world.kitchen, reference_link=surface_link,
                       confs=[attachment], support=surface_name, init=True)
        surface_pose = surface_poses[surface_name]
        world_pose, = compute_pose_kin(obj_name, rel_pose, surface_name, surface_pose)

        init += [
            ('Movable', obj_name),
            ('Graspable', obj_name),
            ('RelPose', obj_name, rel_pose, surface_name),
            ('AtRelPose', obj_name, rel_pose, surface_name),
            ('WorldPose', obj_name, world_pose),
            ('PoseKin', obj_name, world_pose, rel_pose, surface_name, surface_pose),
            ('AtWorldPose', obj_name, world_pose),
        ] + [('Stackable', obj_name, counter) for counter in COUNTERS]
    #for body, ty in problem.body_types:
    #    init += [('Type', body, ty)]

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
        'plan-pick': from_gen_fn(get_pick_gen(world, **kwargs)),
        'plan-pull': from_gen_fn(get_pull_gen(world, **kwargs)),
        'plan-base-motion': from_fn(get_motion_gen(world, **kwargs)),
        'plan-calibrate-motion': from_fn(get_calibrate_gen(world, **kwargs)),

        'fixed-plan-pick': from_gen_fn(get_pick_ik_fn(world, **kwargs)),
        'fixed-plan-pull': from_gen_fn(get_fixed_pull_gen(world, **kwargs)),

        'compute-pose-kin': from_fn(compute_pose_kin),
        'compute-angle-kin': from_fn(compute_angle_kin),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(world, **kwargs)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(world, **kwargs)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(world, **kwargs)),

        # 'MoveCost': move_cost_fn,
        'Distance': base_cost_fn,
    }
    if debug:
        stream_map = DEBUG

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
