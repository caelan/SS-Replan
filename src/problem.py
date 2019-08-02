from __future__ import print_function

import numpy as np
import math

from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, \
    And, Equal, PDDLProblem
from pddlstream.language.stream import DEBUG
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import get_joint_name, child_link_from_joint, get_link_name, parent_joint_from_link, link_from_name, \
    WorldSaver, get_difference_fn

from src.belief import PoseDist
from src.utils import STOVES, GRASP_TYPES, ALL_SURFACES, surface_from_name, COUNTERS, RelPose
from src.stream import get_stable_gen, get_grasp_gen, get_pick_gen_fn, \
    get_base_motion_fn, base_cost_fn, get_pull_gen_fn, get_door_test, CLOSED, DOOR_STATUSES, \
    get_cfree_traj_pose_test, get_cfree_pose_pose_test, get_cfree_approach_pose_test, OPEN, \
    get_calibrate_gen, get_fixed_pick_gen_fn, get_fixed_pull_gen_fn, get_compute_angle_kin, \
    get_compute_pose_kin, get_arm_motion_gen, get_gripper_motion_gen, get_test_near_pose, \
    get_test_near_joint, get_gripper_open_test, BASE_CONSTANT, get_nearby_stable_gen, \
    get_compute_detect, get_ofree_ray_pose_test, get_ofree_ray_grasp_test, \
    get_sample_belief_gen, detect_cost_fn, get_cfree_bconf_pose_test
from src.database import has_place_database


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

ACTION_COSTS = {
    'move_base': BASE_CONSTANT,
    'move_arm': 1,
    'move_gripper': 1,
    'calibrate': 1,
    #'detect': 1,
    'pick': 1,
    'place': 1,
    'pull': 1,
    'cook': 1,
}

def title_from_snake(s):
    return ''.join(x.title() for x in s.split('_'))

def get_streams(world, debug=False, **kwargs):
    stream_pddl = read(get_file_path(__file__, '../pddl/stream.pddl'))
    if debug:
        return stream_pddl, DEBUG
    stream_map = {
        'test-door': from_test(get_door_test(world)),
        'test-gripper': from_test(get_gripper_open_test(world)),

        'sample-pose': from_gen_fn(get_stable_gen(world, **kwargs)),
        'sample-grasp': from_gen_fn(get_grasp_gen(world)),
        'sample-nearby-pose': from_gen_fn(get_nearby_stable_gen(world, **kwargs)),

        'plan-pick': from_gen_fn(get_pick_gen_fn(world, **kwargs)),
        'plan-pull': from_gen_fn(get_pull_gen_fn(world, **kwargs)),

        'plan-base-motion': from_fn(get_base_motion_fn(world, **kwargs)),
        'plan-arm-motion': from_fn(get_arm_motion_gen(world, **kwargs)),
        'plan-gripper-motion': from_fn(get_gripper_motion_gen(world, **kwargs)),
        'plan-calibrate-motion': from_fn(get_calibrate_gen(world, **kwargs)),

        'test-near-pose': from_test(get_test_near_pose(world, **kwargs)),
        'test-near-joint': from_test(get_test_near_joint(world, **kwargs)),

        'fixed-plan-pick': from_gen_fn(get_fixed_pick_gen_fn(world, **kwargs)),
        'fixed-plan-pull': from_gen_fn(get_fixed_pull_gen_fn(world, **kwargs)),

        'compute-pose-kin': from_fn(get_compute_pose_kin(world)),
        # 'compute-angle-kin': from_fn(compute_angle_kin),
        'compute-detect': from_fn(get_compute_detect(world, **kwargs)),
        'sample-belief': from_gen_fn(get_sample_belief_gen(world, **kwargs)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(world, **kwargs)),
        'test-cfree-bconf-pose': from_test(get_cfree_bconf_pose_test(world, **kwargs)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(world, **kwargs)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(world, **kwargs)),

        'test-ofree-ray-pose': from_test(get_ofree_ray_pose_test(world, **kwargs)),
        'test-ofree-ray-grasp': from_test(get_ofree_ray_grasp_test(world, **kwargs)),

        'DetectCost': detect_cost_fn,
        #'MoveCost': move_cost_fn,
        # 'Distance': base_cost_fn,
    }
    return stream_pddl, stream_map

def pdddlstream_from_problem(belief, additional_init=[], **kwargs):
    world = belief.world # One world per state
    task = world.task # One task per world
    print(task)
    domain_pddl = read(get_file_path(__file__, '../pddl/domain.pddl'))

    init_bq = Conf(world.robot, world.base_joints)
    init_aq = Conf(world.robot, world.arm_joints)
    init_gq = Conf(world.robot, world.gripper_joints)

    carry_aq = world.carry_conf
    calibrate_aq = world.calibrate_conf
    # TODO: order goals for serialization
    # TODO: return set of facts that dist_support the previous plan
    # TODO: repackage stream outputs to avoid recomputation

    constant_map = {
        '@world': 'world',
        '@gripper': 'gripper',
        '@stove': 'stove',

        '@rest_aq': carry_aq,
        '@calibrate_aq': calibrate_aq,
        '@open_gq': world.open_gq,
        '@closed_gq': world.closed_gq,
        '@open': OPEN,
        '@closed': CLOSED,
    }

    init = [
        ('BConf', init_bq),
        ('AtBConf', init_bq),
        ('AConf', init_bq, carry_aq),
        ('RestAConf', carry_aq),
        ('AConf', init_bq, calibrate_aq),

        ('AConf', init_bq, init_aq),
        ('AtAConf', init_aq),

        ('GConf', init_gq),
        ('AtGConf', init_gq),
        ('GConf', world.open_gq),
        ('GConf', world.closed_gq),

        ('Grasp', None, None),
        ('AtGrasp', None, None),

        ('Calibrated',),
        ('CanMoveBase',),
        ('CanMoveArm',),
        ('CanMoveGripper',),
    ] + list(additional_init)
    for action_name, cost in ACTION_COSTS.items():
        function_name = '{}Cost'.format(title_from_snake(action_name))
        function = (function_name,)
        init.append(Equal(function, cost))
    init += [('Type', obj_name, 'stove') for obj_name in STOVES] + \
            [('Stackable', name, surface) for name, surface in task.goal_on.items()] + \
            [('Status', status) for status in DOOR_STATUSES] + \
            [('GraspType', ty) for ty in GRASP_TYPES] # TODO: grasp_type per object
            #[('Camera', name) for name in world.cameras]
    if task.movable_base:
        init.append(('MovableBase',))
    #if task.fixed_base:
    init.append(('InitBConf', init_bq))
    if task.noisy_base:
        init.append(('NoisyBase',))

    compute_pose_kin = get_compute_pose_kin(world)
    compute_angle_kin = get_compute_angle_kin(world)

    goal_literals = []
    goal_literals += [('Holding', name) for name in task.goal_holding] + \
                     [('On', name, surface) for name, surface in task.goal_on.items()] + \
                     [('Cooked', name) for name in task.goal_cooked] + \
                     [('Localized', name) for name in task.goal_detected] + \
                     [Exists(['?a'], And(('AngleWithin', joint_name, '?a', CLOSED),
                                         ('AtAngle', joint_name, '?a')))
                      for joint_name in task.goal_closed] + \
                     [Exists(['?a'], And(('AngleWithin', joint_name, '?a', OPEN),
                                         ('AtAngle', joint_name, '?a')))
                      for joint_name in task.goal_open]

    if task.goal_hand_empty:
        goal_literals.append(('HandEmpty',))
    if task.return_init_bq:
        with WorldSaver():
            world.initial_saver.restore()
            goal_bq = Conf(world.robot, world.base_joints)
            goal_aq = Conf(world.robot, world.arm_joints)
        if not task.movable_base:
            goal_bq = init_bq
        init.extend([
            ('BConf', goal_bq),
            ('AConf', goal_bq, carry_aq),
            ('CloseTo', goal_bq, goal_bq),
        ])
        base_difference_fn = get_difference_fn(world.robot, world.base_joints)
        if np.less_equal(np.abs(base_difference_fn(init_bq.values, goal_bq.values)),
                         [0.05, 0.05, math.radians(10)]).all():
            print('Close to goal base configuration')
            init.append(('CloseTo', init_bq, goal_bq))
        goal_literals.append(Exists(['?bq'], And(
            ('CloseTo', '?bq', goal_bq), ('AtBConf', '?bq'))))
        if task.return_init_aq:
            arm_distance_fn = get_difference_fn(world.robot, world.arm_joints)
            if np.less_equal(np.abs(arm_distance_fn(init_aq.values, goal_aq.values)),
                             math.radians(10)*np.ones(len(world.arm_joints))).all():
                print('Close to goal arm configuration')
                init.append(('CloseTo', init_aq, goal_aq))
            init.extend([
                ('AConf', goal_bq, goal_aq),
                ('CloseTo', goal_aq, goal_aq),
            ])
            goal_literals.append(Exists(['?aq'], And(
                ('CloseTo', '?aq', goal_aq), ('AtAConf', '?aq'))))

    surface_poses = {}
    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        #joint_name = str(joint_name.decode('UTF-8'))
        link = child_link_from_joint(joint)
        # Relies on the fact that drawers have identical surface and link names
        link_name = get_link_name(world.kitchen, link)
        #link_name = str(link_name.decode('UTF-8'))
        init_conf = Conf(world.kitchen, [joint], init=True)
        open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])
        #init_conf = open_conf
        closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
        for conf in [init_conf, open_conf, closed_conf]:
            # TODO: return to initial poses?
            world_pose, = compute_angle_kin(link_name, joint_name, conf)
            init.extend([
                ('Joint', joint_name),
                ('Angle', joint_name, conf),
                ('Obstacle', link_name),
                ('AngleKin', link_name, world_pose, joint_name, conf),
                ('WorldPose', link_name, world_pose),
            ])
            if conf == init_conf:
                surface_poses[link_name] = world_pose
                init.extend([
                    ('AtAngle', joint_name, conf),
                    ('AtWorldPose', link_name, world_pose),
                ])

    for surface_name in ALL_SURFACES:
        surface = surface_from_name(surface_name)
        surface_link = link_from_name(world.kitchen, surface.link)
        parent_joint = parent_joint_from_link(surface_link)
        if parent_joint in world.kitchen_joints:
            assert surface_name in surface_poses
            #joint_name = get_joint_name(world.kitchen, parent_joint)
            world_pose = surface_poses[surface_name]
        else:
            world_pose = RelPose(world.kitchen, surface_link, init=True)
            surface_poses[surface_name] = world_pose
            init += [
                ('Counter', surface_name, world_pose),  # Fixed surface
                #('RelPose', surface_name, world_pose, 'world'),
                ('WorldPose', surface_name, world_pose),
                #('AtRelPose', surface_name, world_pose, 'world'),
                ('AtWorldPose', surface_name, world_pose),
            ]
        init.extend([
            ('CheckNearby', surface_name),
            #('InitPose', world_pose),
            ('Localized', surface_name),
            ('Sample', world_pose),
            ('Value', world_pose),
        ])
        for grasp_type in GRASP_TYPES:
            if has_place_database(world.robot_name, surface_name, grasp_type):
                init.append(('AdmitsGraspType', surface_name, grasp_type))

    if belief.grasped is None:
        init.append(('HandEmpty',))
    else:
        obj_name = belief.grasped.body_name
        assert obj_name not in belief.pose_dists
        grasp = belief.grasped
        init += [
            # Static
            ('Grasp', obj_name, grasp),
            ('IsGraspType', obj_name, grasp, grasp.grasp_type),
            # Fluent
            ('AtGrasp', obj_name, grasp),
            ('Holding', obj_name),
            ('Localized', obj_name),
        ]

    for obj_name in world.movable:
        init += [
            ('Obstacle', obj_name),
            ('Graspable', obj_name),
            ('CheckNearby', obj_name),
        ] + [('Stackable', obj_name, counter) for counter in set(ALL_SURFACES) & set(COUNTERS)]

    # TODO: track poses over time to produce estimates
    for obj_name, pose_dist in belief.pose_dists.items():
        body = world.get_body(obj_name)
        dist_support = pose_dist.dist.support()
        localized = (len(dist_support) == 1)
        if localized:
            init.append(('Localized', obj_name))
        # Could also fully decompose into points (but many samples)
        # Could immediately add likely points for collision checking
        for rel_pose in (dist_support if localized else pose_dist.decompose()):
            surface_name = rel_pose.support
            if surface_name is None:
                # Treats as obstacle
                world_pose = RelPose(body, init=True)
                init += [
                    ('WorldPose', obj_name, world_pose),
                    ('AtWorldPose', obj_name, world_pose),
                ]
                poses = [world_pose]
                #raise RuntimeError(obj_name, supporting)
            else:
                surface_pose = surface_poses[surface_name]
                world_pose, = compute_pose_kin(obj_name, rel_pose, surface_name, surface_pose)
                init += [
                    # Static
                    ('RelPose', obj_name, rel_pose, surface_name),
                    ('WorldPose', obj_name, world_pose),
                    ('PoseKin', obj_name, world_pose, rel_pose, surface_name, surface_pose),
                    # Fluent
                    ('AtRelPose', obj_name, rel_pose, surface_name),
                    ('AtWorldPose', obj_name, world_pose),
                ]
                if localized:
                    init.append(('On', obj_name, surface_name))
                poses = [rel_pose, world_pose]
            for pose in poses:
                if isinstance(pose, PoseDist):
                    init.append(('Dist', pose))
                else:
                    init.extend([('Sample', pose), ('Value', pose)])

    #for body, ty in problem.body_types:
    #    init += [('Type', body, ty)]
    #bodies_from_type = get_bodies_from_type(problem)
    #bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]

    goal_formula = existential_quantification(goal_literals)
    stream_pddl, stream_map = get_streams(world, **kwargs)

    print('Init:', sorted(init, key=lambda f: f[0]))
    print('Goal:', goal_formula)
    #print('Streams:', stream_map.keys()) # DEBUG

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
