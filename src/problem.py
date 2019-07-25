from __future__ import print_function

from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, \
    And, Equal, PDDLProblem
from pddlstream.language.stream import DEBUG
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import get_joint_name, child_link_from_joint, get_link_name, parent_joint_from_link, link_from_name, \
    WorldSaver

from src.utils import STOVES, GRASP_TYPES, ALL_SURFACES, surface_from_name, COUNTERS, RelPose, \
    create_surface_attachment, create_relative_pose
from src.stream import get_stable_gen, get_grasp_gen, get_pick_gen_fn, \
    get_base_motion_fn, base_cost_fn, get_pull_gen_fn, get_door_test, CLOSED, DOOR_STATUSES, \
    get_cfree_traj_pose_test, get_cfree_pose_pose_test, get_cfree_approach_pose_test, OPEN, \
    get_calibrate_gen, get_fixed_pick_gen_fn, get_fixed_pull_gen_fn, get_compute_angle_kin, \
    get_compute_pose_kin, get_arm_motion_gen, get_gripper_motion_gen, get_test_near_pose, \
    get_test_near_joint, get_gripper_open_test, BASE_CONSTANT, get_nearby_stable_gen, \
    get_compute_detect, get_ofree_ray_pose_test, get_ofree_ray_grasp_test, get_sample_belief_gen
from src.issac import load_calibrate_conf
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
    'detect': 1,
    'pick': 1,
    'place': 1,
    'pull': 1,
    'cook': 1,
}

def title_from_snake(s):
    return ''.join(x.title() for x in s.split('_'))

def get_streams(world, debug=False, **kwargs):
    stream_pddl = read(get_file_path(__file__, '../pddl/stream.pddl'))

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
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(world, **kwargs)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(world, **kwargs)),

        'test-ofree-ray-pose': from_test(get_ofree_ray_pose_test(world, **kwargs)),
        'test-ofree-ray-grasp': from_test(get_ofree_ray_grasp_test(world, **kwargs)),
        # 'MoveCost': move_cost_fn,
        # 'Distance': base_cost_fn,
    }
    if debug:
        stream_map = DEBUG
    return stream_pddl, stream_map

def pdddlstream_from_problem(belief, **kwargs):
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
    # TODO: return set of facts that support the previous plan
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

        ('HandEmpty',),
        ('Grasp', None, None),
        ('AtGrasp', None, None),

        ('Calibrated',),
        ('CanMoveBase',),
        ('CanMoveArm',),
        ('CanMoveGripper',),
    ]
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
                     [('DoorStatus', joint_name, CLOSED) for joint_name in task.goal_closed] + \
                     [('Cooked', name) for name in task.goal_cooked] + \
                     [('Detected', name) for name in task.goal_detected]
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
            ('AConf', goal_bq, calibrate_aq),
        ])
        goal_literals.append(('AtBConf', goal_bq))
        if task.return_init_aq:
            init.append(('AConf', goal_bq, goal_aq))
            goal_literals.append(('AtAConf', goal_aq))

    surface_poses = {}
    for joint in world.kitchen_joints:
        joint_name = get_joint_name(world.kitchen, joint)
        link = child_link_from_joint(joint)
        # Relies on the fact that drawers have identical surface and link names
        link_name = get_link_name(world.kitchen, link)
        init_conf = Conf(world.kitchen, [joint], init=True)
        open_conf = Conf(world.kitchen, [joint], [world.open_conf(joint)])
        closed_conf = Conf(world.kitchen, [joint], [world.closed_conf(joint)])
        for conf in [init_conf, open_conf, closed_conf]:
            # TODO: return to initial poses?
            pose, = compute_angle_kin(link_name, joint_name, conf)
            init.extend([
                ('Joint', joint_name),
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

    for surface_name in ALL_SURFACES:
        surface = surface_from_name(surface_name)
        surface_link = link_from_name(world.kitchen, surface.link)
        parent_joint = parent_joint_from_link(surface_link)
        if parent_joint in world.kitchen_joints:
            assert surface_name in surface_poses
            #joint_name = get_joint_name(world.kitchen, parent_joint)
            init.append(('CheckNearby', surface_name))
        else:
            pose = RelPose(world.kitchen, surface_link, init=True)
            surface_poses[surface_name] = pose
            init += [
                ('CheckNearby', surface_name),
                #('RelPose', surface_name, pose, 'world'),
                ('WorldPose', surface_name, pose),
                #('AtRelPose', surface_name, pose, 'world'),
                ('AtWorldPose', surface_name, pose),
                ('Counter', surface_name, pose), # Fixed surface
            ]
        for grasp_type in GRASP_TYPES:
            if has_place_database(world.robot_name, surface_name, grasp_type):
                init.append(('AdmitsGraspType', surface_name, grasp_type))

    if belief.grasped is not None:
        obj_name = belief.grasped.body_name
        assert obj_name not in belief.pose_dists
        grasp = belief.grasped
        init += [
            ('Movable', obj_name),
            ('Graspable', obj_name),
            ('CheckNearby', obj_name),
            ('Grasp', obj_name, grasp),
            ('IsGraspType', obj_name, grasp, grasp.grasp_type),
            ('AtGrasp', obj_name, grasp),
            ('Holding', obj_name),
        ]
    for obj_name, pose_dist in belief.pose_dists.items():
        body = world.get_body(obj_name)
        support = pose_dist.dist.support()
        assert len(support) == 1
        [rel_pose] = support
        surface_name = rel_pose.support
        if surface_name is None:
            # Treats as obstacle
            world_pose = RelPose(body, init=True)
            init += [
                ('Movable', obj_name), # TODO: misnomer
                ('WorldPose', obj_name, world_pose),
                ('AtWorldPose', obj_name, world_pose),
            ]
            #raise RuntimeError(obj_name, supporting)
        else:
            surface_pose = surface_poses[surface_name]
            world_pose, = compute_pose_kin(obj_name, rel_pose, surface_name, surface_pose)
            init += [
                ('Movable', obj_name),
                ('Graspable', obj_name),
                ('CheckNearby', obj_name),
                ('RelPose', obj_name, rel_pose, surface_name),
                ('AtRelPose', obj_name, rel_pose, surface_name),
                ('WorldPose', obj_name, world_pose),
                ('PoseKin', obj_name, world_pose, rel_pose, surface_name, surface_pose),
                ('AtWorldPose', obj_name, world_pose),
                ('On', obj_name, surface_name),
            ] + [('Stackable', obj_name, counter) for counter in COUNTERS]

    #for body, ty in problem.body_types:
    #    init += [('Type', body, ty)]
    #bodies_from_type = get_bodies_from_type(problem)
    #bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]

    goal_formula = existential_quantification(goal_literals)
    stream_pddl, stream_map = get_streams(world, **kwargs)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)
