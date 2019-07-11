from pddlstream.language.constants import get_args, is_parameter, get_parameter_name, Exists, \
    And, Equal, PDDLProblem
from pddlstream.language.stream import DEBUG
from pddlstream.language.generator import from_gen_fn, from_fn, from_test
from pddlstream.utils import read, get_file_path

from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.utils import get_joint_name, is_placed_on_aabb, create_attachment, \
    child_link_from_joint, get_link_name, parent_joint_from_link, is_fixed, link_from_name, \
    get_link_pose, wait_for_user, get_min_limits, get_max_limits, spaced_colors, draw_point, \
    Point, LockRenderer, add_segments, convex_hull, add_text, convex_centroid, get_pose, \
    multiply, point_from_pose, RED, get_point

import numpy as np

from src.utils import STOVES, GRASP_TYPES, ALL_SURFACES, CABINETS, DRAWERS, \
    get_surface, COUNTERS, RelPose, ALL_JOINTS, get_grasps, invert
from src.stream import get_stable_gen, get_grasp_gen, get_pick_gen_fn, \
    get_base_motion_fn, base_cost_fn, get_pull_gen_fn, compute_surface_aabb, get_door_test, CLOSED, DOOR_STATUSES, \
    get_cfree_traj_pose_test, get_cfree_pose_pose_test, get_cfree_approach_pose_test, OPEN, \
    get_calibrate_gen, get_fixed_pick_gen_fn, \
    get_fixed_pull_gen_fn, get_compute_angle_kin, get_compute_pose_kin, get_arm_motion_gen, get_gripper_motion_gen
from src.database import has_place_database, load_pull_base_poses, \
    get_surface_reference_pose, load_placements, load_place_base_poses

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

def visualize_base_confs(world, name, base_confs, floor_z=0.005, **kwargs):
    print(name, len(base_confs))
    handles = []
    if not base_confs:
        return handles
    z = get_point(world.floor)[2] + floor_z
    base_points = [base_conf[:2] for base_conf in base_confs]
    # for x, y in base_points:
    #    handles.extend(draw_point(Point(x, y, z), color=color))
    hull = convex_hull(base_points)
    vertices = [Point(x, y, z) for x, y, in hull.vertices]
    handles.extend(add_segments(vertices, closed=True, **kwargs))
    cx, cy = convex_centroid(hull.vertices)
    centroid = [cx, cy, z]
    # draw_point(centroid, color=color)
    handles.append(add_text(name, position=centroid, **kwargs))
    return handles

# https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_online.py
# https://github.mit.edu/caelan/stripstream/blob/master/robotics/openrave/belief_tamp.py
# https://github.mit.edu/caelan/ss/blob/master/belief/belief_online.py

def add_markers(world):
    handles = []
    for joint_name, color in zip(ALL_JOINTS, spaced_colors(len(ALL_JOINTS))):
        base_confs = list(load_pull_base_poses(world, joint_name))
        handles.extend(visualize_base_confs(world, joint_name, base_confs, color=color))

    for surface_name in ALL_SURFACES:
        surface_pose = get_surface_reference_pose(world.kitchen, surface_name)
        for grasp_type, color in zip(GRASP_TYPES, spaced_colors(len(GRASP_TYPES))):
            object_points = []
            for surface_from_object in load_placements(world, surface_name, grasp_types=[grasp_type]):
                object_pose = multiply(surface_pose, surface_from_object)
                object_points.append(point_from_pose(object_pose))
            if not object_points:
                continue
            #for object_point in object_points:
            #    handles.extend(draw_point(object_point, color=color))
            _, _, z = np.average(object_points, axis=0)
            hull = convex_hull([object_point[:2] for object_point in object_points])
            vertices = [Point(x, y, z) for x, y, in hull.vertices]
            handles.extend(add_segments(vertices, color=color, closed=True))

    for name in world.movable:
        body = world.get_body(name)
        pose = get_pose(body)
        surface_name = get_supporting(world, name)
        if surface_name is None:
            continue
        for grasp_type, color in zip(GRASP_TYPES, spaced_colors(len(GRASP_TYPES))):
            base_confs = []
            for grasp in get_grasps(world, name, grasp_types=[grasp_type]):
                tool_pose = multiply(pose, invert(grasp.grasp_pose))
                base_confs.extend(load_place_base_poses(world, tool_pose, surface_name, grasp_type))
            handles.extend(visualize_base_confs(world, grasp_type, base_confs, color=color))
    return handles

def get_supporting(world, obj_name):
    body = world.get_body(obj_name)
    supporting = [surface for surface in ALL_SURFACES if is_placed_on_aabb(
        body, compute_surface_aabb(world, surface),
        above_epsilon=1e-2, below_epsilon=5e-2)]
    if len(supporting) != 1:
        print('{} is not supported by a single surface ({})!'.format(obj_name, supporting))
        return None
    [surface_name] = supporting
    return surface_name

def pdddlstream_from_problem(task, debug=False, **kwargs):
    print(task)
    world = task.world
    domain_pddl = read(get_file_path(__file__, '../pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, '../pddl/stream.pddl'))

    with LockRenderer():
        add_markers(world)
    #wait_for_user()

    init_bq = Conf(world.robot, world.base_joints)
    init_aq = Conf(world.robot, world.arm_joints)
    init_gq = Conf(world.robot, world.gripper_joints)

    constant_map = {
        '@world': 'world',
        '@gripper': 'gripper',
        '@stove': 'stove',

        '@rest_aq': world.carry_conf,
        '@open_gq': world.open_gq,
        '@closed_gq': world.closed_gq,
        '@open': OPEN,
        '@closed': CLOSED,
    }

    init = [
        ('BConf', init_bq),
        ('AtBConf', init_bq),
        ('AConf', init_aq),
        ('AtAConf', init_aq),
        ('GConf', init_gq),
        ('AtGConf', init_gq),
        ('HandEmpty',),

        ('AConf', world.carry_conf),
        ('GConf', world.open_gq),
        ('GConf', world.closed_gq),

        ('CanMoveBase',),  # TODO: could always remove this
        ('CanMoveArm',),
        ('CanMoveGripper',),

        Equal(('CalibrateCost',), 1),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('PullCost',), 1),
        Equal(('CookCost',), 1),
    ]
    init += [('Type', obj_name, 'stove') for obj_name in STOVES] + \
            [('Stackable', name, surface) for name, surface in task.goal_on.items()] + \
            [('Status', status) for status in DOOR_STATUSES] + \
            [('GraspType', ty) for ty in GRASP_TYPES] # TODO: grasp_type per object
    if task.movable_base:
        init.append(('MovableBase',))
    if task.fixed_base:
        init.append(('InitBConf', init_bq))
    if task.noisy_base:
        init.append(('NoisyBase',))

    compute_pose_kin = get_compute_pose_kin(world)
    compute_angle_kin = get_compute_angle_kin(world)

    goal_literals = []
    goal_literals += [('Holding', name) for name in task.goal_holding] + \
                     [('On', name, surface) for name, surface in task.goal_on.items()] + \
                     [('DoorStatus', joint_name, CLOSED) for joint_name in task.goal_closed] + \
                     [('Cooked', name) for name in task.goal_cooked]
    if task.goal_hand_empty:
        goal_literals.append(('HandEmpty',))
    if task.goal_init_conf:
        goal_bq = init_bq
        init.append(('BConf', goal_bq))
        goal_literals.append(('AtBConf', goal_bq))

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
                ('Counter', surface_name, pose),
            ]
        for grasp_type in GRASP_TYPES:
            if has_place_database(world.robot_name, surface_name, grasp_type):
                init.append(('AdmitsGraspType', surface_name, grasp_type))

    for obj_name in world.movable:
        # TODO: raise above surface and simulate to exploit physics
        body = world.get_body(obj_name)
        surface_name =  get_supporting(world, obj_name)
        if surface_name is None:
            world_pose = RelPose(body, init=True)
            # Treats as obstacle
            init += [
                ('Movable', obj_name), # TODO: misnomer
                ('WorldPose', obj_name, world_pose),
                ('AtWorldPose', obj_name, world_pose),
            ]
            continue
            #raise RuntimeError(obj_name, supporting)
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
    #bodies_from_type = get_bodies_from_type(problem)
    #bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]

    goal_formula = existential_quantification(goal_literals)

    stream_map = {
        'test-door': from_test(get_door_test(world)),
        'sample-pose': from_gen_fn(get_stable_gen(world, **kwargs)),
        'sample-grasp': from_gen_fn(get_grasp_gen(world)),
        'plan-pick': from_gen_fn(get_pick_gen_fn(world, **kwargs)),
        'plan-pull': from_gen_fn(get_pull_gen_fn(world, **kwargs)),

        'plan-base-motion': from_fn(get_base_motion_fn(world, **kwargs)),
        'plan-arm-motion': from_fn(get_arm_motion_gen(world, **kwargs)),
        'plan-gripper-motion': from_fn(get_gripper_motion_gen(world, **kwargs)),
        'plan-calibrate-motion': from_fn(get_calibrate_gen(world, **kwargs)),

        'fixed-plan-pick': from_gen_fn(get_fixed_pick_gen_fn(world, **kwargs)),
        'fixed-plan-pull': from_gen_fn(get_fixed_pull_gen_fn(world, **kwargs)),

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
