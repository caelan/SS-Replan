import numpy as np

from pybullet_tools.utils import get_point, convex_hull, Point, add_segments, convex_centroid, add_text, spaced_colors, \
    multiply, point_from_pose, get_pose, invert, link_from_name, grow_polygon, GREEN, get_link_pose
from src.database import load_pull_base_poses, get_surface_reference_pose, load_placements, \
    load_place_base_poses, load_forward_placements, load_inverse_placements
from src.utils import ALL_JOINTS, ALL_SURFACES, get_grasps, surface_from_name, STOVES

GROW_INVERSE_BASE = 0.05 # 0.05 | 0.1
GROW_FORWARD_RADIUS = 0.25 # Done for block. Incorrect for other object types

def get_floor_z(world, floor_z=0.005):
    return get_point(world.floor)[2] + floor_z

def visualize_base_confs(world, name, base_confs, **kwargs):
    handles = []
    if not base_confs:
        return handles
    z = get_floor_z(world)
    # for x, y in base_points:
    #    handles.extend(draw_point(Point(x, y, z), color=color))
    vertices = grow_polygon(base_confs, radius=GROW_INVERSE_BASE)
    points = [Point(x, y, z) for x, y, in vertices]
    handles.extend(add_segments(points, closed=True, **kwargs))
    cx, cy = convex_centroid(vertices)
    centroid = [cx, cy, z]
    # draw_point(centroid, color=color)
    handles.append(add_text(name, position=centroid, **kwargs))
    return handles


def add_markers(task, placements=True, forward_place=True, pull_bases=True, inverse_place=False):
    # TODO: decompose
    world = task.world
    handles = []
    if placements:
        for surface_name in ALL_SURFACES:
            surface = surface_from_name(surface_name)
            surface_link = link_from_name(world.kitchen, surface.link)
            surface_point = point_from_pose(get_link_pose(world.kitchen, surface_link))
            for grasp_type, color in zip(task.grasp_types, spaced_colors(len(task.grasp_types))):
                object_points = list(map(point_from_pose, load_placements(world, surface_name,
                                                                          grasp_types=[grasp_type])))
                if (surface_name not in STOVES) and object_points:
                    #for object_point in object_points:
                    #    handles.extend(draw_point(object_point, color=color))
                    _, _, z = np.average(object_points, axis=0)
                    object_points = [Point(x, y, z) for x, y in grow_polygon(object_points, radius=0.0)]
                    handles.extend(add_segments(object_points, color=color, closed=True,
                                                parent=world.kitchen, parent_link=surface_link))
                base_points = list(map(point_from_pose, load_inverse_placements(world, surface_name,
                                                                                grasp_types=[grasp_type])))
                if (surface_name in STOVES) and base_points: # and inverse_place
                    #continue
                    #_, _, z = np.average(base_points, axis=0)
                    z = get_floor_z(world) - surface_point[2]
                    base_points = [Point(x, y, z) for x, y in grow_polygon(base_points, radius=GROW_INVERSE_BASE)]
                    handles.extend(add_segments(base_points, color=color, closed=True,
                                                parent=world.kitchen, parent_link=surface_link))

    if forward_place:
        # TODO: do this by taking the union of all grasps
        object_points = list(map(point_from_pose, load_forward_placements(world)))
        robot_point = point_from_pose(get_link_pose(world.robot, world.base_link))
        #z = 0.
        z = get_floor_z(world) - robot_point[2]
        object_points = [Point(x, y, z) for x, y in grow_polygon(object_points, radius=GROW_FORWARD_RADIUS)]
        handles.extend(add_segments(object_points, color=GREEN, closed=True,
                                    parent=world.robot, parent_link=world.base_link))

    if pull_bases:
        for joint_name, color in zip(ALL_JOINTS, spaced_colors(len(ALL_JOINTS))):
            base_confs = list(load_pull_base_poses(world, joint_name))
            handles.extend(visualize_base_confs(world, joint_name, base_confs, color=color))

    #if inverse_place:
    #    for name in world.movable:
    #        body = world.get_body(name)
    #        pose = get_pose(body)
    #        surface_name = world.get_supporting(name)
    #        if surface_name is None:
    #            continue
    #        for grasp_type, color in zip(GRASP_TYPES, spaced_colors(len(GRASP_TYPES))):
    #            base_confs = []
    #            for grasp in get_grasps(world, name, grasp_types=[grasp_type]):
    #                tool_pose = multiply(pose, invert(grasp.grasp_pose))
    #                base_confs.extend(load_place_base_poses(world, tool_pose, surface_name, grasp_type))
    #            handles.extend(visualize_base_confs(world, grasp_type, base_confs, color=color))

    return handles