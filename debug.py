import numpy as np

from pybullet_tools.utils import get_links, get_link_name, draw_aabb, get_aabb, add_text, wait_for_user, remove_debug, \
    link_from_name, get_joints, get_sample_fn, set_joint_positions, sample_placement, set_pose, get_pose, draw_pose, \
    BASE_LINK, get_aabb_center, approximate_as_prism, set_point, Point, pairwise_link_collision, get_link_descendants, \
    set_color, get_collision_data, read_obj, spaced_colors, get_link_pose, aabb_from_points, get_data_pose, tform_mesh, \
    multiply, draw_mesh, get_ray, Ray, get_point, ray_collision, draw_ray
from utils import get_grasps


# Top row: from left right
# 1) baker (1 door) + sektion as the whole shelf
# 2) chewie (2 doors)
# sektion (frame)
# 3) extractor_hood (green)
# 4) dagger (2 doors + frame)
# Bottom row: from left right
# 5) hitman: (hitman_drawer_bottom, hitman_drawer_top, hitman_tmp)
# 6) range (yellow)
# 7) indigo: (indigo_tmp, indigo_drawer_bottom, indigo_drawer_top, indigo_tmp)
# left_handle, right_handle, handle_top, handle_bottom

def label_kitchen_links(kitchen):
    links_from_prefix = {}
    for link in get_links(kitchen):
        name = get_link_name(kitchen, link)
        parts = name.split('_')
        links_from_prefix.setdefault(parts[0], []).append(link)

    handles = []
    for prefix in sorted(links_from_prefix):
        links = links_from_prefix[prefix]
        print(prefix, [get_link_name(kitchen, link) for link in links])
        for link in links:
            name = get_link_name(kitchen, link)
            # 'handle'
            # if 'bottom' in name:
            handles.extend(draw_aabb(get_aabb(kitchen, link)))
            handles.append(add_text(name, parent=kitchen, parent_link=link))
            # set_color(kitchen, color=apply_alpha(WHITE, 1), link=link)
        # for link in links:
        #    set_color(kitchen, color=apply_alpha(BLACK, 1), link=link)
        wait_for_user()
        for handle in handles:
            remove_debug(handle)


def test_kitchen_joints(world):
    for link_name in world.kitchen_yaml['active_task_spaces']:
       link = link_from_name(world.kitchen, link_name)
       aabb = get_aabb(world.kitchen, link)
       draw_aabb(aabb)
       #draw_pose(get_link_pose(kitchen, link), length=0.25)

    #print(get_aabb(kitchen))
    wait_for_user()
    joints = get_joints(world.kitchen)
    sample_fn = get_sample_fn(world.kitchen, joints)
    for _ in range(10):
        conf = sample_fn()
        set_joint_positions(world.kitchen, joints, conf)
        wait_for_user()

def test_eve_joints(robot):
    joint_names = [j.format(a='l') for j in EVE_ARM_JOINTS]
    joint_names = EVE_HIP_JOINTS # EVE_HIP_JOINTS | EVE_ANKLE_JOINTS
    joints = joints_from_names(robot, joint_names)
    #joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, joints)
    wait_for_user()
    while True:
       q = sample_fn()
       set_joint_positions(robot, joints, q)
       wait_for_user()

################################################################################

def place_on_surface(item, drawer, drawer_link=BASE_LINK, step_size=0.001):
    drawer_aabb = get_aabb(drawer, drawer_link)
    draw_aabb(drawer_aabb)

    draw_center = get_aabb_center(drawer_aabb)
    item_center, item_extent = approximate_as_prism(item)
    x, y, z1 = np.array(draw_center) - np.array(item_center)
    z2 = drawer_aabb[0][2] - item_center[2] + item_extent[2]/2
    # TODO: could align the orientation to be in the same frame as the bounding box

    set_point(item, Point(x, y, z1))
    #print(pairwise_link_collision(drawer, drawer_link, item))
    #print(pairwise_collision(drawer, item))

    delta = z2 - z1
    distance = abs(delta)
    path = [z1 + t * delta / distance for t in np.arange(0, distance, step_size)] + [z2]
    for i, z in enumerate(path):
        set_point(item, Point(x, y, z))
        if pairwise_link_collision(drawer, drawer_link, item):
            print(i)
            if i == 0: # Shouldn't happen
                return z1
            return path[i-1]
    # TODO: alternatively, could just place on the appropriate geometry
    return path[-1]

def test_placements(kitchen, block):
    # TODO: can use the door bounding boxes to estimate regions
    surface_link = link_from_name(kitchen, 'indigo_drawer_top')
    place_on_surface(block, kitchen, surface_link)
    wait_for_user()
    for _ in range(10):
        pose = sample_placement(block, kitchen, bottom_link=surface_link)
        set_pose(block, pose)
        wait_for_user()

################################################################################

def test_grasps(world, name):
    #link = link_from_name(robot, 'panda_leftfinger') # panda_leftfinger | panda_rightfinger
    #draw_pose(get_link_pose(robot, link))
    #aabb = get_aabb(robot, link)
    #draw_aabb(aabb)
    #wait_for_user()

    #tool_pose = get_link_pose(world.robot, link_from_name(world.robot, FRANKA_TOOL_LINK))
    for grasp in get_grasps(world, name):
        print(grasp)
        grasp.get_attachment().assign()
        world.set_gripper(grasp.grasp_width)
        #pregrasp = multiply(Pose(point=pre_direction), grasp)
        #block_pose = multiply(tool_pose, grasp) # grasp | pregrasp
        #set_pose(world.get_body(name), block_pose)
        block_pose = get_pose(world.get_body(name))
        handles = draw_pose(block_pose)
        wait_for_user()
        for handle in handles:
            remove_debug(handle)

################################################################################

def create_box_geometry(dx, dy, dz):
    lower, upper = zip(dx, dy, dz)
    center = (np.array(upper) + np.array(lower)) / 2.
    extent = (np.array(upper) - np.array(lower))
    print('Center: {}'.format(center))
    print('Extent: {}'.format(extent))
    return center, extent


def dump_link_cross_sections(world, link_name='hitman_tmp', digits=3):
    #for joint in world.kitchen_joints:
    #    world.open_door(joint)
    link = link_from_name(world.kitchen, link_name)  # hitman_tmp
    for descendant_link in get_link_descendants(world.kitchen, link):
        set_color(world.kitchen, link=descendant_link, color=np.zeros(4))

    [data] = get_collision_data(world.kitchen, link)
    meshes = read_obj(data.filename)
    colors = spaced_colors(len(meshes))
    link_pose = get_link_pose(world.kitchen, link)
    for i, (name, mesh) in enumerate(meshes.items()):
        print(link_name, name)
        for k in range(3):
            print(k, sorted({round(vertex[k], digits) for vertex in mesh.vertices}))
        print(aabb_from_points(mesh.vertices))
        local_pose = get_data_pose(data)
        tformed_mesh = tform_mesh(multiply(link_pose, local_pose), mesh=mesh)
        draw_mesh(tformed_mesh, color=colors[i])
        wait_for_user()

################################################################################

def test_rays(zed_left_point, entity_body):
    vector = get_ray(Ray(zed_left_point, get_point(entity_body)))
    ray = Ray(zed_left_point, zed_left_point + 2*vector)

    ray_result = ray_collision(ray)
    print(ray_result)
    draw_ray(ray, ray_result)
    wait_for_user()