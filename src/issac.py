from __future__ import print_function

import numpy as np
import os
import signal
import pickle
import rospy
import tf2_ros
import time

#from collections import Counter
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from brain_ros.ros_world_state import RobotArm, FloatingRigidBody, Drawer, RigidBody
from brain_ros.ros_world_state import make_pose

from pybullet_tools.utils import set_joint_positions, joints_from_names, pose_from_tform, link_from_name, get_link_pose, \
    multiply, invert, set_pose, joint_from_name, set_joint_position, get_links, \
    BASE_LINK, base_values_from_pose, point_from_pose, unit_pose, \
    pose_from_base_values, INF, wait_for_user, get_joint_limits, violates_limit, \
    set_renderer, draw_pose, Pose, Point, set_all_static, elapsed_time, get_aabb, aabb_contains_point
from src.utils import get_srl_path, CAMERA_TEMPLATE

SRL_PATH = get_srl_path()

RIGHT = 'right'
LEFT = 'left'
SIDES = [RIGHT, LEFT]

PREFIX_TEMPLATE = '{:02d}'
PREFIX_FROM_SIDE = {
    RIGHT: PREFIX_TEMPLATE.format(0),
    LEFT: PREFIX_TEMPLATE.format(1),
}
# self.domain.view_tags # {'right': '00', 'left': '01'}

KINECT_TEMPLATE = 'kinect{}'
KINECT_FROM_SIDE = {
    RIGHT: KINECT_TEMPLATE.format(1), # indexes from 1
    LEFT: KINECT_TEMPLATE.format(2),
}

DEPTH_PREFIX = 'depth_camera' # Was changed to now be kinect1_depth_optical_frame
DEPTH_FROM_SIDE = {
    RIGHT: DEPTH_PREFIX, # indexes from 1
    LEFT: '{}_2'.format(DEPTH_PREFIX),
}

RIGHT_PREFIX = '{}_'.format(PREFIX_FROM_SIDE[RIGHT])

# TODO: rosparam get /world_frame
ISSAC_FRANKA_FRAME = 'base_link' # Robot base
UNREAL_WORLD_FRAME = 'ue_world'
ISSAC_WORLD_FRAME = 'world' # world | walls | sektion
ISSAC_CARTER_FRAME = 'chassis_link' # The link after theta
#CONTROL_TOPIC = '/sim/desired_joint_states'

NULL_POSE = Pose(Point(z=-2))

PANDA_FULL_CONFIG_PATH = os.path.join(SRL_PATH, 'packages/isaac_bridge/configs/panda_full_config.json')

# current_view = view  # Current environment area we are in
# view_root = "%s_base_link" % view_tags[view]

################################################################################

def update_observer(observer):
    #print('Waiting for observer update')
    last_time = time.time()
    while not observer.update(noise=False):
        rospy.sleep(0.1)
        if 1 < elapsed_time(last_time):
            print('Waiting for observer update')
            last_time = time.time()
    return observer.current_state

def kill_lula():
    # Kill Lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
    # sim_manager.exit()
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py

def dump_dict(obj):
    print()
    print(obj)
    for i, key in enumerate(sorted(obj.__dict__)):
        print(i, key, obj.__dict__[key])
    print(dir(obj))

################################################################################

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/4a902e24b6272fbc50ee5d9ac1f873f49640d93a/packages/external/lula_franka/scripts/move_carter.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/4a902e24b6272fbc50ee5d9ac1f873f49640d93a/packages/brain/src/brain_ros/carter_predicates.py#L218

KEYPOINT_PATH = 'package://lula_franka/data/keypoint_frames/'
LOCALIZATION_ROSPATHS = {
    LEFT: os.path.join(KEYPOINT_PATH, 'dart_localization_frame_right2.pkl'),
    #'open_chewie': os.path.join(keypoint_dir, 'dart_localization_frame_left.pkl'),
    RIGHT: os.path.join(KEYPOINT_PATH, 'dart_localization_frame_left.pkl'),
}
RETRACT_POSTURE_PATH = os.path.join(KEYPOINT_PATH, 'retract_posture.pkl')
# TODO: could use pickled grasps as well

def load_calibrate_conf(side='left'):
    # TODO: sample arm configurations that are visible
    # TODO: calibration by line of sight with many particles on the robot's arm
    from lula_pyutil.util import parse_pkg_name
    #from lula_pyutil.math_util import pack_transform_to_T, unpack_transform_to_frame
    rospath = LOCALIZATION_ROSPATHS[side]
    init_path = parse_pkg_name(rospath)
    dart_init_transform_stamped, dart_init_config = pickle.load(
        open(init_path, 'rb'))
    #init_pose = pack_transform_to_T(
    #    dart_init_transform_stamped.transform)
    #init_frame = unpack_transform_to_frame(init_pose)
    return dart_init_config

################################################################################

# def get_robot_reference_frame(domain):
#    entity = domain.root.entities[domain.robot]
#    _, reference_frame = entity.base_frame.split('/')  # 'measured/right_gripper'
#    return reference_frame

def get_base_pose(observer, entity=None):
    if entity is None:
        domain = observer.domain
        entity = domain.get_robot() #  domain.root.entities[domain.robot]
    base_frame = entity.current_root
    # world_from_base = unit_pose()
    world_from_base = lookup_pose(observer.tf_listener, source_frame=base_frame) # view_root
    # entity.view_root # 00_base_link
    return world_from_base

def get_world_from_model(observer, entity, body, model_link=BASE_LINK):
    # TODO: be careful with how base joints are handled
    world_from_reference = get_base_pose(observer, entity)
    reference_frame = entity.current_root  # Likely ISSAC_REFERENCE_FRAME
    entity_frame = entity.base_frame.split('/')[-1]
    reference_from_entity = pose_from_tform(entity.pose)
    world_from_entity = multiply(world_from_reference, reference_from_entity)
    #print(name, entity_frame, reference_frame)
    # print(entity.get_frames())
    # print(entity.get_pose_semantic_safe(arg1, arg2))
    print('Model frame: {} | Reference frame: {}'.format(entity_frame, reference_frame))
    # TODO: could just lookup frame in TF
    #dump_dict(entity)

    entity_link = link_from_name(body, entity_frame) if get_links(body) else BASE_LINK
    frame_from_entity = get_link_pose(body, entity_link)
    frame_from_model = get_link_pose(body, model_link)
    entity_from_model = multiply(invert(frame_from_entity), frame_from_model)
    world_from_model = multiply(world_from_entity, entity_from_model)
    #print(world_from_model)

    return world_from_model


def lookup_pose(tf_listener, source_frame, target_frame=ISSAC_WORLD_FRAME):
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/sim_test_tools.py
    try:
        tf_pose = tf_listener.lookupTransform(
            target_frame, source_frame, rospy.Time(0))
        return pose_from_tform(make_pose(tf_pose))
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logwarn("UPDATE TRANSFORM:\n"
                      "{}, {}\n"
                      "Err:{}".format(target_frame, source_frame, e))
        return None


################################################################################

def update_robot_conf(interface, entity=None):
    # https://gitlab-master.nvidia.com/srl/srl_system/blob/ea286e95d3e2d46ff5a3389085beb4f9f3fc3f84/packages/brain/src/brain_ros/ros_world_state.py#L494
    # Update joint positions
    if entity is None:
        world_state = interface.update_state()
        entity = world_state.entities[interface.domain.robot]
    world = interface.world
    arm_joints = joints_from_names(world.robot, entity.joints)
    set_joint_positions(world.robot, arm_joints, entity.q)
    world.set_gripper(entity.gripper)  # 'gripper_joint': 'panda_finger_joint1'
    check_limits(world, entity)
    return dict(zip(entity.joints, entity.q))


def update_robot_base(interface, entity=None):
    if entity is None:
        entity = interface.update_state().entities[interface.domain.robot]
    world = interface.world
    carter_values = entity.carter_pos
    # carter_values = entity.carter_interface.running_pose
    print('Carter base:', carter_values)  # will be zero if a carter object isn't created
    # world.set_base_conf(carter_values)
    world.set_base_conf(np.zeros(3))
    world_from_entity = get_world_from_model(interface.observer, entity, world.robot)  # , model_link=BASE_LINK)

    base_values = base_values_from_pose(world_from_entity, tolerance=INF)
    entity_from_origin = pose_from_base_values(base_values)
    world_from_origin = multiply(world_from_entity, invert(entity_from_origin))
    set_pose(world.robot, world_from_origin)
    world.set_base_conf(base_values)
    print('Initial base:', np.array(base_values).round(3))
    # draw_pose(get_pose(world.robot), length=3)
    # draw_pose(get_link_pose(world.robot, world.base_link), length=1)


def update_robot(interface):
    world_state = interface.observer.current_state
    entity = world_state.entities[interface.domain.robot]
    update_robot_conf(interface, entity) # Must come before the base
    update_robot_base(interface, entity)

def check_limits(world, entity):
    violation = False
    arm_joints = joints_from_names(world.robot, entity.joints)
    for i, joint in enumerate(arm_joints):
        if violates_limit(world.robot, joint, entity.q[i]):
            print('Joint {} violates limits: index={}, position={}, range={}'.format(
                entity.joints[i], i, entity.q[i], get_joint_limits(world.robot, joint)))
            violation = True
            # TODO: change the link's color
    if violation:
        set_renderer(enable=True)
        wait_for_user()

################################################################################

def display_kinect(interface, side):
    world = interface.world
    if world.cameras:
        return
    camera_infos = []
    def callback(camera_info):
        if camera_infos:
            return
        camera_infos.append(camera_info)
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(camera_info)
        #cam_model.height, cam_model.width # TODO: also pass these parameters
        camera_matrix = np.array(cam_model.projectionMatrix())[:3, :3]

        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/perception_tools/ros_perception.py
        if interface.simulation:
            camera_frame = RIGHT_PREFIX + CAMERA_TEMPLATE.format(side)
        else:
            #camera_frame = KINECT_FROM_SIDE[side] # 'kinect1'
            camera_frame = DEPTH_FROM_SIDE[side] # depth_camera_2
            #camera_frame = '{}_link'.format(camera_name)
        camera_name = '{}'.format(side)
        print('Received camera info from camera', camera_name)
        world_from_camera = lookup_pose(interface.observer.tf_listener, camera_frame)
        if world_from_camera is None:
            print('Failed to detect pose for camera', camera_name)
        else:
            world.add_camera(camera_name, world_from_camera, camera_matrix)
        camera_sub.unregister()
        # draw_viewcone(world_from_camera)
        # /sim/left_color_camera/camera_info
        # /sim/left_depth_camera/camera_info
        # /sim/right_color_camera/camera_info

    if interface.simulation:
        calibration_topic = "/sim/{}_{}_camera/camera_info".format(side, 'color')
    else:
        camera_name = KINECT_FROM_SIDE[side]
        calibration_topic = '/{}/rgb/camera_info'.format(camera_name)

    camera_sub = rospy.Subscriber(
        calibration_topic,
        CameraInfo, callback, queue_size=1) # right, depth


################################################################################

def update_kitchen(world, world_state):
    position_from_joint = {}
    for name, entity in world_state.entities.items():
        if isinstance(entity, RobotArm) or isinstance(entity, FloatingRigidBody):
            continue
        elif isinstance(entity, Drawer):
            # https://gitlab-master.nvidia.com/srl/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py
            # TODO: ArticulatedRigidBody, Drawer
            # /tracker/axe/joint_states
            # /tracker/baker/joint_states
            joint = joint_from_name(world.kitchen, entity.joint_name)
            set_joint_position(world.kitchen, joint, entity.q)
            position_from_joint[entity.joint_name] = entity.q
            #belief.update_door_conf(entity.joint_name, entity.q)
            #entity.closed_dist
            #entity.open_dist
        elif isinstance(entity, RigidBody):
            # TODO: indigo_countertop does not exist
            print("Warning! {} was not processed".format(name))
        else:
            raise NotImplementedError(entity.__class__)
    return position_from_joint

################################################################################

def load_objects(task):
    world = task.world
    for name in task.objects:
        #if name in world.body_from_name:
        #    continue
        world.add_body(name, color=np.ones(4), mass=0)
        body = world.get_body(name)
        set_pose(body, NULL_POSE)
        draw_pose(unit_pose(), parent=body)
    set_all_static()

def update_objects(interface, world_state, visible): #=set()):
    base_aabb = interface.world.get_base_aabb()
    observation = {}
    for name, entity in world_state.entities.items():
        # entity.obj_type, entity.semantic_frames
        if isinstance(entity, RobotArm):
            continue
        elif isinstance(entity, FloatingRigidBody):  # Must come before RigidBody
            # entity.is_tracked, entity.location_belief, entity.view
            if name not in interface.world.task.objects:
                # TODO: discard/correct later in the pipeline
                continue
            # entity.obj_type
            if name in visible:
                frame_name = RIGHT_PREFIX + name
                pose = lookup_pose(interface.observer.tf_listener, frame_name)
                # pose = get_world_from_model(observer, entity, body)
                # pose = pose_from_tform(entity.pose)
                point = point_from_pose(pose)
                if not aabb_contains_point(point, base_aabb):
                    observation.setdefault(name, []).append(pose)
            #else:
            #    pose = NULL_POSE  # TODO: modify world_state directly?
        elif isinstance(entity, Drawer):
            continue
        elif isinstance(entity, RigidBody):
            # TODO: indigo_countertop does not exist
            print("Warning! {} was not processed".format(name))
        else:
            raise NotImplementedError(entity.__class__)
    return observation

def observe_world(interface, visible=set(), **kwargs):
    #dump_dict(world_state)
    #print(world_state.get_frames())
    # Using state is nice because it applies noise
    world_state = interface.update_state()
    print('Entities:', sorted(world_state.entities))
    print('Visible:', sorted(visible))
    #world.reset()
    update_robot(interface)
    print('Kitchen joints:', update_kitchen(interface.world, world_state))
    return update_objects(interface, world_state, visible=visible, **kwargs)
