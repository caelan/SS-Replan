import numpy as np
import os
import signal

from pybullet_tools.pr2_utils import draw_viewcone, get_detection_cone, get_viewcone
from pybullet_tools.utils import set_joint_positions, joints_from_names, pose_from_tform, link_from_name, get_link_pose, \
    child_link_from_joint, multiply, invert, set_pose, joint_from_name, set_joint_position, get_pose, tform_from_pose, \
    get_movable_joints, get_joint_names, get_joint_positions, unit_pose, get_links, \
    BASE_LINK, apply_alpha, RED, wait_for_user, LockRenderer, dump_body, draw_pose, \
    point_from_pose, euler_from_quat, quat_from_pose, Pose, Point, Euler, base_values_from_pose, \
    pose_from_base_values, INF
from src.utils import get_ycb_obj_path

ISSAC_PREFIX = '00_'
ISSAC_REFERENCE_FRAME = 'base_link' # Robot base
UNREAL_WORLD_FRAME = 'ue_world'
ISSAC_WORLD_FRAME = 'world' # world | walls | sektion
ISAAC_CAMERA_FRAME = ISSAC_PREFIX + 'zed_left' # Prefix of 00 for movable objects and camera
ISSAC_CARTER_FRAME = 'chassis_link' # The link after theta
CONTROL_TOPIC = '/sim/desired_joint_states'

# current_view = view  # Current environment area we are in
# view_root = "%s_base_link" % view_tags[view]


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

def get_robot_reference_frame(domain):
    entity = domain.root.entities[domain.robot]
    _, reference_frame = entity.base_frame.split('/')  # 'measured/right_gripper'
    return reference_frame

def get_world_from_model(observer, entity, body, model_link=BASE_LINK):
    # TODO: be careful with how base joints are handled
    reference_frame = entity.current_root  # Likely ISSAC_REFERENCE_FRAME
    # world_from_reference = unit_pose()
    world_from_reference = lookup_pose(observer.tf_listener, source_frame=reference_frame)

    entity_frame = entity.base_frame.split('/')[-1]
    reference_from_entity = pose_from_tform(entity.pose)
    world_from_entity = multiply(world_from_reference, reference_from_entity)
    #print(name, entity_frame, reference_frame)
    # print(entity.get_frames())
    # print(entity.get_pose_semantic_safe(arg1, arg2))

    entity_link = link_from_name(body, entity_frame) if get_links(body) else BASE_LINK
    frame_from_entity = get_link_pose(body, entity_link)
    frame_from_model = get_link_pose(body, model_link)
    entity_from_model = multiply(invert(frame_from_entity), frame_from_model)
    world_from_model = multiply(world_from_entity, entity_from_model)
    return world_from_model

def update_robot(world, domain, observer, world_state):
    entity = world_state.entities[domain.robot]
    # Update joint positions
    world.set_base_conf(entity.carter_pos) # Should be 0
    arm_joints = joints_from_names(world.robot, entity.joints)
    set_joint_positions(world.robot, arm_joints, entity.q)
    world.set_gripper(entity.gripper)  # 'gripper_joint': 'panda_finger_joint1'
    world_from_entity = get_world_from_model(observer, entity, world.robot) #, model_link=BASE_LINK)

    base_values = base_values_from_pose(world_from_entity, tolerance=INF)
    entity_from_origin = pose_from_base_values(base_values)
    world_from_origin = multiply(world_from_entity, invert(entity_from_origin))
    set_pose(world.robot, world_from_origin)
    world.set_base_conf(base_values)

    print(entity.carter_pos)
    print(base_values)

def lookup_pose(tf_listener, source_frame, target_frame=ISSAC_WORLD_FRAME):
    from brain_ros.ros_world_state import make_pose
    import rospy
    import tf2_ros
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

def display_kinect(world, observer):
    from image_geometry import PinholeCameraModel
    from sensor_msgs.msg import CameraInfo
    import rospy

    if world.kinects:
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
        #camera_matrix[:2, 2] *= 2
        #print(camera_matrix)
        #print(camera_info)
        #camera_matrix = None
        #print(camera_matrix)

        # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/perception_tools/ros_perception.py
        world_from_camera = lookup_pose(observer.tf_listener, ISAAC_CAMERA_FRAME)
        cone_body = get_viewcone(camera_matrix=camera_matrix, color=apply_alpha(RED, 0.1))
        set_pose(cone_body, world_from_camera)
        world.kinects.append(cone_body)
        #kitchen_from_camera = multiply(invert(get_pose(world.kitchen)), world_from_camera)
        #print(kitchen_from_camera)

        # draw_viewcone(world_from_camera)
        # /sim/left_color_camera/camera_info
        # /sim/left_depth_camera/camera_info
        # /sim/right_color_camera/camera_info
        # TODO: would be cool to display the kinect2 as well

    observer.camera_sub = rospy.Subscriber(
        "/sim/{}_{}_camera/camera_info".format('left', 'color'),
        CameraInfo, callback, queue_size=1) # right, depth

def update_world(world, domain, observer, world_state):
    from brain_ros.ros_world_state import RobotArm, FloatingRigidBody, Drawer, RigidBody
    #dump_dict(world_state)
    #print(world_state.get_frames())
    # Using state is nice because it applies noise

    print('Entities:', sorted(world_state.entities))
    #world.reset()
    update_robot(world, domain, observer, world_state)
    for name, entity in world_state.entities.items():
        #entity.obj_type
        #entity.semantic_frames
        if isinstance(entity, RobotArm):
            pass
        elif isinstance(entity, FloatingRigidBody): # Must come before RigidBody
            if name not in world.body_from_name:
                ycb_obj_path = get_ycb_obj_path(entity.obj_type)
                world.add_body(name, ycb_obj_path, color=np.ones(4), mass=1)
            body = world.get_body(name)
            world_from_entity = get_world_from_model(observer, entity, body)
            set_pose(body, world_from_entity)
        elif isinstance(entity, Drawer):
            joint = joint_from_name(world.kitchen, entity.joint_name)
            set_joint_position(world.kitchen, joint, entity.q)
            world_from_entity = get_world_from_model(observer, entity, world.kitchen)
            with LockRenderer():
                set_pose(world.kitchen, world_from_entity)
            #entity.closed_dist
            #entity.open_dist
        elif isinstance(entity, RigidBody):
            # TODO: indigo_countertop does not exist
            print("Warning! {} was not processed".format(name))
        else:
            raise NotImplementedError(entity.__class__)
    # TODO: draw floor under the robot instead?
    display_kinect(world, observer)
    #draw_pose(get_pose(world.robot), length=3)
    #draw_pose(get_link_pose(world.robot, world.base_link), length=1)
    #wait_for_user()

################################################################################

TEMPLATE = '%s_1'

def update_isaac_sim(domain, observer, sim_manager, world):
    # RobotConfigModulator seems to just change the default config
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/robot_config_modulator.py
    #sim_manager = trial_manager.sim
    #ycb_objects = kitchen_poses.supported_ycb_objects

    unreal_from_world = lookup_pose(observer.tf_listener, source_frame=ISSAC_WORLD_FRAME,
                                    target_frame=UNREAL_WORLD_FRAME)

    for name, body in world.body_from_name.items():
        full_name = TEMPLATE % name
        world_from_urdf = get_pose(body)
        unreal_from_urdf = multiply(unreal_from_world, world_from_urdf)
        sim_manager.set_pose(full_name, tform_from_pose(unreal_from_urdf), do_correction=False)

    for body in [world.kitchen]: #, world.robot]:
        # TODO: doesn't seem to work for robots
        # TODO: set kitchen base pose
        joints = get_movable_joints(body)
        #joints = world.arm_joints
        # Doesn't seem to fail if the kitchen joint doesn't exist
        names = get_joint_names(body, joints)
        positions = get_joint_positions(body, joints)
        sim_manager.set_joints(names, positions)
        print('Kitchen joints:', names)

    # Changes the default configuration
    #config_modulator = domain.config_modulator
    #print(kitchen_poses.ycb_place_in_drawer_q) # 7 DOF
    #config_modulator.send_config(get_joint_positions(world.robot, world.arm_joints)) # Arm joints

    #robot_name = domain.robot # arm
    robot_name = TEMPLATE % sim_manager.robot_name
    carter_link = link_from_name(world.robot, ISSAC_CARTER_FRAME)
    world_from_carter = get_link_pose(world.robot, carter_link)
    unreal_from_carter = multiply(unreal_from_world, world_from_carter)
    sim_manager.set_pose(robot_name, tform_from_pose(unreal_from_carter), do_correction=False)

    #sim_manager.reset()
    #sim_manager.wait_for_services()
    #sim_manager.dr() # Domain randomization
    #for name in world.all_bodies:
    #    sim_manager.set_pose(name, get_pose(body))
