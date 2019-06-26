import numpy as np
import os
import signal

from pybullet_tools.pr2_utils import draw_viewcone
from pybullet_tools.utils import set_joint_positions, joints_from_names, pose_from_tform, link_from_name, get_link_pose, \
    child_link_from_joint, multiply, invert, set_pose, joint_from_name, set_joint_position, get_pose, tform_from_pose, \
    get_movable_joints, get_joint_names, get_joint_positions, unit_pose, get_links, BASE_LINK
from utils import get_ycb_obj_path

ISSAC_REFERENCE_FRAME = 'base_link' # Robot base
ISSAC_WORLD_FRAME = 'world' # ue_world | world | walls | sektion
ISAAC_CAMERA_FRAME = '00_zed_left' # Prefix of 00 for movable objects and camera

# current_view = view  # Current environment area we are in
# view_root = "%s_base_link" % view_tags[view]


def kill_lula():
    # Kill Lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)

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

def get_world_from_model(observer, entity, body):
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
    frame_from_model = get_pose(body)
    entity_from_model = multiply(invert(frame_from_entity), frame_from_model)
    world_from_model = multiply(world_from_entity, entity_from_model)
    return world_from_model

def update_robot(world, domain, observer, world_state):
    entity = world_state.entities[domain.robot]
    # Update joint positions
    set_joint_positions(world.robot, world.base_joints, entity.carter_pos)
    arm_joints = joints_from_names(world.robot, entity.joints)
    set_joint_positions(world.robot, arm_joints, entity.q)
    world.set_gripper(entity.gripper)  # 'gripper_joint': 'panda_finger_joint1'
    world_from_entity = get_world_from_model(observer, entity, world.robot)
    set_pose(world.robot, world_from_entity)

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

def update_world(world, domain, observer, world_state):
    from brain_ros.ros_world_state import RobotArm, FloatingRigidBody, Drawer, RigidBody
    #dump_dict(world_state)
    #print(world_state.get_frames())
    # Using state is nice because it applies noise

    print('Entities:', sorted(world_state.entities))
    world.reset()
    update_robot(world, domain, observer, world_state)
    for name, entity in world_state.entities.items():
        #dump_dict(entity)
        #entity.obj_type
        #entity.semantic_frames
        if isinstance(entity, RobotArm):
            pass
        elif isinstance(entity, FloatingRigidBody): # Must come before RigidBody
            ycb_obj_path = get_ycb_obj_path(entity.obj_type)
            world.add_body(name, ycb_obj_path, color=np.ones(4), mass=1)
            body = world.get_body(name)
            world_from_entity = get_world_from_model(observer, entity, body)
            set_pose(body, world_from_entity)
        elif isinstance(entity, Drawer):
            joint = joint_from_name(world.kitchen, entity.joint_name)
            set_joint_position(world.kitchen, joint, entity.q)
            world_from_entity = get_world_from_model(observer, entity, world.kitchen)
            set_pose(world.kitchen, world_from_entity)
            #entity.closed_dist
            #entity.open_dist
        elif isinstance(entity, RigidBody):
            # TODO: indigo_countertop does not exist
            print("Warning! {} was not processed".format(name))
        else:
            raise NotImplementedError(entity.__class__)
    world.update_floor()
    world.update_custom_limits()
    #draw_viewcone(CAMERA_POSE)

################################################################################

TEMPLATE = '%s_1'

def update_isaac_sim(domain, sim_manager, world):
    # RobotConfigModulator seems to just change the default config
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/robot_config_modulator.py
    #sim_manager = trial_manager.sim
    #ycb_objects = kitchen_poses.supported_ycb_objects

    #sim_manager.pause()
    for name, body in world.body_from_name.items():
        full_name = TEMPLATE % name
        sim_manager.set_pose(full_name, tform_from_pose(get_pose(body)), do_correction=False)
        print(full_name)

    for body in [world.kitchen]: #, world.robot]:
        # TODO: set kitchen base pose
        joints = get_movable_joints(body)
        names = get_joint_names(body, joints)
        positions = get_joint_positions(body, joints)
        sim_manager.set_joints(names, positions)
        print(names)
        #for joint in get_movable_joints(body):
        #    # Doesn't seem to fail if the kitchen joint doesn't exist
        #    name = get_joint_name(body, joint)
        #    value = get_joint_position(body, joint)
        #     sim_manager.set_joints([name], [value])

    config_modulator = domain.config_modulator
    #robot_name = domain.robot # arm
    robot_name = TEMPLATE % sim_manager.robot_name
    #print(kitchen_poses.ycb_place_in_drawer_q) # 7 DOF
    #config_modulator.send_config(get_joint_positions(world.robot, world.arm_joints)) # Arm joints
    # TODO: config modulator doesn't seem to have a timeout
    reference_link = link_from_name(world.robot, get_robot_reference_frame(domain))
    robot_pose = get_link_pose(world.robot, reference_link)
    sim_manager.set_pose(robot_name, tform_from_pose(robot_pose), do_correction=False)
    print(robot_name)

    #sim_manager.reset()
    #sim_manager.wait_for_services()
    #sim_manager.dr() # Domain randomization
    #rospy.sleep(1.)
    #for name in world.all_bodies:
    #    sim_manager.set_pose(name, get_pose(body))
