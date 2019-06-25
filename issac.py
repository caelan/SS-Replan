import numpy as np

from pybullet_tools.pr2_utils import draw_viewcone
from pybullet_tools.utils import set_joint_positions, joints_from_names, pose_from_tform, link_from_name, get_link_pose, \
    child_link_from_joint, multiply, invert, set_pose, joint_from_name, set_joint_position, get_pose, tform_from_pose, \
    get_movable_joints, get_joint_names, get_joint_positions
from utils import get_ycb_obj_path, CAMERA_POSE


def get_robot_reference_frame(domain):
    entity = domain.root.entities[domain.robot]
    _, reference_frame = entity.base_frame.split('/')  # 'measured/right_gripper'
    return reference_frame


def update_robot(domain, world_state, world):
    entity = world_state.entities[domain.robot]
    body = world.robot
    # Update joint positions
    set_joint_positions(body, world.base_joints, entity.carter_pos)
    arm_joints = joints_from_names(body, entity.joints)
    set_joint_positions(body, arm_joints, entity.q)
    world.set_gripper(entity.gripper)  # 'gripper_joint': 'panda_finger_joint1'

    # Update base pose
    pose = pose_from_tform(entity.pose)
    _, reference_frame = entity.base_frame.split('/')  # 'measured/right_gripper'
    tool_link = link_from_name(body, get_robot_reference_frame(domain))
    tool_pose = get_link_pose(body, tool_link)
    base_link = child_link_from_joint(world.base_joints[-1])
    # base_link = parent_link_from_joint(body, world.arm_joints[0])
    base_pose = get_link_pose(body, base_link)
    arm_from_base = multiply(invert(tool_pose), base_pose)
    #arm_from_base = get_relative_pose(tool_link, base_link)
    reference_pose = multiply(pose, arm_from_base)
    # print(entity.get_pose_semantic_safe(arg1, arg2))
    set_pose(body, reference_pose)


def update_world(world, domain, world_state):
    from brain_ros.ros_world_state import RobotArm, FloatingRigidBody, Drawer, RigidBody
    #dump_dict(world_state)
    #print(world_state.get_frames())
    # TODO: remove old bodies

    print('Entities:', sorted(world_state.entities))
    update_robot(domain, world_state, world)
    for name, entity in world_state.entities.items():
        #dump_dict(entity)
        body = None
        #entity.obj_type
        #entity.semantic_frames
        matrix = entity.pose
        pose = pose_from_tform(matrix)
        #print(entity.get_frames())
        if isinstance(entity, RobotArm):
            continue
        elif isinstance(entity, FloatingRigidBody): # Must come before RigidBody
            ycb_obj_path = get_ycb_obj_path(entity.obj_type)
            world.add_body(name, ycb_obj_path, color=np.ones(4), mass=1)
            body = world.get_body(name)
        elif isinstance(entity, Drawer):
            body = world.kitchen
            joint = joint_from_name(world.kitchen, entity.joint_name)
            set_joint_position(world.kitchen, joint, entity.q)
            reference_frame = entity.base_frame
            tool_link = link_from_name(body, reference_frame)
            tool_pose = get_link_pose(body, tool_link)
            base_pose = get_pose(world.kitchen)
            arm_from_base = multiply(invert(tool_pose), base_pose)
            pose = multiply(pose, arm_from_base)
            #entity.closed_dist
            #entity.open_dist
        elif isinstance(entity, RigidBody):
            # TODO: indigo_countertop does not exist
            print("Warning! {} was not processed".format(name))
        else:
            raise NotImplementedError(entity.__class__)
        if body is None:
            continue
        set_pose(body, pose)
    world.update_floor()
    world.update_custom_limits()
    draw_viewcone(CAMERA_POSE)

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