import time

import rospy
from pybullet_tools.utils import link_from_name, get_link_pose, multiply, tform_from_pose, get_pose, get_movable_joints, \
    get_joint_names, get_joint_positions
from src.issac import lookup_pose, ISSAC_WORLD_FRAME, UNREAL_WORLD_FRAME, TEMPLATE, ISSAC_CARTER_FRAME
from src.utils import LEFT_CAMERA


def set_isaac_camera(sim_manager, camera_pose):
    # TODO: could make the camera follow the robot_entity around
    from brain_ros.ros_world_state import make_pose
    from isaac_bridge.manager import ros_camera_pose_correction
    camera_tform = make_pose(camera_pose)
    camera_tform = ros_camera_pose_correction(camera_tform, LEFT_CAMERA)
    sim_manager.set_pose(LEFT_CAMERA, camera_tform, do_correction=False)
    # trial_manager.set_camera(randomize=False)


def update_isaac_robot(observer, sim_manager, world):
    unreal_from_world = lookup_pose(observer.tf_listener, source_frame=ISSAC_WORLD_FRAME,
                                    target_frame=UNREAL_WORLD_FRAME)
    # robot_name = domain.robot # arm
    robot_name = TEMPLATE % sim_manager.robot_name
    carter_link = link_from_name(world.robot, ISSAC_CARTER_FRAME)
    world_from_carter = get_link_pose(world.robot, carter_link)
    unreal_from_carter = multiply(unreal_from_world, world_from_carter)
    sim_manager.set_pose(robot_name, tform_from_pose(unreal_from_carter), do_correction=False)


def update_isaac_poses(interface, world):
    unreal_from_world = lookup_pose(interface.observer.tf_listener, source_frame=ISSAC_WORLD_FRAME,
                                    target_frame=UNREAL_WORLD_FRAME)
    for name, body in world.body_from_name.items():
        full_name = TEMPLATE % name
        world_from_urdf = get_pose(body)
        unreal_from_urdf = multiply(unreal_from_world, world_from_urdf)
        interface.sim_manager.set_pose(full_name, tform_from_pose(unreal_from_urdf), do_correction=False)


def update_kitchen_joints(interface, world):
    for body in [world.kitchen]: #, world.robot]:
        # TODO: doesn't seem to work for robots
        # TODO: set kitchen base pose
        joints = get_movable_joints(body)
        #joints = world.arm_joints
        # Doesn't seem to fail if the kitchen joint doesn't exist
        # TODO: doesn't seem to actually work
        names = get_joint_names(body, joints)
        positions = get_joint_positions(body, joints)
        interface.sim_manager.set_joints(names, positions, duration=rospy.Duration(5))
        print('Kitchen joints:', names)


def update_isaac_sim(interface, world):
    # RobotConfigModulator just changes the rest config
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/robot_config_modulator.py
    #sim_manager = trial_manager.sim
    #ycb_objects = kitchen_poses.supported_ycb_objects

    interface.pause_simulation()
    update_isaac_poses(interface, world)
    # TODO: freezes here with newest version of srl_system
    #update_kitchen_joints(interface, world)

    # Changes the default configuration
    #config_modulator = domain.config_modulator
    #print(kitchen_poses.ycb_place_in_drawer_q) # 7 DOF
    #config_modulator.send_config(get_joint_positions(world.robot, world.arm_joints)) # Arm joints
    update_isaac_robot(interface.observer, interface.sim_manager, world)
    #print(get_camera())
    #set_isaac_camera(sim_manager, camera_pose)
    time.sleep(1.0)
    # rospy.sleep(1.) # Small sleep might be needed
    interface.resume_simulation()

    #sim_manager.reset()
    #sim_manager.wait_for_services()
    #sim_manager.dr() # Domain randomization
    #for name in world.all_bodies:
    #    sim_manager.set_pose(name, get_pose(body))