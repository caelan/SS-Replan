import numpy as np
import brain_ros.kitchen_policies as kitchen_policies

from src.update_isaac import update_isaac_sim
from pybullet_tools.utils import wait_for_user, set_point, get_point, get_movable_joints, set_joint_positions, \
    get_sample_fn

#from brain_ros.moveit import MoveitBridge
#from isaac_bridge.carter import Carter

# Simple loop to try reaching the goal. It uses the execution policy.
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/test_tools.py#L12

# set_pose, set_joints, exit
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py

# class CaelanManager(TrialManager):
#     def get_plan(self, goal, plan, plan_args):
#         return []

################################################################################

def open_gripper():
    return kitchen_policies.OpenGripper(
        speed=0.1, duration=1., actuate=True, unsuppress=True, v_tol=0.1)

def close_gripper():
    return kitchen_policies.CloseGripper(
        speed=0.1, actuate=True, duration=0.)

# go_config, wait_for_target_config
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/frame_commander.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_tools.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L463
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/frame_commander.py#L743
# Policies: LulaStepAxis, LulaSendPose, OpenGripper, CloseGripper, PlanExecutionPolicy, LulaTracker, CarterSimpleMove, CarterMoveToPose
# wait_for_target_config

# def stuff():
#     lula_policies.LulaSendPose(
#         lookup_table=kitchen_poses.drawer_to_approach_handle,
#         config_modulator=self.config_modulator,
#         is_transport=True,
#         config=kitchen_poses.open_drawer_q, )

# open(self, speed=.2, actuate_gripper=True, wait=True)
# close(self, attach_obj=None, speed=.2, force=40., actuate_gripper=True, wait=True)
# self.end_effector.go_config(retract_posture_config
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/lula_franka/franka.py

################################################################################

# class FollowTrajectory(Policy):
#
#     def __init__(self):
#         pass
#
#     def enter(self, domain, world_state, actor, goal):
#         self.cmd_sent = False
#         self.start_time = rospy.Time.now()
#         return True
#
#     def __call__(self, domain, world_state, actor, goal):
#         #if domain.sigma > 0:
#         #    self.noise = np.random.randn(3) * domain.sigma
#         #else:
#         #    self.noise = np.zeros((3,))
#         # Get the arm/actor -- this should be the franka.
#         arm = world_state[actor]
#         move = arm.get_motion_interface()
#
#     def exit(self, domain, world_state, actor, x):
#         return True


################################################################################

def randomize_sim(world, domain, observer, sim_manager):
    for name in world.movable:
       body = world.get_body(name)
       set_point(body, get_point(body) + np.array([0, 0, 1]))
    kitchen_joints = get_movable_joints(world.kitchen)
    set_joint_positions(world.kitchen, kitchen_joints, get_sample_fn(world.kitchen, kitchen_joints)())
    robot_joints = world.arm_joints + world.gripper_joints
    set_joint_positions(world.robot, robot_joints, get_sample_fn(world.robot, robot_joints)())
    set_point(world.robot, get_point(world.robot) + np.array([0, 0, 1]))
    sim_manager.pause()
    update_isaac_sim(domain, observer, sim_manager, world)
    wait_for_user()
    sim_manager.pause()

################################################################################

def stuff():
    pass
    #arm = domain.get_robot() # actor
    #move = arm.get_motion_interface()
    #move.execute(plan=JointTrajectory([]), required_orig_err=0.005, timeout=5.0, publish_display_trajectory=True)
    #domain.operators.clear()

    #dump_dict(trial_manager)
    #dump_dict(trial_manager.sim) # set_pose(), set_joints()
    #dump_dict(trial_manager.observer) # current_state, tf_listener, world
    #for name in sorted(domain.root.entities):
    #    dump_dict(domain.root.entities[name])
    #world_state = trial_manager.observer.observe()
    #dump_dict(world_state)
    #for name in sorted(world_state.entities):
    #    dump_dict(world_state.entities[name])

    # RobotArm: carter_pos, gripper, gripper_joint, joints, q, robot
    # FloatingRigidBody: pose, semantic_frames, base_frame, manipulable, attached
    # Drawer: joint_name, pose, semantic_frames, q, closed_dist, open_dist, open_tol, manipulable
    # RigidBody: pose, obj_type

    #print(useful_objects)
    #print(goal)
    #print(plan)

    #problem = TaskPlanningProblem(domain)
    #res = problem.verify(world_state, instantiated_goal, plan)
    #domain.update_logical(root)
    #domain.check(world_state, goal_conditions):

    #res, tries = trial_manager.test(goal, plan, ["arm", useful_objects[0]], task)
    #trial_manager.go_to_random_start(domain)

    #trial_manager.get_plan(goal, plan, plan_args)
    #trial_manager.do_random_trial(task="put away", reset=True)

    #execute = PlanExecutionPolicy(goal=goal, plan=plan)
    #execute.enter(domain, world_state, *plan_args)

    #if args.iter <= 0:
    #    while not rospy.is_shutdown():
    #        trial_manager.do_random_trial(task="put away", reset=True)
    #else:
    #    for i in range(args.iter):
    #        trial_manager.do_random_trial(task="put away", reset=(i == 0))

"""
caelan@driodeka:~/Programs/srlstream$ rostopic list | grep -o -P '^.*(?=/feedback)'
/execute_trajectory
/move_group
/pickup
/place
/robot_control_interactive_markers
/world/interactive_control
"""

################################################################################

def test_carter(domain):
    #dump_dict(domain) # domain.view_tags
    #dump_dict(domain.root)  # WorldState: actor, entities
    #print(domain.attachments)  # planner_interface | sigma
    #print(domain.entities)
    #print(domain.config_modulator) # sets the robot config

    #print(domain.robot) # string
    #print(domain.base_link) # string
    #print(domain.get_robot()) # RobotArm

    #observer = ros.RosObserver(domain, sigma=domain.sigma, p_sample=0)

    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/launch/sim_franka.launch#L48
    robot_entity = domain.get_robot()
    #print(dump_dict(robot_entity))
    franka = robot_entity.robot
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/lula_franka/franka.py
    #print(dump_dict(franka))

    # /home/caelan/Programs/srl_system/packages/isaac_bridge/scripts/fake_localization.py
    # packages/isaac_bridge/scripts/fake_localization.py
    # packages/isaac_bridge/src/isaac_bridge/publish_carter_pose_as_tf.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py#L105
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/scripts/move_carter.py
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter_sim.py
    print(robot_entity.carter_interface, robot_entity.carter_pos, robot_entity.carter_vel)

    carter = Carter(goal_threshold_tra=0.15, goal_threshold_rot=np.radians(10.))
    carter.move_to_safe(goal)
    wait_for_user()

    #gripper = franka.end_effector.gripper
    #gripper.open(speed=.2, actuate_gripper=True, wait=True)

    #moveit = MoveitBridge(group_name="panda_arm",
    #    robot_interface=None, dilation=1., lula_world_objects=None, verbose=False, home_q=None)
    moveit = robot_entity.get_motion_interface() # equivalently robot_entity.planner
    #moveit.tracked_objs
    #moveit.use_lula = False
    #moveit.dialation = 1.0
