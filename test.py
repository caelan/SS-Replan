#!/usr/bin/env python

import sys
import os

PDDLSTREAM_PATH = os.path.abspath(os.path.join(os.getcwd(), 'pddlstream'))
PYBULLET_PATH = os.path.join(PDDLSTREAM_PATH, 'examples/pybullet/utils')
sys.path.extend([PDDLSTREAM_PATH, PYBULLET_PATH])

import rospy
import signal
import traceback
import numpy as np

import brain_ros.kitchen_domain as kitchen_domain
import brain_ros.parse as parse
import brain_ros.lula_policies as lula_policies
import brain_ros.kitchen_policies as kitchen_policies
import brain.status as status

from brain_ros.sim_test_tools import TrialManager
from brain.action import Policy

#from moveit_msgs.msg import RobotTrajectory


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

# Simple loop to try reaching the goal. It uses the execution policy.
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/test_tools.py#L12

# set_pose, set_joints, exit
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/manager.py

class CaelanManager(TrialManager):
    def get_plan(self, goal, plan, plan_args):
        return []

################################################################################

def open_gripper():
    return kitchen_policies.OpenGripper(speed=0.1, duration=1., actuate=True, unsuppress=True, v_tol=0.1)

def close_gripper():
    return kitchen_policies.CloseGripper(speed=0.1, actuate=True, duration=0.)

# go_config, wait_for_target_config
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/frame_commander.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_tools.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/moveit.py#L463
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_control/lula_control/frame_commander.py#L743
# Policies: LulaStepAxis, LulaSendPose, OpenGripper, CloseGripper, PlanExecutionPolicy, LulaTracker, CarterSimpleMove, CarterMoveToPose
# wait_for_target_config

def stuff():
    lula_policies.LulaSendPose(
        lookup_table=kitchen_poses.drawer_to_approach_handle,
        config_modulator=self.config_modulator,
        is_transport=True,
        config=kitchen_poses.open_drawer_q, )

# open(self, speed=.2, actuate_gripper=True, wait=True)
# close(self, attach_obj=None, speed=.2, force=40., actuate_gripper=True, wait=True)
# self.end_effector.go_config(retract_posture_config
# https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/lula_franka/franka.py

################################################################################

class FollowTrajectory(Policy):

    def __init__(self):
        pass

    def enter(self, domain, world_state, actor, goal):
        self.cmd_sent = False
        self.start_time = rospy.Time.now()
        return True

    def __call__(self, domain, world_state, actor, goal):
        #if domain.sigma > 0:
        #    self.noise = np.random.randn(3) * domain.sigma
        #else:
        #    self.noise = np.zeros((3,))
        # Get the arm/actor -- this should be the franka.
        arm = world_state[actor]
        move = arm.get_motion_interface()

    def exit(self, domain, world_state, actor, x):
        return True


################################################################################

from pybullet_tools.utils import user_input, wait_for_user, HideOutput, \
    set_joint_positions, joints_from_names, pose_from_tform, joint_from_name, \
    set_joint_position, set_pose, get_link_pose, link_from_name, LockRenderer, \
    child_link_from_joint, multiply, invert, parent_link_from_joint, dump_body, get_pose
from utils import World, load_ycb
import brain_ros.ros_world_state as ros

def update_world(world, world_state):
    dump_dict(world_state)
    #dump_body(world.kitchen)
    #print(world_state.get_frames())
    #user_input()

    for name, entity in world_state.entities.items():
        dump_dict(entity)
        body = None
        #entity.obj_type
        #entity.semantic_frames
        matrix = entity.pose
        pose = pose_from_tform(matrix)
        #print(entity.get_frames())
        if isinstance(entity, ros.RobotArm):
            body = world.robot
            set_joint_positions(body, world.base_joints, entity.carter_pos)
            arm_joints = joints_from_names(body, entity.joints)
            set_joint_positions(body, arm_joints, entity.q)
            world.set_gripper(entity.gripper) # 'gripper_joint': 'panda_finger_joint1'
            _, reference_frame = entity.base_frame.split('/') # 'measured/right_gripper'
            tool_link = link_from_name(body, reference_frame)
            tool_pose = get_link_pose(body, tool_link)
            base_link = child_link_from_joint(world.base_joints[-1])
            #base_link = parent_link_from_joint(body, world.arm_joints[0])
            base_pose = get_link_pose(body, base_link)
            arm_from_base = multiply(invert(tool_pose), base_pose)
            pose = multiply(pose, arm_from_base)
            #print(entity.get_pose_semantic_safe(arg1, arg2))
        elif isinstance(entity, ros.FloatingRigidBody): # Must come before RigidBody
            body = load_ycb(entity.obj_type)
            print(body)
            assert body is not None
        elif isinstance(entity, ros.Drawer):
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
        elif isinstance(entity, ros.RigidBody):
            # TODO: indigo_countertop does not exist
            pass
        else:
            raise NotImplementedError(entity.__class__)
        if body is None:
            continue
        set_pose(body, pose)

def main():
    rospy.init_node("test")
    #with HideOutput():
    domain = kitchen_domain.KitchenDomain(sim=True, sigma=0, lula=True)
    #dump_dict(domain) # domain.view_tags
    #dump_dict(domain.get_robot())
    #print(domain.base_link)
    #dump_dict(domain.root)  # WorldState: actor, entities
    #print(domain.attachments)  # planner_interface | sigma
    #print(domain.entities)
    #print(domain.config_modulator) # sets the robot config

    world = World(use_gui=True)
    observer = ros.RosObserver(domain, sigma=domain.sigma, p_sample=0)
    #world_state = domain.root
    world_state = observer.observe()
    with LockRenderer():
        update_world(world, world_state)
    wait_for_user()
    return

    #arm = domain.get_robot() # actor
    #move = arm.get_motion_interface()
    #move.execute(plan=JointTrajectory([]), required_orig_err=0.005, timeout=5.0, publish_display_trajectory=True)

    #domain.operators.clear()

    args = parse.parse_kitchen_args()
    man = TrialManager(args, domain)
    #dump_dict(man)
    #dump_dict(man.sim) # set_pose(), set_joints()
    #dump_dict(man.observer) # current_state, tf_listener, world
    #for name in sorted(domain.root.entities):
    #    dump_dict(domain.root.entities[name])
    #world_state = man.observer.observe()
    #dump_dict(world_state)
    #for name in sorted(world_state.entities):
    #    dump_dict(world_state.entities[name])

    # RobotArm: carter_pos, gripper, gripper_joint, joints, q, robot
    # FloatingRigidBody: pose, semantic_frames, base_frame, manipulable, attached
    # Drawer: joint_name, pose, semantic_frames, q, closed_dist, open_dist, open_tol, manipulable
    # RigidBody: pose, obj_type

    # Tasks
    #task = "put away"
    #task = "put spam"
    #task = "open top"
    #task = "open bottom"
    task = "pick spam"

    #objs, goal, plan = man.get_task(task=task, reset=True) # Need to reset at the start
    #print(objs)
    #print(goal)
    #print(plan)

    #problem = TaskPlanningProblem(domain)
    #res = problem.verify(world_state, instantiated_goal, plan)
    #domain.update_logical(root)
    #domain.check(world_state, goal_conditions):

    #res, tries = man.test(goal, plan, ["arm", objs[0]], task)
    #man.go_to_random_start(domain)

    #man.get_plan(goal, plan, plan_args)
    #man.do_random_trial(task="put away", reset=True)

    #execute = PlanExecutionPolicy(goal=goal, plan=plan)
    #execute.enter(domain, world_state, *plan_args)

    #if args.iter <= 0:
    #    while not rospy.is_shutdown():
    #        man.do_random_trial(task="put away", reset=True)
    #else:
    #    for i in range(args.iter):
    #        man.do_random_trial(task="put away", reset=(i == 0))

if __name__ == '__main__':
    #main()
    try:
        main()
    except: # BaseException as e:
        traceback.print_exc()
        #raise e
    finally:
        kill_lula()