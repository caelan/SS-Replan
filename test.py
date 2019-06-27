#!/usr/bin/env python

import sys
import os

PDDLSTREAM_PATH = os.path.abspath(os.path.join(os.getcwd(), 'pddlstream'))
PYBULLET_PATH = os.path.join(PDDLSTREAM_PATH, 'examples/pybullet/utils')
sys.path.extend([PDDLSTREAM_PATH, PYBULLET_PATH])

import rospy
import traceback
import numpy as np

import brain_ros.kitchen_domain as kitchen_domain
import brain_ros.kitchen_policies as kitchen_policies
from brain_ros.sim_test_tools import TrialManager
#from brain_ros.moveit import MoveitBridge

from pybullet_tools.utils import LockRenderer, set_camera_pose, WorldSaver, \
    wait_for_user, draw_pose, get_pose, set_point, get_point, get_movable_joints, set_joint_positions, get_sample_fn

from issac import update_world, kill_lula, update_isaac_sim
from utils import World
from run import solve_pddlstream, create_args, simulate_plan
from problem import pdddlstream_from_problem

from pddlstream.language.constants import Not, And


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

def constraints_from_plan(plan):
    # TODO: use the task plan to constrain solution
    raise NotImplementedError()

def goal_formula_from_goal(goals):
    #regex = re.compile(r"(\w+)\((\)\n")
    # TODO: return initial literals as well
    init = []
    goal_literals = []
    for head, value in goals:
        predicate, arguments = head.strip(')').split('(')
        args = [arg.strip() for arg in arguments.split(',')]
        if predicate == 'on_counter':
            obj, = args
            surface = 'indigo_tmp'
            atom = ('On', obj, surface)
        #elif predicate == '???':
        #    atom = ('Holding', obj)
        elif predicate == 'in_drawer':
            obj, drawer = args
            surface = '{}_joint'.format(drawer)
            init.append(('Stackable', obj, surface))
            atom = ('On', obj, surface)
        else:
            print('Skipping {}={}'.format(head, value))
            continue
            #raise NotImplementedError(predicate)
        goal_literals.append(atom if value else Not(atom))
    return init, And(*goal_literals)

def create_trial_args(**kwargs):
    args = lambda: None # Dummy class
    args.side = 'right'
    args.drawer = 'top'
    args.script_timeout = None
    args.no_planning = True
    args.debug_planner = False
    args.pause = False
    args.image = 'img%02d.png'
    args.max_count = 999999
    args.disrupt = False
    args.linear = False
    args.replan = False
    args.seed = None
    args.image_topic = '/sim/left_color_camera/image'
    args.iter = 1
    args.max_t = 3*60
    args.randomize_textures = 0.
    args.randomize_camera = 0.
    args.sigma = 0.
    args.p_sample = 0.
    args.lula_collisions = False
    args.babble = False
    # TODO: use setattr
    return args

TASKS = [
    'open bottom', 'open top', 'pick spam',
    'put away', # tomato_soup_can
    'put spam',
]


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-task', default=TASKS[2], choices=TASKS,
    #                    help='TBD')
    #parser.add_argument('-visualize', action='store_true',
    #                    help='TBD')
    #args = parser.parse_args()
    #task = args.task
    task = TASKS[2]
    #task = TASKS[4]
    args = create_args()
    #if args.seed is not None:
    #    set_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)

    lula = False
    rospy.init_node("test")
    #with HideOutput():
    domain = kitchen_domain.KitchenDomain(sim=True, sigma=0, lula=lula)
    #dump_dict(domain) # domain.view_tags
    #dump_dict(domain.root)  # WorldState: actor, entities
    #print(domain.attachments)  # planner_interface | sigma
    #print(domain.entities)
    #print(domain.config_modulator) # sets the robot config

    #print(domain.robot) # string
    #print(domain.base_link) # string
    #print(domain.get_robot()) # RobotArm

    #trial_args = parse.parse_kitchen_args()
    trial_args = create_trial_args()
    trial_manager = TrialManager(trial_args, domain, lula=lula)
    sim_manager = trial_manager.sim
    objects, goal, plan = trial_manager.get_task(task=task, reset=True) # Need to reset at the start
    goals = [(h.format(o), v) for h, v in goal for o in objects]
    print(goals)
    additional_init, goal_formula = goal_formula_from_goal(goals)
    # TODO: fixed_base_suppressors

    #trial_manager.disable() # Disables collisions

    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/launch/sim_franka.launch#L48

    robot_entity = domain.get_robot()
    #print(dump_dict(robot_entity))
    franka = robot_entity.robot
    # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/lula_franka/franka.py
    #print(dump_dict(franka))
    #gripper = franka.end_effector.gripper
    #gripper.open(speed=.2, actuate_gripper=True, wait=True)
    trial_manager.set_camera(randomize=False)

    world = World(use_gui=True) # args.visualize)
    set_camera_pose(camera_point=[1, -1, 2])

    #observer = ros.RosObserver(domain, sigma=domain.sigma, p_sample=0)
    observer = trial_manager.observer
    #world_state = domain.root
    world_state = observer.observe()
    with LockRenderer():
        update_world(world, domain, observer, world_state)

    #for name in world.movable:
    #    body = world.get_body(name)
    #    set_point(body, get_point(body) + np.array([0, 0, 1]))
    #kitchen_joints = get_movable_joints(world.kitchen)
    #set_joint_positions(world.kitchen, kitchen_joints, get_sample_fn(world.kitchen, kitchen_joints)())
    #robot_joints = world.arm_joints + world.gripper_joints
    #set_joint_positions(world.robot, robot_joints, get_sample_fn(world.robot, robot_joints)())
    #set_point(world.robot, get_point(world.robot) + np.array([0, 0, 1]))
    #sim_manager.pause()
    #update_isaac_sim(domain, observer, sim_manager, world)
    #wait_for_user()
    #sim_manager.pause()
    #return

    # TODO: initial robot base conf is in collision
    problem = pdddlstream_from_problem(world, movable_base=False, fixed_base=True,
                                       collisions=not args.cfree, teleport=args.teleport)
    problem[-2].extend(additional_init)
    problem = problem[:-1] + (goal_formula,)
    saver = WorldSaver()
    commands = solve_pddlstream(world, problem, args)
    simulate_plan(world, commands, args)
    if commands is None:
        return
    #wait_for_user()
    saver.restore()

    #sim_manager.pause() # Careful! This actually does pause the system
    #rospy.sleep(1.) # Small sleep might be needed
    #sim_manager.pause() # The second invocation resumes

    # Either work
    #moveit = MoveitBridge(group_name="panda_arm",
    #    robot_interface=None, dilation=1., lula_world_objects=None, verbose=False, home_q=None)
    moveit = robot_entity.get_motion_interface() # equivalently robot_entity.planner
    #moveit.tracked_objs
    #moveit.use_lula = False
    #moveit.dialation = 1.0
    moveit.open_gripper()

    for command in commands:
        #world_state = observer.observe()
        command.execute(domain, moveit, observer)
    world.destroy()
    return

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

if __name__ == '__main__':
    #main()
    try:
        main()
    except: # BaseException as e:
        traceback.print_exc()
        #raise e
    finally:
        kill_lula()