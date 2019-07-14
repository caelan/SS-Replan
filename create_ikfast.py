#!/usr/bin/env python

# Run this file from lis_ltamp/control_tools
# That way it has the right file path for the .dae file

# This specific file depends on openrave.
# But once the C++ files are generated, openrave is not needed.
from openravepy import *
from openravepy import ikfast
from openravepy.misc import InitOpenRAVELogging

InitOpenRAVELogging()
env = Environment()
# If needed, use the following command to generate a .dae file from a .urdf file.
# rosrun collada_urdf urdf_to_collada <input-urdf> <output.dae>
# There's also a pr2.dae file. No idea what the differences are.
# Using this file because the example in ltamp-pr2 does.
#kinbody = env.ReadRobotURI('pr2_with_sensor.dae')
kinbody = env.ReadRobotURI('../../../../models/pr2_description/pr2_with_sensor.dae')
env.Add(kinbody)
solver = ikfast.IKFastSolver(kinbody=kinbody)

# arm = 'left'
arm = 'right'

base_link = 0
ee_link = 0
free_joints = []

if arm == 'left':
	# l_shoulder_pan_link is usually link 36
	# base_link = kinbody.GetLink('l_shoulder_pan_link').GetIndex()
	# l_gripper_tool_frame is usually link 57
	ee_link = kinbody.GetLink('l_gripper_tool_frame').GetIndex()
	free_joints = [kinbody.GetJoint('torso_lift_joint').GetDOFIndex(),
					kinbody.GetJoint('l_upper_arm_roll_joint').GetDOFIndex()]
elif arm == 'right':
	# r_shoulder_pan_link is usually link 60
	# base_link = kinbody.GetLink('r_shoulder_pan_link').GetIndex()
	# l_gripper_tool_frame is usually link 81
	ee_link = kinbody.GetLink('r_gripper_tool_frame').GetIndex()
	free_joints = [kinbody.GetJoint('torso_lift_joint').GetDOFIndex(),
					kinbody.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()]

base_link = kinbody.GetLink('base_link').GetIndex() # for base link of pr2 model, usually 1


# baselink and eelink are the link indexes for the links which the ik is done relative to.
# You can get those using kinbody.GetLink('link_name').GetIndex()
# freeindices are the joint DOF indexes for the joints which we provide the values for.
# You can run kinbody.GetChain(baselink, eelink) to get a list of all the joints between the two links
# Check that joint.GetDOFIndex() != -1 to discard the fixed joints in the list
# You will have to provide values for the joints you provide in freeindices
# Since this generates a closed form solver, there must be exactly 6 non-free joints in the chain
# I chose the torso (because it should rarely move) and the upper arm roll (because Tomas also chose it in 2011)
chaintree = solver.generateIkSolver(baselink=base_link,
									eelink=ee_link,
									freeindices=free_joints, # corresponds to torso lift and upper arm roll joint DOF indexes
									solvefn=ikfast.IKFastSolver.solveFullIK_6D)
code = solver.writeIkSolver(chaintree)
open(arm + '_arm_ik.cpp', 'w').write(code)

# It's fine if it prints the following message after it's done writing the C++ code. That's normal.
#
# terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::lock_error> >'
#   what():  boost: mutex lock failed in pthread_mutex_lock: Invalid argument
# Aborted (core dumped)