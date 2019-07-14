#!/usr/bin/env python2

from __future__ import print_function

import os

from openravepy import Environment, ikfast
from openravepy.misc import InitOpenRAVELogging

InitOpenRAVELogging()
env = Environment()
# If needed, use the following command to generate a .dae file from a .urdf file.
# rosrun collada_urdf urdf_to_collada <input-urdf> <output.dae>

dae_path = 'panda_arm_hand_on_carter.dae'

kinbody = env.ReadRobotURI(dae_path)
env.Add(kinbody)
solver = ikfast.IKFastSolver(kinbody=kinbody)

base_link = kinbody.GetLink('chassis_link').GetIndex()
ee_link = kinbody.GetLink('right_gripper').GetIndex()
free_joints = [
    #kinbody.GetJoint('l_upper_arm_roll_joint').GetDOFIndex(), # Third link
    kinbody.GetJoint('panda_joint4').GetDOFIndex(),
]

chaintree = solver.generateIkSolver(baselink=base_link,
									eelink=ee_link,
									freeindices=free_joints,
									solvefn=ikfast.IKFastSolver.solveFullIK_6D)

#filename = os.path.basename(dae_path)
filename, extension = os.path.split(dae_path)
cpp_path = '{}.cpp'.format(filename)

code = solver.writeIkSolver(chaintree)
with open(cpp_path, 'w') as f:
    f.write(code)
print('Wrote', cpp_path)
