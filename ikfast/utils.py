import os

IKFAST = 'ikfast_franka_panda'
DAE_PATH = 'panda_arm_hand_on_carter.dae'
BASE_LINK = 'chassis_link'
TOOL_LINK = 'right_gripper'
FREE_JOINT = 'panda_joint4' # TODO: try wita link closer to the gripper

#filename = os.path.basename(dae_path)
filename, extension = os.path.splitext(DAE_PATH)
CPP_PATH = '{}.cpp'.format(filename)

ARM_JOINTS = ['panda_joint{}'.format(1+i) for i in range(7)]