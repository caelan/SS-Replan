import random

from utils import BASE_LINK, TOOL_LINK, FREE_JOINT, ARM_JOINTS

from pybullet_tools.ikfast.utils import get_ik_limits, compute_forward_kinematics, \
    compute_inverse_kinematics, select_solution
from pybullet_tools.utils import multiply, get_link_pose, \
    link_from_name, get_joint_positions, invert, violates_limits, joint_from_name, joints_from_names, get_movable_joints, \
    get_distance


# https://github.com/caelan/ss-pybullet/blob/master/pybullet_tools/ikfast/pr2/ik.py

def get_tool_pose(robot):
    from .ikfast_franka_panda import get_fk
    ik_joints = joints_from_names(robot, ARM_JOINTS)
    conf = get_joint_positions(robot, ik_joints)
    # TODO: this should be linked to ikfast's get numOfJoint function
    assert len(conf) == 7
    base_from_tool = compute_forward_kinematics(get_fk, conf)
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_LINK))
    return multiply(world_from_base, base_from_tool)

#####################################

def is_ik_compiled():
    try:
        from .ikfast_franka_panda import ikfast_franka_panda
        return True
    except ImportError:
        return False


def get_ik_generator(robot, tool_pose, track_limits=False, prev_free_list=[]):
    from .ikfast_franka_panda import get_ik
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_LINK))
    base_from_tool = multiply(invert(world_from_base), tool_pose)
    sampled_limits = get_ik_limits(robot, joint_from_name(robot, *FREE_JOINT), track_limits)
    while True:
        if not prev_free_list:
            sampled_values = [random.uniform(*sampled_limits)]
        else:
            sampled_values = prev_free_list
        ik_joints = joints_from_names(robot, ARM_JOINTS)
        confs = compute_inverse_kinematics(get_ik, base_from_tool, sampled_values)
        yield [q for q in confs if not violates_limits(robot, ik_joints, q)]


def sample_tool_ik(robot, tool_pose, max_attempts=10, closest_only=False, get_all=False, prev_free_list=[], **kwargs):
    generator = get_ik_generator(robot, tool_pose, prev_free_list=prev_free_list, **kwargs)
    ik_joints = get_movable_joints(robot)
    for _ in range(max_attempts):
        try:
            solutions = next(generator)
            if closest_only and solutions:
                current_conf = get_joint_positions(robot, ik_joints)
                solutions = [min(solutions, key=lambda conf: get_distance(current_conf, conf))]
            solutions = list(filter(lambda conf: not violates_limits(robot, ik_joints, conf), solutions))
            return solutions if get_all else select_solution(robot, ik_joints, solutions, **kwargs)
        except StopIteration:
            break
    return None