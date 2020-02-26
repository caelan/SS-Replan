from src.command import *

# https://gitlab-master.nvidia.com/SRL/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/isaac_bridge/launch/carter_localization_priors.launch#L6
# /isaac/odometry is zero
# /isaac/pose_status is 33.1
#CARTER_X = 33.1
#CARTER_Y = 7.789

class IsaacTrajectory(Trajectory):

    def execute_base(self, interface):
        assert self.joints == self.world.base_joints
        # assert not moveit.use_lula
        carter = interface.carter
        # from src.base import follow_base_trajectory
        # from src.update_isaac import update_isaac_robot
        # from isaac_bridge.manager import SimulationManager
        if carter is None:
            raise NotImplementedError()
            # follow_base_trajectory(self.world, self.path, interface.moveit, interface.observer)
        # elif isinstance(carter, SimulationManager):
        #    sim_manager = carter
        #    interface.pause_simulation()
        #    set_joint_positions(self.robot, self.joints, self.path[-1])
        #    update_isaac_robot(interface.observer, sim_manager, self.world)
        #    time.sleep(DEFAULT_SLEEP)
        #    interface.resume_simulation()
        #    # TODO: teleport attached
        else:
            from src.isaac.carter import command_carter_to_pybullet_goal
            return command_carter_to_pybullet_goal(interface, self.path[-1])

            world_state = domain.root
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/isaac_bridge/src/isaac_bridge/carter_sim.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/2cb8df9ac14b56a5955251cf4325369172c2ba72/packages/isaac_bridge/src/isaac_bridge/carter.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/external/lula_franka/scripts/move_carter.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_policies.py
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/carter_predicates.py#L164
            # TODO: ensure that I transform into the correct base units
            # The base uses its own motion planner
            # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/fixed_base_suppressor.py
            world_state[domain.robot].suppress_fixed_bases()
            carter.move_to_async(self.path[-1])
            # for conf in waypoints_from_path(self.path):
            #    carter.move_to_safe(conf)
            #    #carter.move_to_async(conf)
            #    #carter.simple_move(cmd)
            # carter.simple_stop()
            world_state[domain.robot].unsuppress_fixed_bases()
            # carter.pub_disable_deadman_switch.publish(True)
        return True

    def execute_gripper(self, interface):
        assert self.joints == self.world.gripper_joints
        position = self.path[-1][0]
        # move_gripper_action(position)
        joint = self.joints[0]
        average = np.average(get_joint_limits(self.robot, joint))
        from src.isaac.execution import franka_open_gripper, franka_close_gripper
        if position < average:
            franka_close_gripper(interface)
        else:
            franka_open_gripper(interface)
        return True

    def execute(self, interface):
        # TODO: ensure the same joint name order
        if self.joints == self.world.base_joints:
            success = self.execute_base(interface)
        elif self.joints == self.world.gripper_joints:
            success = self.execute_gripper(interface)
        else:
            from src.isaac.execution import franka_control
            success = franka_control(self.robot, self.joints, self.path, interface)
        # status = joint_state_control(self.robot, self.joints, self.path, domain, moveit, observer)
        # time.sleep(DEFAULT_SLEEP)
        return success

        # if self.joints == self.world.arm_joints:
        #    #world_state = observer.current_state
        #    world_state = interface.update_state()
        #    robot_entity = world_state[interface.domain.robot]
        #    print('Error:', (np.array(robot_entity.q) - np.array(self.path[-1])).round(5))
        #    update_robot(interface)
        #    #wait_for_user('Continue?')

################################################################################

#class BaseTrajectory(Trajectory):
#    def __init__(self, world, robot, joints, path, **kwargs):
#        super(BaseTrajectory, self).__init__(world, robot, joints, path, **kwargs)

class IsaacApproachTrajectory(ApproachTrajectory):
    def execute(self, interface):
        from src.isaac.execution import franka_control
        from src.retime import DEFAULT_SPEED_FRACTION
        if len(self.path) == 1:
            return True
        return franka_control(self.robot, self.joints, self.path, interface,
                              velocity_fraction=self.speed * DEFAULT_SPEED_FRACTION)
        # return super(ApproachTrajectory, self).execute(interface)
        # TODO: finish if error is still large

        # from src.issac import update_robot
        # set_joint_positions(self.robot, self.joints, self.path[-1])
        # target_pose = get_link_pose(self.robot, self.world.tool_link)
        # interface.robot_entity.suppress_fixed_bases()
        # interface.robot_entity.unsuppress_fixed_bases()

        # world_state[domain.robot].suppress_fixed_bases()
        # interface.update_state()
        # update_robot(interface)
        # time.sleep(DEFAULT_SLEEP)

class IsaacDoorTrajectory(DoorTrajectory):
    def execute(self, interface):
        #update_robot(self.world, domain, observer, observer.observe())
        #wait_for_user()
        # TODO: do I still need these?
        #if self.do_pull:
        #    franka_close_gripper(interface)
        #    time.sleep(DEFAULT_SLEEP)
        from src.isaac.execution import franka_control
        success = franka_control(self.robot, self.joints, self.path, interface)
        #time.sleep(DEFAULT_SLEEP)
        return success
        #if self.do_pull:
        #    franka_open_gripper(interface)
        #    time.sleep(DEFAULT_SLEEP)

        #close_gripper(self.robot, moveit)
        #status = joint_state_control(self.robot, self.robot_joints, self.robot_path,
        #                             domain, moveit, observer)
        #open_gripper(self.robot, moveit)

################################################################################s

class IsaacAttachGripper(AttachGripper):
    def execute(self, interface):
        name = self.world.get_name(self.body)
        effort = EFFORT_FROM_OBJECT[name]
        print('Grasping {} with effort {}'.format(name, effort))
        from src.isaac.execution import franka_close_gripper
        franka_close_gripper(interface, effort=effort)
        interface.stop_tracking(name)
        time.sleep(DEFAULT_SLEEP)
        #return close_gripper(self.robot, moveit)
        return True

class IsaacDetach(Detach):
    def execute(self, interface):
        from src.isaac.execution import franka_open_gripper
        if self.world.robot == self.robot:
            franka_open_gripper(interface)
            time.sleep(DEFAULT_SLEEP)
            #return open_gripper(self.robot, moveit)
        return True
