from __future__ import print_function

import rospy

from src.isaac.issac import update_observer

class Interface(object):
    def __init__(self, args, task, observer, trial_manager=None, deepim=None):
        self.args = args
        self.task = task
        self.observer = observer
        self.trial_manager = trial_manager
        self.deepim = deepim
        self.paused = False
        self.world.interface = self
    @property
    def perception(self):
        return self.deepim
    @property
    def simulation(self):
        return self.trial_manager is not None
    @property
    def observable(self):
        return self.args.observable
    @property
    def world(self):
        return self.task.world
    @property
    def domain(self):
        return self.observer.domain
    @property
    def robot_entity(self): # TODO: actor
        #return self.domain.root.entities[domain.robot]
        return self.domain.get_robot()
    @property
    def moveit(self):
        return self.robot_entity.get_motion_interface()
        #return self.robot_entity.planner
    @property
    def carter(self):
        return self.robot_entity.carter_interface
    @property
    def sim_manager(self):
        if self.trial_manager is None:
            return None
        return self.trial_manager.sim
    #def optimize_base(self):
    #    return self.robot_entity.suppress_fixed_bases()
    #def fix_base(self):
    #    return self.robot_entity.unsuppress_fixed_bases()
    def disable_collisions(self):
        self.trial_manager.disable()
    def pause_simulation(self):
        # TODO: context manager
        if not self.paused:
            self.sim_manager.pause()
            self.paused = True
    def resume_simulation(self):
        if self.paused:
            self.sim_manager.pause()
            self.paused = False
    def update_state(self):
        return update_observer(self.observer)
    def stop_tracking(self, name):
        assert self.perception is not None
        return self.perception.stop_tracking(name)
    def localize_all(self):
        if self.simulation:
            return
        # Detection & Tracking
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/object_administrator.py
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/fixed_base_suppressor.py
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py#L182
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L470

        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L427
        #dart = LulaInitializeDart(localization_rospaths=LOCALIZATION_ROSPATHS,
        #                          time=6., config_modulator=domain.config_modulator, views=domain.view_tags)
        # Robot calibration policy

        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/lula_policies.py#L46
        #obj = world_state.entities[goal]
        #obj.localize()
        #obj.detect()
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py#L182
        # https://gitlab-master.nvidia.com/SRL/srl_system/blob/master/packages/lula_dart/lula_dartpy/object_administrator.py
        #administrator = ObjectAdministrator(obj_frame, wait_for_connection=False)
        #administrator.activate() # localize
        #administrator.detect_once() # detect
        #administrator.deactivate() # stop_localizing

        #return
        world_state = self.observer.current_state
        for name in self.task.prior:
            obj = world_state.entities[name]
            #wait_for_duration(1.0)
            print('Localizing', name)
            obj.localize() # Needed to ensure detectable
            print('Detecting', name)
            for _ in range(5):
                rospy.sleep(0.1)
                obj.administrator.detect()
                #obj.detect() # Actually applies the blue model
            #print(world_state.entities[name])
            #obj.administrator.detect()
            #print(obj.pose[:3, 3])
        rospy.sleep(5.0)
        print('Localized:', self.task.objects)
        # TODO: stop_localizing
        # TODO: wait until the variance in estimates is low