from __future__ import print_function

import math
import time

from collections import defaultdict
from itertools import product, combinations

import numpy as np
import rospy
import tf
from cv_bridge import CvBridge

#from posecnn_pytorch_msgs.msg import DetectionList
from geometry_msgs.msg import PoseStamped
from brain_ros.ros_world_state import make_pose_from_pose_msg

from pybullet_tools.utils import INF, pose_from_tform, point_from_pose, get_distance, quat_from_pose, \
    quat_angle_between, read_json, elapsed_time
from sensor_msgs.msg import Image
from src.issac import ISSAC_WORLD_FRAME, DEPTH_PREFIX, PANDA_FULL_CONFIG_PATH, PREFIX_FROM_SIDE, RIGHT, lookup_pose

#DEEPIM_POSE_TEMPLATE = '/deepim/raw/objects/prior_pose/{}_{}' # ['kinect1_depth_optical_frame']
DEEPIM_POSE_TEMPLATE = '/objects/prior_pose/{}_{}' # ['kinect1_depth_optical_frame', 'depth_camera']

POSECNN_POSE_TEMPLATE = '/posecnn/{}/info' # /posecnn/00/info # posecnn_pytorch/DetectionList
#POSECNN_POSE_TEMPLATE = '/objects/prior_pose/{}_{}/decayable_weight'

# Detection time
# min 0.292s max: 0.570s
DETECTIONS_PER_SEC = 0.6 # /deepim/raw/objects/prior_pose
#DETECTIONS_PER_SEC = 2.5 # /objects/prior_pose/

# https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/lula_dart/lula_dartpy/object_administrator.py

def get_pose_distance(pose1, pose2):
    translation = get_distance(point_from_pose(pose1), point_from_pose(pose2))
    rotation = quat_angle_between(quat_from_pose(pose1), quat_from_pose(pose2))
    return translation, rotation

def mean_pose_deviation(poses):
    points = [point_from_pose(pose) for pose in poses]
    pos_deviation = np.mean([get_distance(*pair) for pair in combinations(points, r=2)])
    quats = [quat_from_pose(pose) for pose in poses]
    ori_deviation = np.mean([quat_angle_between(*pair) for pair in combinations(quats, r=2)])
    return pos_deviation, ori_deviation

def wait_until_frames_stabilize(interface, frames, min_updates=10, timeout=10.0,
                                pos_threshold=0.005, ori_threshold=math.radians(1)):
    start_time = time.time()
    history = defaultdict(list)
    num_updates = 0
    while elapsed_time(start_time) < timeout:
        num_updates += 1
        print('Update: {} | Frames: {} | Time: {}'.format(num_updates, frames, elapsed_time(start_time)))
        success = True
        for frame in frames:
            history[frame].append(lookup_pose(interface.observer.tf_listerner, frame))
            if min_updates <= len(history[frame]):
                pos_deviation, ori_deviation = mean_pose_deviation(history[frame][-min_updates:])
                print('{}) position deviation: {:.3f} meters | orientation deviation: {:.3f} degrees'.format(
                    frame, pos_deviation, math.degrees(ori_deviation)))
                success &= (pos_deviation <= pos_threshold) and (ori_deviation <= math.radians(ori_threshold))
            else:
                success = False
        if success:
            return True
        rospy.sleep(0.001) # TODO: multiply with min_updates
    # TODO: return last observation?
    return False

################################################################################

class Perception(object):
    def __init__(self, domain):
        self.domain = domain
    def detect_all(self, obj_types=None):
        raise NotImplementedError()
    def stop_tracking(self, name):
        return True

class FullObserver(Perception):
    def __init__(self, domain):
        super(FullObserver, self).__init__(domain)
    @property
    def obj_types(self):
        return set(self.domain.entities)
        #return set(self.domain.root.entities)
    def detect_all(self, obj_types=None):
        return self.obj_types

################################################################################

#SEGMENTATION_TOPIC = '/sim/left_segmentation_camera/instance_image' # {0, 1}
SEGMENTATION_TOPIC = '/sim/left_segmentation_camera/label_image' # {0, ..., 13}

class Segmentator(Perception):
    def __init__(self, domain):
        super(Segmentator, self).__init__(domain)
        # from brain.scripts.logger import Logger
        # logger = Logger()
        self.cv_bridge = CvBridge()
        config_data = read_json(PANDA_FULL_CONFIG_PATH)
        camera_data = config_data['LeftCamera']['CameraComponent']
        self.segmentation_labels = [d['name'] for d in camera_data['segmentation_classes']['static_mesh']]
        print('Labels:', self.segmentation_labels)
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/a1255229910a30f9c510bad1c4719c1c59c7b8ec/packages/isaac_bridge/configs/panda_full_config.json#L411
        # python packages/brain/scripts/segmentation_visualizer.py

        self.detections = []
        self.subscriber = rospy.Subscriber(SEGMENTATION_TOPIC, Image, self.callback, queue_size=1)
    def callback(self, data):
        segmentation = self.cv_bridge.imgmsg_to_cv2(data)
        # frequency = Counter(segmentation.flatten().tolist()) # TODO: use the area
        # print(frequency)
        indices = np.unique(segmentation)
        # print(indices)
        self.detections.append({self.segmentation_labels[i - 1] for i in indices})  # wraps around [-1]
        #self.subscriber.unregister()
    def detect_all(self, obj_types=None):
        #rospy.sleep(0.1)  # This sleep is needed
        while not self.detections:
            rospy.sleep(0.01)
        print('Detections:', self.detections[-1])
        return self.detections[-1]

################################################################################

class DeepIM(Perception):
    def __init__(self, domain, sides=[], obj_types=[]):
        super(DeepIM, self).__init__(domain)
        self.sides = tuple(sides)
        self.obj_types = tuple(obj_types)
        self.tf_listener = tf.TransformListener()
        self.observations = defaultdict(list)
        self.tracked = set()
        # TODO: finish using the confidences that Chris added
        self.posecnn_subscriber = {}
        #for side in self.sides:
        #    prefix = PREFIX_FROM_SIDE[side]
        #    topic = POSECNN_POSE_TEMPLATE.format(prefix)
        #    cb = lambda data, s=side: self.posecnn_callback(data, s)
        #    self.posecnn_subscriber[side] = rospy.Subscriber(
        #        topic, DetectionList, cb, queue_size=1)
        self.subscribers = {}
        for side, obj_type in product(self.sides, self.obj_types):
            prefix = PREFIX_FROM_SIDE[side]
            topic = DEEPIM_POSE_TEMPLATE.format(prefix, obj_type)
            #print('Starting', topic)
            cb = lambda data, s=side, ty=obj_type: self.callback(data, s, ty)
            self.subscribers[side, obj_type] = rospy.Subscriber(
                topic, PoseStamped, cb, queue_size=1)
        # TODO: use the pose_fixer topic to send DART a prior
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/b38a70fda63f5556bcba2ccb94eca54124e40b65/packages/lula_dart/lula_dartpy/pose_fixer.py#L4
    def posecnn_callback(self, data, side):
        print(data)
    def callback(self, pose_stamped, side, obj_type):
        #print('Received {} camera detection of {}'.format(side, obj_type))
        #if not pose_stamped.header.frame_id.startswith(DEPTH_PREFIX):
        #    return
        self.observations[side, obj_type].append(pose_stamped)
    def last_detected(self, side, obj_type):
        if not self.observations[side, obj_type]:
            return INF
        pose_stamped = self.observations[side, obj_type][-1]
        current_time = rospy.Time.now() # rospy.get_rostime()
        # TODO: could call detect with every new observation
        return (current_time - pose_stamped.header.stamp).to_sec()
    def get_recent_observations(self, side, obj_type, duration):
        detections = []
        current_time = rospy.Time.now() # rospy.get_rostime()
        for pose_stamped in self.observations[side, obj_type][::-1]:
            time_passed = (current_time - pose_stamped.header.stamp).to_sec()
            if duration < time_passed:
                break
            detections.append(pose_stamped)
        return detections
    def last_world_pose(self, side, obj_type):
        if not self.observations[side, obj_type]:
            return None
        # TODO: search over orientations
        pose_kinect = self.observations[side, obj_type][-1]
        tf_pose = self.tf_listener.transformPose(ISSAC_WORLD_FRAME, pose_kinect)
        return pose_from_tform(make_pose_from_pose_msg(tf_pose))
    def stop_tracking(self, obj_type):
        # Nothing happens if already not tracked
        self.tracked.discard(obj_type)
        entity = self.domain.root.entities[obj_type]
        entity.stop_localizing()
        #administrator = entity.administrator
        #administrator.deactivate()
    def detect(self, obj_type):
        if obj_type in self.tracked:
            return True
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/lula_dart/lula_dartpy/object_administrator.py
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py
        #self.domain.entities
        #dump_dict(self.domain)
        #dump_dict(self.domain.root)
        #dump_dict(entity)
        entity = self.domain.root.entities[obj_type]
        administrator = entity.administrator

        duration = 5.0
        min_fraction = 0.25
        expected_detections = duration * DETECTIONS_PER_SEC
        observations = self.get_recent_observations(RIGHT, obj_type, duration)
        print('{}) observations={}, duration={}, rate={}'.format(
            obj_type, len(observations), duration, len(observations) / duration))
        if len(observations) < min_fraction*expected_detections:
            return False
        detections_from_frame = {}
        for pose_stamped in observations:
            detections_from_frame.setdefault(pose_stamped.header.frame_id, []).append(pose_stamped)
        #print('Frames:', detections_from_frame.keys())
        assert len(detections_from_frame) == 1
        poses = [pose_from_tform(make_pose_from_pose_msg(pose_stamped)) for pose_stamped in observations]
        pos_deviation, ori_deviation = mean_pose_deviation(poses)
        print('{}) position deviation: {:.3f} meters | orientation deviation: {:.3f} degrees'.format(
            obj_type, pos_deviation, np.math.degrees(ori_deviation)))
        # TODO: symmetries
        # TODO: prune if not on surface
        # TODO: prune if incorrect orientation
        # TODO: small sleep after each detection to ensure time to converge

        self.tracked.add(obj_type)
        administrator.activate() # entity.localize()
        #entity.set_tracked() # entity.is_tracked = True # Doesn't do anything
        #administrator.detect_and_wait(wait_time=2.0)
        entity.detect() # administrator.detect_once()
        #for _ in range(1):
        #    rospy.sleep(DETECTIONS_PER_SEC)
        #    administrator.detect()
        #administrator.is_active
        #administrator.is_detecting

        #entity.is_tracked False
        #entity.last_clock None
        #entity.last_t None
        #entity.location_belief None
        return True
    def detect_all(self, obj_types=None):
        if obj_types is None:
            obj_types = self.obj_types
        rospy.sleep(1.0)
        return {obj_type for obj_type in obj_types if self.detect(obj_type)}
