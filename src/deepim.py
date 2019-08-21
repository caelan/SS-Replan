from __future__ import print_function

from collections import defaultdict
from itertools import product, combinations

import numpy as np
import rospy
import tf

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from brain_ros.ros_world_state import make_pose_from_pose_msg
from lula_dartpy.object_administrator import ObjectAdministrator

from pybullet_tools.utils import INF, pose_from_tform, point_from_pose, get_distance, quat_from_pose, \
    quat_angle_between
from src.issac import ISSAC_WORLD_FRAME, CAMERA_PREFIX
from src.parse_brain import SUGAR, YCB_OBJECTS, SPAM

PREFIX_TEMPLATE = '{:02d}'
PREFIX_FROM_SIDE = {
    'right': PREFIX_TEMPLATE.format(0),
    'left': PREFIX_TEMPLATE.format(1),
}
KINECT_TEMPLATE = 'kinect{}'
KINECT_FROM_SIDE = {
    'right': KINECT_TEMPLATE.format(1), # indexes from 1!
    'left': KINECT_TEMPLATE.format(2),
}
#DEEPIM_POSE_TEMPLATE = '/deepim/raw/objects/prior_pose/{}_{}' # ['kinect1_depth_optical_frame']
DEEPIM_POSE_TEMPLATE = '/objects/prior_pose/{}_{}' # ['kinect1_depth_optical_frame', 'depth_camera']
POSECNN_POSE_TEMPLATE = '/objects/prior_pose/{}_{}/decayable_weight'

RIGHT = 'right'
LEFT = 'left'
SIDES = [RIGHT, LEFT]

# Detection time
# min 0.292s max: 0.570s
DETECTIONS_PER_SEC = 0.6 # /deepim/raw/objects/prior_pose
#DETECTIONS_PER_SEC = 2.5 # /objects/prior_pose/

# https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/lula_dart/lula_dartpy/object_administrator.py

# TODO: it looks like DeepIM publishes each pose individually

################################################################################

# TODO: use the confidences that Chris added

class DeepIM(object):
    def __init__(self, domain, sides=[], obj_types=[]):
        self.domain = domain
        self.sides = tuple(sides)
        self.obj_types = tuple(obj_types)
        self.tf_listener = tf.TransformListener()

        self.subscribers = {}
        self.observations = defaultdict(list)
        for side, obj_type in product(self.sides, self.obj_types):
            prefix = PREFIX_FROM_SIDE[side]
            topic = DEEPIM_POSE_TEMPLATE.format(prefix, obj_type)
            #print('Starting', topic)
            cb = lambda data, s=side, ty=obj_type: self.callback(data, s, ty)
            self.subscribers[side, obj_type] = rospy.Subscriber(
                topic, PoseStamped, cb, queue_size=1)
        # TODO: use the pose_fixer topic to send DART a prior
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/b38a70fda63f5556bcba2ccb94eca54124e40b65/packages/lula_dart/lula_dartpy/pose_fixer.py#L4
    def callback(self, pose_stamped, side, obj_type):
        #print('Received {} camera detection of {}'.format(side, obj_type))
        if not pose_stamped.header.frame_id.startswith(CAMERA_PREFIX):
            return
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
        entity = self.domain.root.entities[obj_type]
        entity.stop_localizing()
        #administrator = entity.administrator
        #administrator.deactivate()
    def detect(self, obj_type):
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/lula_dart/lula_dartpy/object_administrator.py
        # https://gitlab-master.nvidia.com/srl/srl_system/blob/master/packages/brain/src/brain_ros/ros_world_state.py
        #self.domain.entities
        #self.domain.view_tags # {'right': '00', 'left': '01'}
        #dump_dict(self.domain)
        #dump_dict(self.domain.root)
        #dump_dict(entity)
        entity = self.domain.root.entities[obj_type]
        administrator = entity.administrator

        duration = 5.0
        expected_detections = duration * DETECTIONS_PER_SEC
        observations = self.get_recent_observations(RIGHT, obj_type, duration)
        print('{}) observations={}, duration={}, rate={}'.format(
            obj_type, len(observations), duration, len(observations) / duration))
        if len(observations) < 0.5*expected_detections:
            return False
        detections_from_frame = {}
        for pose_stamped in observations:
            detections_from_frame.setdefault(pose_stamped.header.frame_id, []).append(pose_stamped)
        #print('Frames:', detections_from_frame.keys())
        assert len(detections_from_frame) == 1
        poses = [pose_from_tform(make_pose_from_pose_msg(pose_stamped)) for pose_stamped in observations]
        points = [point_from_pose(pose) for pose in poses]
        pos_deviation = np.mean([get_distance(*pair) for pair in combinations(points, r=2)])
        quats = [quat_from_pose(pose) for pose in poses]
        ori_deviation = np.mean([quat_angle_between(*pair) for pair in combinations(quats, r=2)])
        print('{}) position deviation: {:.3f} meters | orientation deviation: {:.3f} degrees'.format(
            obj_type, pos_deviation, np.math.degrees(ori_deviation)))
        # TODO: symmetries
        # TODO: prune if not on surface
        # TODO: prune if incorrect orientation
        # TODO: small sleep after each detection to ensure time to converge
        # TODO: wait until DART convergence

        administrator.activate() # entity.localize()
        #entity.set_tracked() # entity.is_tracked = True # Doesn't do anything
        #entity.detect() # administrator.detect_once()
        administrator.detect_and_wait()
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

################################################################################

def detect_classes():

    cv_bridge = CvBridge()
    #config_data = read_json(PANDA_FULL_CONFIG_PATH)
    #camera_data = config_data['LeftCamera']['CameraComponent']
    #segmentation_labels = [d['name'] for d in camera_data['segmentation_classes']['static_mesh']]
    #print('Labels:', segmentation_labels)

    detections = []
    def callback(data):
        segmentation = cv_bridge.imgmsg_to_cv2(data)
        #frequency = Counter(segmentation.flatten().tolist()) # TODO: use the area
        #print(frequency)
        indices = np.unique(segmentation)
        #print(indices)
        #detections.append({segmentation_labels[i-1] for i in indices}) # wraps around [-1]
        #subscriber.unregister()

    # DeepIM trained on bowl, cracker_box, holiday_cup1, holiday_cup2, mustard_bottle
    # potted_meat_can, sugar_box, tomato_soup_can
    side = 'right'
    prefix = PREFIX_FROM_SIDE[side]
    obj_type = SUGAR

    # kinect from side
    # kinect1_depth_optical_frame | kinect2_depth_optical_frame
    DEEPIM_POSE_TOPIC = DEEPIM_POSE_TEMPLATE.format(prefix, obj_type)
    pose_subscriber = rospy.Subscriber(DEEPIM_POSE_TOPIC, PoseStamped, callback, queue_size=1)
    # https://gitlab-master.nvidia.com/srl/srl_system/blob/b38a70fda63f5556bcba2ccb94eca54124e40b65/packages/lula_dart/lula_dartpy/pose_fixer.py

    # All of these are images
    POSECNN_LABEL_TOPIC = '/posecnn_label_{}'.format(side)
    POSECNN_POSE_TOPIC = '/posecnn_pose_{}'.format(side)
    DEEPIM_IMAGE_TOPIC = '/deepim_pose_image_{}'.format(side)
    image_topic = DEEPIM_IMAGE_TOPIC

    rospy.sleep(0.1) # This sleep is needed
    image_subscriber = rospy.Subscriber(image_topic, Image, callback, queue_size=1)
    while not detections:
        rospy.sleep(0.01)
    print('Detections:', detections[-1])
    return detections[-1]

################################################################################

def test_deepim(domain):
    # # https://gitlab-master.nvidia.com/srl/srl_system/blob/c5747181a24319ed1905029df6ebf49b54f1c803/packages/lula_dart/lula_dartpy/object_administrator.py
    deepim = DeepIM(domain, sides=[RIGHT], obj_types=YCB_OBJECTS)

    side = 'right'
    prefix = PREFIX_FROM_SIDE[side]
    # obj_type = SUGAR
    obj_type = SPAM
    base_frame = '{}_{}'.format(prefix, obj_type)
    administrator = ObjectAdministrator(
        base_frame, wait_for_connection=True)  # wait_for_connection=False
    print(administrator.is_active, administrator.is_detecting)

    rospy.sleep(2.0)
    deepim.detect(SUGAR)
    deepim.detect(SPAM)

    rospy.spin()
    return

    rate = rospy.Rate(1000)
    while INF <= deepim.last_detected(side, obj_type):
        rate.sleep()

    # print(deepim.last_world_pose(side, obj_type))

    # Could redetect on every step
    print('Detected', obj_type)
    # Doesn't look like the order matters actually
    administrator.activate()  # localize
    administrator.detect_once()  # detect
    # administrator.detect_and_wait()
    # administrator.wait_for_detection_complete()
    # administrator.deactivate() # stop_localizing

    # rospy.sleep(10)
    # TODO: test how far away from deepim's estimate
    # Redetect for a fixed number of times until close
    # print('Redetecting', obj_type)
    # administrator.detect_once() # Every redetect causes the objects to spaz

    rospy.sleep(5)
    # print('Finished detecting', obj_type)
    # administrator.deactivate() # stop_localizing

    # TODO: if orientation is bad and make not manipulable

    rospy.spin()