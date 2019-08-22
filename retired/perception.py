from __future__ import print_function

import numpy as np
import rospy

from src.deepim import PREFIX_FROM_SIDE, DEEPIM_POSE_TEMPLATE, DeepIM, RIGHT
from geometry_msgs.msg import PoseStamped
from lula_dartpy.object_administrator import ObjectAdministrator
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from pybullet_tools.utils import INF
from src.parse_brain import SUGAR, YCB_OBJECTS, SPAM

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