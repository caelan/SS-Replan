#!/usr/bin/env python2

from __future__ import print_function

import rospy
import tf.transformations as tra
import numpy as np
import json
import os
import sys
import tf

sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d))
                for d in ['pddlstream', 'ss-pybullet'])

from pybullet_tools.utils import write_json, tform_from_pose
from src.issac import UNREAL_WORLD_FRAME, ISSAC_WORLD_FRAME, lookup_pose

from isaac_bridge.manager import corrections
from tf2_msgs.msg import TFMessage

POSES_PATH = 'kitchen_poses.json'

class TFRepublisher:
    objs = [
        "banana", "sugar_box", "mustard_bottle", "tomato_soup_can", "potted_meat_can",
        "shapenet_mug", "cracker_box",
        "big_block_blue", "small_block_blue",
        "big_block_yellow", "small_block_yellow",
        "big_block_red", "small_block_red",
        "big_block_green", "small_block_green",
    ]
    # ["table", "chassis_link", "zed_left", "zed_right",]

    def __init__(self):
        self.sub = rospy.Subscriber("/sim/tf", TFMessage, self.tf_callback)
        self.tf_listener = tf.TransformListener()
        self.transforms = {}
        self.updated = False

    def tf_callback(self, msg):
        self.sub.unregister()

        world_from_unreal = tform_from_pose(lookup_pose(self.tf_listener, UNREAL_WORLD_FRAME))
        for t in msg.transforms:
            if t.header.frame_id != ISSAC_WORLD_FRAME:
                continue
            t.header.frame_id = UNREAL_WORLD_FRAME
            # https://gitlab-master.nvidia.com/srl/srl_system/blob/fb94253c60b1bd1308a37c1aeb9dc4a4c453c512/packages/isaac_bridge/src/isaac_bridge/manager.py#L26
            frame = t.child_frame_id # TODO: prune frame suffix
            if any(obj in frame for obj in self.objs):
                continue
            # DO CORRECTION HERE

            # convert tf pose into homogeneous matrix
            obj_pose = tra.quaternion_matrix(
                [t.transform.rotation.x, t.transform.rotation.y,
                 t.transform.rotation.z, t.transform.rotation.w]
            )
            obj_pose[:3, 3] = [
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ]
            #correction = corrections.get(frame, np.eye(4)) # Causes the table to float?
            correction = np.eye(4)
            corrected_pose = obj_pose.dot(correction)
            world_pose = world_from_unreal.dot(corrected_pose)

            # convert back to message
            trans = tra.translation_from_matrix(world_pose)
            rot = tra.quaternion_from_matrix(world_pose)
            self.transforms[frame] = (list(trans), list(rot))
        self.updated = True

    def spin(self, rate=120):
        rate = rospy.Rate(rate)
        while not self.updated and not rospy.is_shutdown():
            rate.sleep()
        print(sorted(self.transforms))
        write_json(POSES_PATH, self.transforms)
        print('Saved', POSES_PATH)

if __name__ == "__main__":
    rospy.init_node('log_poses')
    t = TFRepublisher()
    t.spin()
