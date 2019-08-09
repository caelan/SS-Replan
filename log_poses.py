#!/usr/bin/env python2

from __future__ import print_function

import copy
import rospy
import tf.transformations as tra
import numpy as np
import json

from isaac_bridge.manager import corrections
from tf2_msgs.msg import TFMessage


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
        self.transforms = None

    def tf_callback(self, msg):
        self.sub.unregister()
        self.transforms = {}
        for t in msg.transforms:
            if t.header.frame_id != "world": # ue_world
                continue
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
            correction = corrections.get(frame, np.eye(4))
            corrected_pose = obj_pose.dot(correction)

            # convert back to message
            rot = tra.quaternion_from_matrix(corrected_pose)
            trans = tra.translation_from_matrix(corrected_pose)

            self.transforms[frame] = (list(rot), list(trans))

    def spin(self, rate=120):
        rate = rospy.Rate(rate)
        while (self.transforms is None) and not rospy.is_shutdown():
            rate.sleep()
        path = 'kitchen_poses.json'
        with open(path, 'w') as f:
            json.dump(self.transforms, f, indent=2, sort_keys=True)
        print('Saved', path)

if __name__ == "__main__":
    rospy.init_node('log_poses')
    t = TFRepublisher()
    t.spin()
