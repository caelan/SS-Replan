#!/usr/bin/env python

import os
import rospy
import signal

import brain_ros.kitchen_domain as kitchen_domain
import brain_ros.parse as parse

from brain_ros.sim_test_tools import TrialManager

if __name__ == '__main__':
    rospy.init_node("test_grasping")
    domain = kitchen_domain.KitchenDomain()
    args = parse.parse_kitchen_args()
    man = TrialManager(args, domain)

    if args.iter <= 0:
        while not rospy.is_shutdown():
            man.do_random_trial(task="put away", reset=True)
    else:
        for i in range(args.iter):
            man.do_random_trial(task="put away", reset=(i == 0))
    # Kill Lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
