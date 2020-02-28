# SS-Replan

Online observation, estimatation, planning, and control for a Franka Panda Robot operating in [NVIDIA SRL](https://www.nvidia.com/en-us/research/robotics/)'s simulated kitchen environment.

## Installation

SS-Replan supports both Python 2 and Python 3.

<!--* `sudo apt install cmake g++ make python ros-kinetic-trac-ik`-->
* `$ pip install numpy scipy pybullet sklearn`
* `$ git clone --branch master --recurse-submodules https://github.com/caelan/SS-Replan.git`
* `$ cd SS-Replan`
* `SS-Replan$ ./pddlstream/FastDownward/build.py release64`
* `SS-Replan$ cd ss-pybullet/pybullet_tools/ikfast/franka_panda`
* `SS-Replan/ss-pybullet/pybullet_tools/ikfast/franka_panda$ ./setup.py`

It's also possible to use [TRAC-IK](http://wiki.ros.org/trac_ik) instead of [IKFast](http://openrave.org/docs/0.8.2/openravepy/ikfast/); however it requires installing ROS (`$ sudo apt install ros-kinetic-trac-ik`).

<!--https://bitbucket.org/traclabs/trac_ik/src/master/-->

## PyBullet Examples

* `git pull --recurse-submodules`
* `./run_pybullet.py [-h]`

[<img src="https://img.youtube.com/vi/TvZqMDBZEnc/0.jpg" height="250">](https://youtu.be/TvZqMDBZEnc)

<!--&emsp;-->

## IsaacSim Examples

Executed using the [IssacSim](https://developer.nvidia.com/isaac-sim) 3D robot simulation environment.

[<img src="https://img.youtube.com/vi/XSZbCp0M1rw/0.jpg" height="250">](https://youtu.be/XSZbCp0M1rw)

## Real-World Examples

[<img src="https://img.youtube.com/vi/-Jl6GtvtWb8/0.jpg" height="250">](https://youtu.be/-Jl6GtvtWb8)

<!-- https://developer.nvidia.com/isaac-sdk -->

## Resources

* This repository uses [PDDLStream](https://github.com/caelan/pddlstream) to perform hybrid robotic planning. 
* PDDLStream leverages [FastDownward](http://www.fast-downward.org/), a classical planner, as a discrete search subroutine.
* Common robotics primitives are implemented using [PyBullet](https://pypi.org/project/pybullet/).

### PDDLStream

<!-- * [SS-Replan Paper](https://arxiv.org/abs/1911.04577) -->
* [PDDLStream Paper](https://arxiv.org/abs/1802.08705)
* [PDDLStream Github Repository](https://github.com/caelan/pddlstream)
* [PDDLStream Tutorial](http://web.mit.edu/caelan/www/presentations/6.881_19-11-12.pdf)

### PDDL and FastDownward

* [Planning Domain Definition Language (PDDL)](http://users.cecs.anu.edu.au/~patrik/pddlman/writing.html)
* [PDDL Derived Predicates](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume28/coles07a-html/node18.html)
* [FastDownward Homepage](http://www.fast-downward.org/)

### PyBullet

* [PyBullet Package](https://pypi.org/project/pybullet/)
* [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)

<!--# Resources

Please email Caelan Garrett at <caelan@mit.edu> for installation and usage help.-->

## Publications

* [Online Replanning in Belief Space for Partially Observable Task and Motion Problems](https://arxiv.org/abs/1911.04577)
* [PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning](https://arxiv.org/abs/1802.08705)

## Videos

* [SS-Replan](https://www.youtube.com/watch?v=o_RW91sm9PU&list=PLNpZKR7uv5ARTi1sNQRcd5rpa8XxamW2l)
* [PDDLStream](https://www.youtube.com/playlist?list=PLNpZKR7uv5AQIyT6Az31a3WqiXyQJX7Rx)

## Citation

Caelan R. Garrett, Chris Paxton, Tomás Lozano-Pérez, Leslie P. Kaelbling, Dieter Fox. Online Replanning in Belief Space for Partially Observable Task and Motion Problems, IEEE International Conference on Robotics and Automation (ICRA), 2020.
