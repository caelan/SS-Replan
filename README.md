# SS-Replan

PDDLStream planning for NVIDIA SRL's Franka Panda Robot in a kitchen environment.

## Installation

<!--* `sudo apt install cmake g++ make python ros-kinetic-trac-ik`-->
* `pip install numpy scipy pybullet sklearn`
* `git lfs clone --branch master --recurse-submodules https://github.com/caelan/SS-Replan.git`
* `cd ss-replan`
* `./pddlstream/FastDownward/build.py release64`

<!--https://bitbucket.org/traclabs/trac_ik/src/master/-->

## PyBullet Examples

* `git pull --recurse-submodules`
* `./run_pybullet.py`

[<img src="images/stow_block.png" height="200">](https://drive.google.com/open?id=103NSqEeumZxFLbyrzAt6fxBiuz18_zTD)

[<img src="images/put_spam.png" height="200">](https://drive.google.com/open?id=1N6W_KZQOpNY2ZjIRtQURV_Chsnx3P75K)

## Resources

This repository uses PDDLStream to perform hybrid robotic planning. 
PDDLStream leverages FastDownward, a classical planner, as a discrete search subroutine.
Common robotics primitives are implemented using PyBullet.

### PDDLStream

* [PDDLStream Paper](https://arxiv.org/abs/1802.08705)
* [PDDLStream Github Repository](https://github.com/caelan/pddlstream)
* [PDDLStream Tutorial](https://web.mit.edu/caelan/www/presentations/6.881_TAMP.pdf)

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

## Citation

Caelan R. Garrett, Chris Paxton, Tomás Lozano-Pérez, Leslie P. Kaelbling, Dieter Fox. Online Replanning in Belief Space for Partially Observable Task and Motion Problems, IEEE International Conference on Robotics and Automation (ICRA), 2020.
