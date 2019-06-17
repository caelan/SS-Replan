# Installation

* `sudo apt install cmake g++ mercurial make python`
* `pip install numpy scipy pybullet`
* `git lfs clone ssh://git@gitlab-master.nvidia.com:12051/cgarrett/srl-stripstream.git`
* `cd srl-stripstream`
* `git submodule update --init --recursive`
* `./pddlstream/FastDownward/build.py release64`

# Execution

`./run.py`
