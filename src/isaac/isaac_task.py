from __future__ import print_function

from brain_ros.sim_test_tools import TrialManager

import numpy as np

from pddlstream.language.constants import Not
from pddlstream.utils import Verbose
from src.isaac.deepim import FullObserver, Segmentator
from src.isaac.interface import Interface
from src.task import Task, sample_placement
from src.problem import door_closed_formula, door_open_formula
from examples.discrete_belief.dist import DeltaDist, UniformDist
from src.isaac.update_isaac import update_isaac_sim
from src.isaac.issac import NULL_POSE, set_pose
from src.utils import SPAM, SUGAR, CHEEZIT, INDIGO_COUNTER, TOP_DRAWER, \
    BOTTOM_DRAWER

TRIAL_MANAGER_TASKS = [
    'open_bottom', 'open_top', 'pick_spam',
    'put_away', # tomato_soup_can
    'put_spam',
]


################################################################################

# cage_handle_from_drawer = ([0.28, 0.0, 0.0], [0.533, -0.479, -0.501, 0.485])

def task_from_trial_manager(world, trial_manager, task_name, fixed=False, **kwargs):
    assert task_name in TRIAL_MANAGER_TASKS
    with Verbose(False):
        task_name = task_name.replace('_', ' ')
        objects, goal, plan = trial_manager.get_task(task=task_name, reset=True)
    trial_goals = [(h.format(o), v) for h, v in goal for o in objects]
    print('Objects:', objects)
    print('Goals:', trial_goals)
    #regex = re.compile(r"(\w+)\((\)\n")

    # TODO: use the task plan to constrain solution
    init = []
    goal_literals = []
    for head, value in trial_goals:
        predicate, arguments = head.strip(')').split('(')
        args = [arg.strip() for arg in arguments.split(',')]
        if predicate == 'on_counter':
            obj, = args
            surface = INDIGO_COUNTER
            formula = ('On', obj, surface)
        elif predicate == 'is_free':
            formula = ('HandEmpty',)
        elif predicate == 'gripper_closed':
            assert value is False
            value = True
            formula = ('AtGConf', world.open_gq)
        elif predicate == 'gripper_open':
            assert value is True
            formula = ('AtGConf', world.open_gq)
        elif predicate == 'cabinet_is_open':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            #formula = ('DoorStatus', joint_name, 'open')
            formula = door_open_formula(joint_name)
        elif predicate == 'cabinet_is_closed':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            #formula = ('DoorStatus', joint_name, 'closed')
            formula = door_closed_formula(joint_name)
        elif predicate == 'in_drawer':
            obj, surface = args
            # TODO: ensure that it actually is a surface?
            init.append(('Stackable', obj, surface))
            formula = ('On', obj, surface)
        else:
            raise NotImplementedError(predicate)
        goal_literals.append(formula if value else Not(formula))
    prior = {
        SPAM: DeltaDist(INDIGO_COUNTER),
        CHEEZIT: DeltaDist(INDIGO_COUNTER),
        #MUSTARD: DeltaDist(ECHO_COUNTER),
        #TOMATO_SOUP: DeltaDist(ECHO_COUNTER),
        #SUGAR: DeltaDist(ECHO_COUNTER),
    }
    return Task(world, prior=prior, movable_base=not fixed, init=init, goal=goal_literals,
                return_init_bq=True, return_init_aq=True, real=True, **kwargs)

################################################################################

def create_trial_args(**kwargs):
    args = lambda: None # Dummy class
    args.side = 'right'
    args.drawer = 'top'
    args.script_timeout = None
    args.no_planning = True
    args.debug_planner = False
    args.pause = False
    args.image = 'img%02d.png'
    args.max_count = 999999
    args.disrupt = False
    args.linear = False
    args.replan = False
    args.seed = None
    args.image_topic = '/sim/left_color_camera/image'
    args.iter = 1
    args.max_t = 3*60
    args.randomize_textures = 0.
    args.randomize_camera = 0.
    args.sigma = 0.
    args.p_sample = 0.
    args.lula_collisions = False
    args.babble = False
    # TODO: use setattr
    return args

################################################################################

def set_isaac_sim(interface):
    assert interface.simulation
    task = interface.task
    world = task.world
    #close_all_doors(world)
    if task.movable_base:
        world.set_base_conf([2.0, 0, -np.pi / 2])
        # world.set_initial_conf()
    else:
        for name, dist in task.prior.items():
            if name in task.prior:
                surface = task.prior[name].sample()
                sample_placement(world, name, surface, learned=True)
            else:
                set_pose(world.get_body(name), NULL_POSE)
                #sample_placement(world, name, ECHO_COUNTER, learned=False)
        # pose2d_on_surface(world, SPAM, INDIGO_COUNTER, pose2d=SPAM_POSE2D)
        # pose2d_on_surface(world, CHEEZIT, INDIGO_COUNTER, pose2d=CRACKER_POSE2D)
    update_isaac_sim(interface, world)
    # wait_for_user()

################################################################################

def simulation_setup(domain, world, args):
    # TODO: forcibly reset robot arm configuration
    # trial_args = parse.parse_kitchen_args()
    trial_args = create_trial_args()
    with Verbose(False):
        trial_manager = TrialManager(trial_args, domain, lula=args.lula)
    observer = trial_manager.observer
    #set_isaac_camera(trial_manager.sim_manager)
    trial_manager.set_camera(randomize=False)

    task_name = args.problem
    if task_name in TRIAL_MANAGER_TASKS:
        task = task_from_trial_manager(world, trial_manager, task_name, fixed=args.fixed)
    else:
        prior = {
            SPAM: UniformDist([TOP_DRAWER, BOTTOM_DRAWER]),
            #SPAM: UniformDist([INDIGO_COUNTER]),
            SUGAR: UniformDist([INDIGO_COUNTER]),
            CHEEZIT: UniformDist([INDIGO_COUNTER]),
        }
        goal_drawer = TOP_DRAWER  # TOP_DRAWER | BOTTOM_DRAWER
        task = Task(world, prior=prior, teleport_base=True,
                    # goal_detected=[SPAM],
                    #goal_holding=SPAM,
                    #goal_on={SPAM: goal_drawer},
                    # goal_closed=[],
                    # goal_closed=[JOINT_TEMPLATE.format(goal_drawer)],
                    # goal_open=[JOINT_TEMPLATE.format(goal_drawer)],
                    movable_base=not args.fixed,
                    goal_aq=world.carry_conf,  # .values,
                    # return_init_aq=True,
                    return_init_bq=True)

    perception = FullObserver(domain) if args.observable else Segmentator(domain)
    interface = Interface(args, task, observer, trial_manager=trial_manager, deepim=perception)
    if args.jump:
        robot_entity = domain.get_robot()
        robot_entity.carter_interface = interface.sim_manager
    return interface