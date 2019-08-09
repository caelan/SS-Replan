from pddlstream.language.constants import Not
from src.task import Task
from src.problem import door_closed_formula, door_open_formula

TASKS = [
    'open_bottom', 'open_top', 'pick_spam',
    'put_away', # tomato_soup_can
    'put_spam',
]

# cage_handle_from_drawer = ([0.28, 0.0, 0.0], [0.533, -0.479, -0.501, 0.485])

def task_from_trial_manager(world, trial_manager, task_name, fixed=False, **kwargs):
    objects, goal, plan = trial_manager.get_task(task=task_name, reset=True)
    trial_goals = [(h.format(o), v) for h, v in goal for o in objects]
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
            surface = 'indigo_tmp'
            formula = ('On', obj, surface)
        elif predicate == 'is_free':
            formula = ('HandEmpty',)
        elif predicate == 'gripper_closed':
            assert value is False
            value = True
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
    task = Task(world, movable_base=not fixed, init=init, goal=goal_literals,
                return_init_bq=False, return_init_aq=False, **kwargs)
    return task

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
