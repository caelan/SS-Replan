from pddlstream.language.constants import Not
from src.task import Task

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
            atom = ('On', obj, surface)
        elif predicate == 'is_free':
            atom = ('HandEmpty',)
        elif predicate == 'gripper_closed':
            assert value is False
            value = True
            atom = ('AtGConf', world.open_gq)
        elif predicate == 'cabinet_is_open':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'open')
        elif predicate == 'cabinet_is_closed':
            cabinet, = args
            joint_name = '{}_joint'.format(cabinet)
            atom = ('DoorStatus', joint_name, 'closed')
        elif predicate == 'in_drawer':
            obj, surface = args
            # TODO: ensure that it actually is a surface?
            init.append(('Stackable', obj, surface))
            atom = ('On', obj, surface)
        else:
            raise NotImplementedError(predicate)
        goal_literals.append(atom if value else Not(atom))
    task = Task(world, movable_base=not fixed, init=init, goal=goal_literals, **kwargs)
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
