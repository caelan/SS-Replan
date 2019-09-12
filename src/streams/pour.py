
def get_fixed_pour_gen_fn(world, max_attempts=25, collisions=True, teleport=False, **kwargs):
    def gen(bowl, wp, cup, g, bq):
        yield None
    return gen
