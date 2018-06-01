import numpy as np

def random_init(ai, shape):
    x,y = shape
    z = ai.shape[2]
    init_b = np.random.randint(ai.shape[0] * ai.shape[1], size=(x,y))

    list = np.reshape(ai, (-1, z))
    color_init_b = np.reshape(init_b, (x*y)).tolist()
    color_init_b = [list[p] for p in color_init_b]
    color_init_b = np.reshape(np.array(color_init_b), (x,y,z))
    return color_init_b, init_b
