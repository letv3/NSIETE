import numpy as np


def dataset_Circles(batch_size=512, radius=0.7, noise=0.0):
    X = np.zeros((2, batch_size))
    Y = np.zeros((1, batch_size))

    for currentN in range(batch_size):
        i, j = 2 * np.random.rand(2) - 1

        r = np.sqrt(i ** 2 + j ** 2)
        if (noise > 0.0):
            r += np.random.rand() * noise

        if (r < radius):
            l = 0
        else:
            l = 1

        X[0, currentN] = i
        X[1, currentN] = j
        Y[0, currentN] = float(l)

    return X, Y