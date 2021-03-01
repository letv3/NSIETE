import numpy as np


def dataset_Circles(n=1000, radius=0.7, noise=0.0):
    X = np.zeros((n, 2, 1))
    Y = np.zeros((n, 1, 1))

    for currentN in range(n):
        i, j = 2 * np.random.rand(2) - 1

        r = np.sqrt(i ** 2 + j ** 2)
        if (noise > 0.0):
            r += np.random.rand() * noise

        if (r < radius):
            l = 0
        else:
            l = 1

        X[currentN, 0] = [i]
        X[currentN, 1] = [j]
        Y[currentN, 0] = [float(l)]

    return X, Y