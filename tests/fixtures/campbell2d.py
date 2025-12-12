import numpy as np

def campbell2d(x, y, theta):
    """
    Modified Campbell2D function from Marrel et al. (2010)
    Parameterized by vector theta of length 8.
    """
    t1, t2, t3, t4, t5, t6, t7, t8 = theta

    return (
        t1 * np.exp(-(x - 1)**2 - (y + 2)**2 / 4)
        + t2 * np.exp(-(x + 1)**2 / 4 - (y - 1)**2)
        + t3 * np.exp(-(x - 2)**2 / 9 - (y - 2)**2 / 9)
        + t4 * (x + y)
        + t5 * x**2
        + t6 * y**2
        + t7 * x * y
        + t8
    )


DEFAULT_THETA = np.array([5, 3, 1, -1, 5, 3, 1, -1], dtype=float)
NEG_THETA = -1 * np.ones(8)
POS_THETA = 5 * np.ones(8)

