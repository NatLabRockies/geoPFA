import numpy as np

def test_tolerance_computation():
    y_true = np.array([1,2,3])
    y_pred = np.array([1.1, 2.1, 2.9])

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    r2 = 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()

    assert rmse < 0.2
    assert r2 > 0.9

