"""Smoke test to verify environment setup, uv dependencies, and basic imports."""


def test_imports():
    import cv2
    import matplotlib
    import numpy as np
    import PIL
    import scipy

    assert np is not None
    assert scipy is not None
    assert cv2 is not None
    assert PIL is not None
    assert matplotlib is not None


def test_numpy_operations():
    import numpy as np

    a = np.array([1, 2, 3, 4], dtype=np.float32)
    assert a.sum() == 10.0
