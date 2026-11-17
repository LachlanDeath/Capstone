import numpy as np

def power(x):
    """Mean power of a window: (1/N) * sum(x[j] * x[j])"""
    return np.dot(x, x) / len(x)