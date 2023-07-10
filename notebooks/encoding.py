"""
Helper functions to encode and decode string labels to integers
"""
import numpy as np


def encode(y):
    labels = y.unique()
    # encode label map
    code = {v: i for i, v in enumerate(labels)}

    # encode labels to new vector
    return np.array(list(map(lambda x: code[x], y))), code


def decode(y, code):
    keys = list(code.keys())
    values = list(code.values())
    return np.array(list(map(lambda x: keys[values.index(x)], y)))
