import pdb
from typing import Union, Tuple
import tensorflow as tf
import numpy as np
from scipy.stats import norm


def bsm_data(batch_size: int,
             variables: dict,
             is_validation: bool = False) -> np.array:

    batch = {}
    for k, v in variables.items():
        if isinstance(v, (float, int)):
            batch[k] = v
        else:
            batch[k] = tf.random.uniform(minval=v[0], maxval=v[1], shape=[batch_size, 1])

    return batch


PDE_DATA = {
    "bsm_call": bsm_data
}
