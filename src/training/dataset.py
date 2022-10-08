from typing import Union, Tuple, Dict

import tensorflow as tf

from .generate_data import PDE_DATA


class Dataset:

    def __init__(self,
                 batch_size: str,
                 variables: Dict,
                 is_validation: bool = False):

        super().__init__()

        self.batch_size = batch_size
        self.is_validation = is_validation
        self.variables = variables

    def generate(self) -> Dict:

        batch = {}
        for k, v in self.variables.items():
            if isinstance(v, (float, int)):
                batch[k] = v
            else:
                batch[k] = tf.random.uniform(minval=v[0], maxval=v[1], shape=[self.batch_size, 1])

        return batch
