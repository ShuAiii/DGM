import os
from pathlib import Path
from typing import Union, Tuple
import random
import numpy as np

import tensorflow as tf

from .generate_data import PDE_DATA


class Dataset:

    def __init__(self, name: str, batch_size: str, directory: str,
                 variables: dict, data_kwargs: dict = None,
                 is_validation: bool = False):
        super().__init__()

        self.name = name
        self.batch_size = batch_size
        self.is_validation = is_validation
        self.directory = os.path.join(directory, "train")
        if self.is_validation:
            self.directory = os.path.join(directory, "validation")
        if data_kwargs:
            Path(self.directory).mkdir(parents=True, exist_ok=True)
            self.generate_date(data_kwargs["num_batches"], variables)

        self.batch_names = [filename for filename in os.listdir(self.directory)
                            if filename.endswith('.npy')]
        assert len(os.listdir(directory)) > 0, "Empty Directory"

    def generate_date(self, num_batches: int, variables: dict) -> None:
        _digits = 5

        generator = PDE_DATA[self.name]

        for idx in range(num_batches):
            npy = generator(self.batch_size, variables, self.is_validation)

            prefix = str(0).zfill(_digits)

            np_file_path = self.directory + "/" + str(self.batch_size) + '-' \
                           + prefix + '-' \
                           + str(idx).zfill(_digits) + '.npy'
            with open(np_file_path, 'wb') as f:
                np.save(f, npy)

    def __len__(self) -> int:
        return len(self.batch_names)

    def __getitem__(self, idx) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        file_path = os.path.join(self.directory, self.batch_names[idx])
        with open(file_path, 'rb') as f:
            arr = np.load(f)

        if idx == self.__len__() - 1 and not self.is_validation:
            random.shuffle(self.batch_names)

        arr_input = arr.copy()
        if not self.is_validation:
            return tf.convert_to_tensor(arr_input, np.float32)
        arr_input = arr[:, :-1].copy()
        arr_output = arr[:, -1:].copy()
        return tf.convert_to_tensor(arr_input, np.float32), \
               tf.convert_to_tensor(arr_output, np.float32)
