""""""
import os
from typing import Dict
import pdb

import yaml
import click
import numpy as np
import tensorflow as tf

from src.training.nets import get_model


def parse_params(params: dict):
    return params


@click.command()
@click.option('-p', '--path', required=True, type=str, help="Path of the training conf file",
              default="/home/app_user/app/conf/black_scholes.yml")
def launch_train(path: str) -> None:

    with open(path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = parse_params(params)

    root_dir = params["training"]["root_dir"]
    checkpoint_path = os.path.join(root_dir, params["training"]['loss'])

    model = get_model(params["network"])

    model.load_weights(checkpoint_path)

    x = np.linspace(1, 200)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape([-1, 1])
    yy = yy.reshape([-1, 1])

    t = np.ones([xx.shape[0], 1])
    inputs = np.concatenate([t, xx, yy], axis=1)

    inputs = tf.convert_to_tensor(inputs, np.float32)
    pdb.set_trace()


if __name__ == '__main__':
    launch_train()
