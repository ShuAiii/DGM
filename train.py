""""""
import os
from typing import Dict
import pdb

import yaml
import click
import logging
from src.training.experiment import Experiment
import wandb


with open("/home/app_user/app/conf/secrets/creds.yml") as fff:
    key = yaml.load(fff, Loader=yaml.FullLoader)["wandb_key"]

os.environ['WANDB_API_KEY'] = key
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_logger(params: Dict):

    use_wandb = params["training"]["use_wandb"]
    project = params["training"].get("project", "DGM")

    if not use_wandb:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(message)s',
                            datefmt='%H:%M:%S')

        logger = logging.getLogger(__name__)

        return lambda x, y: logger.info(f"{x}: {y}")

    else:
        wandb.init(project=project)
        return lambda x, y: wandb.log({x: y})


def parse_params(params: dict):
    return params


@click.command()
@click.option('-c', '--path', required=True, type=str, help="Path of the training conf file",
              default="/home/app_user/app/conf/black_scholes.yml")
def launch_train(path: str) -> None:

    with open(path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = parse_params(params)

    print(params)

    logger = get_logger(params)

    experiment = Experiment(
        training_kwargs=params["training"],
        dataset_kwargs=params["dataset"],
        network_kwargs=params["network"],
        logger=logger
    )
    experiment.trainer()


if __name__ == '__main__':
    launch_train()
