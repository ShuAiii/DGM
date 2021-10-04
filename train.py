""""""
import os
import yaml
import click
from src.training.experiment import Experiment
import wandb


with open("/home/app_user/app/conf/secrets/creds.yml") as fff:
    key = yaml.load(fff, Loader=yaml.FullLoader)["wandb_key"]

os.environ['WANDB_API_KEY'] = key
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_params(params: dict):
    return params


@click.command()
@click.option('-p', '--project', required=True, type=str, help="WandB project name",
              default="DGM")
@click.option('-c', '--path', required=True, type=str, help="Path of the training conf file",
              default="/home/app_user/app/conf/black_scholes.yml")
def launch_train(project: str, path: str) -> None:

    with open(path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = parse_params(params)

    print(params)

    wandb.init(project=project)

    experiment = Experiment(
        training_kwargs=params["training"],
        dataset_kwargs=params["dataset"],
        network_kwargs=params["network"]
    )
    experiment.trainer()


if __name__ == '__main__':
    launch_train()
