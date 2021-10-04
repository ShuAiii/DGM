""""""
from itertools import cycle
import logging
import tensorflow as tf
import wandb

from .nets import get_model
from .loss import get_loss
from .dataset import Dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Experiment:
    """"""
    def __init__(self,
                 training_kwargs: dict,
                 network_kwargs: dict,
                 dataset_kwargs: dict
                 ):
        self.pde_name = dataset_kwargs["name"]
        self.pde_params = dataset_kwargs["variables"]
        self.build_network(network_kwargs)
        self.build_dataset(dataset_kwargs)
        self.num_validation_datasets = len(self.validation_dataset)
        self.build_trainer(training_kwargs)

    def build_dataset(self, dataset_kwargs) -> None:
        """"""
        train_kwargs = dataset_kwargs.pop("train")
        validation_kwargs = dataset_kwargs.pop("validation")
        self.train_dataset = Dataset(data_kwargs=train_kwargs,
                                     **dataset_kwargs)
        self.validation_dataset = Dataset(data_kwargs=validation_kwargs,
                                          is_validation=True,
                                          **dataset_kwargs)
        self.num_train_sets = len(self.train_dataset)
        self.train_cycle = cycle(range(self.num_train_sets))
        self.num_validation_sets = len(self.validation_dataset)

    def build_network(self, network_kwargs):
        """"""
        self.model = get_model(network_kwargs)

    def build_trainer(self, trainer_kwargs):
        """"""
        self.epoch = trainer_kwargs.get("epoch", 1000)
        self.steps = trainer_kwargs["epoch"] * self.num_train_sets
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.loss = get_loss(trainer_kwargs.get("loss", self.pde_name),
                             self.pde_params)

    def train_step(self):
        """"""
        index = next(self.train_cycle)
        train_batch = self.train_dataset.__getitem__(index)
        with tf.GradientTape() as tape:
            loss = self.loss(train_batch, self.model)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        #if index % self.num_train_sets == 0 and index != 0:
        wandb.log({"Train Loss": loss.numpy()})
        logging.info(f"Train Loss: {loss.numpy()}")

    def validation_step(self):
        """"""
        mae = 0
        for index in range(self.num_validation_sets):
            validation_batch, y = self.validation_dataset.__getitem__(index)
            y_prediction = self.model(validation_batch)
            mae += tf.math.reduce_mean(tf.abs(y_prediction - y)).numpy()
        wandb.log({"Mean Absolute Error": mae})
        logging.info(f"Mean Absolute Error: {mae}")

    def trainer(self):
        """"""
        for step in range(self.steps):
            self.train_step()
            if step % self.num_train_sets == 0 and step != 0:
                self.validation_step()
