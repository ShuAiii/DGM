""""""
import pdb
import os
from typing import Tuple
import tensorflow as tf

from .nets import get_model
from .loss import get_loss
from .dataset import Dataset
from pathlib import Path


class Experiment:
    """

    """
    def __init__(self,
                 training_kwargs: dict,
                 network_kwargs: dict,
                 dataset_kwargs: dict,
                 logger
                 ):

        self.root_dir = training_kwargs["root_dir"]
        self.save_path = os.path.join(self.root_dir, training_kwargs['loss'])
        self.pde_params = dataset_kwargs["variables"]
        self.build_network(network_kwargs)
        self.build_dataset(dataset_kwargs)
        self.build_trainer(training_kwargs)
        self.logger = logger

        # Create a project directory
        Path(training_kwargs["root_dir"]).mkdir(parents=True, exist_ok=True)

    def build_dataset(self, dataset_kwargs) -> None:
        """

        """

        self.dataset = Dataset(**dataset_kwargs)

    def build_network(self, network_kwargs):
        """

        """

        self.model = get_model(network_kwargs)

    def build_trainer(self, trainer_kwargs):
        """

        """

        self.steps = trainer_kwargs["steps"]
        self.validation_frequency = trainer_kwargs["validation_frequency"]
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        try:
            self.loss = get_loss(trainer_kwargs["loss"])
        except KeyError:
            raise ValueError("`loss` must be provided in training kwargs")

    def train_step(self) -> float:
        """

        """

        train_batch = self.dataset.generate()
        with tf.GradientTape() as tape:
            loss = self.loss(train_batch, self.model)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def validation_step(self) -> Tuple[float, float]:
        """

        """

        validation_batch = self.dataset.generate()
        loss = self.loss(validation_batch, self.model)
        mae = self.loss.mae(validation_batch, self.model)

        return loss.numpy(), mae.numpy()

    def trainer(self):
        """

        """

        for idx, step in enumerate(range(self.steps)):

            run_eval = True if (idx % self.validation_frequency == 0 and idx != 0) else False
            train_loss = self.train_step()
            if run_eval:
                validation_loss, validation_mae = self.validation_step()
                pdb.set_trace()
                self.model.save_weights(self.save_path)
                #self.model.load_weights(self.save_path)
                self.logger("Train Loss", train_loss)
                self.logger("Validation Loss", validation_loss)
                self.logger("Validation MAE", validation_mae)
