from keras.callbacks import ModelCheckpoint
from tensorflow.data import Dataset
from keras.optimizers import Adam
from os.path import join
from os import listdir
from VAE import VAE


class model:
    def __init__(self,
                 params: dict) -> None:
        self.params = params
        self._get_VAE()

    def _get_VAE(self) -> None:
        self.model = VAE(**self.params["VAE"])
        self._compile_model()
        self._create_callbacks()

    def _compile_model(self) -> None:
        self.model.compile(optimizer=Adam())

    def _create_callbacks(self) -> None:
        filename = "Checkpoint_model.h5"
        filename = join(self.params["path models"],
                        filename)
        self.callbacks = [
            ModelCheckpoint(
                save_weights_only=True,
                save_best_only=True,
                filepath=filename,
                monitor="loss",
                mode="min",
                verbose=1,
            )]

    def run(self,
            dataset: Dataset) -> None:
        batch_size = self.params["dataset"]["train"]["batch_size"]
        steps_per_epoch = self._get_total_train_images() // batch_size
        epochs = self.params["epochs"]
        self.model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks,
            batch_size=batch_size,
            epochs=epochs,
        )
        self._save_weights()

    def _get_total_train_images(self) -> int:
        path = join(self.params["path data"],
                    "train")
        files = listdir(path)
        size = len(files)
        return size

    def _save_weights(self) -> None:
        filename = "Final_model.he5"
        filename = join(self.params["path models"],
                        filename)
        self.model.save_weights(filename)
