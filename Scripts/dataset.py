from keras.utils import image_dataset_from_directory
from tensorflow.data import (AUTOTUNE,
                             Dataset)
from tensorflow.keras import layers
from os.path import join


class dataset_model:
    def __init__(self,
                 params: dict) -> None:
        self.resize = layers.experimental.preprocessing.Rescaling(1./255)
        self.autotune = AUTOTUNE
        self.params = params
        self._read_dataset()

    def _read_dataset(self) -> tuple:
        self._read_train_dataset()

    def _read_train_dataset(self) -> Dataset:
        path = join(self.params["path data"],
                    "train")
        dataset = image_dataset_from_directory(
            directory=path,
            **self.params["dataset"]["train"],
        )
        self.train = self._normalization(dataset)

    def _normalization(self,
                       dataset: Dataset) -> Dataset:
        dataset_resize = dataset.map(lambda x:
                                     (self.resize(x)),
                                     num_parallel_calls=self.autotune)
        return dataset_resize
