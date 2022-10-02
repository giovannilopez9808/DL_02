from keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.data import (
    AUTOTUNE,
    Dataset
)
from os.path import join
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


class dataset_model:
    """
    -
    """

    def __init__(self,
                 params: dict) -> None:
        self.autotune = AUTOTUNE
        self.params = params
        self.train = None
        self.test = None
        self._read_dataset()

    def _read_dataset(self) -> tuple:
        self._read_train_dataset()

    def _read_train_dataset(self) -> Dataset:
        path = join(self.params["path data"],
                    "train")
        dog_path = join(path,
                        "dog")
        cat_path = join(path,
                        "cat")
        dataset = image_dataset_from_directory(
            directory=dog_path,
            seed=2022,
            **self.params["dataset"]["train"],
        )
        dog_dataset = self._normalization_dataset(dataset)
        dataset = image_dataset_from_directory(
            directory=cat_path,
            seed=2022,
            **self.params["dataset"]["train"],
        )
        cat_dataset = self._normalization_dataset(dataset)
        if self.params["dataset"]["type"] == "dog":
            self.train = dog_dataset
        elif self.params["dataset"]["type"] == "cat":
            self.train = cat_dataset
        else:
            self.train = Dataset.zip((dog_dataset,
                                      cat_dataset))
        size = self.train.cardinality().numpy()
        self.test = self.train.take(1)
        self.train = self.train.skip(1).take(size-1)

    def _normalization_dataset(self,
                               dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda image:
            (normalization_layer(image)),
            num_parallel_calls=self.autotune
        )
        dataset = dataset.map(
            lambda image:
            image*2-1
        )
        return dataset
