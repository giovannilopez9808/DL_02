from keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.image import (
    random_flip_left_right,
    ResizeMethod,
    random_crop,
    resize
)
from tensorflow import Tensor
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
            **self.params["dataset"]["train"],
        )
        dog_dataset = self._normalization_train_dataset(dataset)
        dataset = image_dataset_from_directory(
            directory=cat_path,
            seed=2022,
            **self.params["dataset"]["train"],
        )
        cat_dataset = self._normalization_train_dataset(dataset)
        self.train = Dataset.zip((dog_dataset,
                                  cat_dataset))

    def _normalization_train_dataset(self,
                                     dataset: Dataset) -> Dataset:
        dataset = dataset.map(self._random_jitter,
                              num_parallel_calls=self.autotune)
        dataset = self._normalization(dataset)
        return dataset

    def _normalization(self,
                       dataset: Dataset) -> Dataset:
        dataset = dataset.map(normalization_layer,
                              num_parallel_calls=self.autotune)
        dataset = dataset.map(lambda x:
                              2*x-1)
        return dataset

    def _random_jitter(self,
                       image: Tensor) -> Tensor:
        image = resize(image,
                       [286, 286],
                       method=ResizeMethod.NEAREST_NEIGHBOR)
        # randomly cropping to 256 x 256 x 3
        image = random_crop(image,
                            size=[
                                1,
                                256,
                                256,
                                3])
        # random mirroring
        image = random_flip_left_right(image)
        return image
