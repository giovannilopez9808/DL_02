from Modules.dataset import dataset_model
from Modules.model import CycleGAN_model
from Modules.params import get_params
# from Modules.model import model

params = get_params()
dataset = dataset_model(params)
model = CycleGAN_model(params)
model.run(dataset.dog_train,
          dataset.cat_train)
