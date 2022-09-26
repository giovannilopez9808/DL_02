from Modules.dataset import dataset_model
from Modules.model import CycleGAN_model
from Modules.params import get_params
# from Modules.model import model

params = get_params()
dataset = dataset_model(params)
model = CycleGAN_model(params)
history=model.run(dataset.train)
history.to_csv("../history.csv",
               index=False)
