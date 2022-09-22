from dataset import dataset_model
from params import get_params
from model import model

params = get_params()
dataset = dataset_model(params)
model = model(params)
model.run(dataset.train)
