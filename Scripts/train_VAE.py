from Modules.dataset import dataset_model
from Modules.params import get_params
from pandas import DataFrame
from Modules.VAE import VAE
from os.path import join
from sys import argv

params = get_params()
params["dataset"]["type"] = argv[1]
dataset = dataset_model(params)
model = VAE(**params["VAE"])
history = model.fit(dataset.train,
                    epochs=params["epochs"])
history = history.history
history = DataFrame(history)
filename = "../history.csv"
history.to_csv(filename)
filename = "model.h5"
filename = join(params["path model"],
                filename)
model.save_weights(filename)
