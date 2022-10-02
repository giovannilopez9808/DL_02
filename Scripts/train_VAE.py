from Modules.dataset import dataset_model
from Modules.params import get_params
from pandas import DataFrame
from Modules.VAE import VAE
from os.path import exists
from os.path import (
    exists,
    join
)
from sys import argv

params = get_params()
params["dataset"]["type"] = argv[1]
filename = f"VAE_{argv[1]}.h5"
filename = join(params["path models"],
                filename)
dataset = dataset_model(params)
model = VAE(**params["VAE"])
if exists(filename):
    model.load_weights(filename)
history = model.fit(dataset.train,
                    batch_size=16,
                    epochs=params["epochs"])
model.save_weights(filename)
filename = f"../VAE_{argv[1]}_history.csv"
history = DataFrame(history.history)
history.to_csv(filename,
               index=False)
