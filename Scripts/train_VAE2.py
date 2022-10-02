from Modules.VAE_pix2pix import VAE_pix2pix_model
from Modules.dataset import dataset_model
from Modules.params import get_params
from pandas import DataFrame
from os.path import join
from sys import argv

params = get_params()
params["dataset"]["type"] = argv[1]
dataset = dataset_model(params)
model = VAE_pix2pix_model(params)
history = model.fit(dataset,
                    epochs=params["epochs"])
history = DataFrame(history)
filename = f"../VAE_{argv[1]}_history.csv"
history.to_csv(filename,
               index=False)
