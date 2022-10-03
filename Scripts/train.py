from Modules.VAE_pix2pix import VAE_pix2pix_model
from Modules.dataset import dataset_model
from Modules.params import get_params
from pandas import DataFrame
from os.path import join

params = get_params()
params["dataset"]["type"] = "all"
dataset = dataset_model(params)
model = VAE_pix2pix_model(params)
print(params["epochs"])
history = model.fit(dataset,
                    epochs=params["epochs"])
history = DataFrame(history)
filename = "cycleGAN_history.csv"
filename = join(params["path log"],
                filename)
history.to_csv(filename,
               index=False)
