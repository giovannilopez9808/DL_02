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
history = model.fit(dataset,
                    epochs=params["epochs"])
history = DataFrame(history)
filename = f"../VAE_{argv[1]}_history.csv"
history.to_csv(filename,
               index=False)
filename = f"VAE_{argv[1]}.h5"
filename = join(params["path models"],
                filename)
model.save_weights(filename)
