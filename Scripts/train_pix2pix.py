from Modules.dataset import dataset_model
from Modules.pix2pix import pix2pix_model
from Modules.params import get_params
from pandas import DataFrame
from os.path import join
from sys import argv

params = get_params()
params["dataset"]["type"] = argv[1]
dataset = dataset_model(params)
model = pix2pix_model()
history = model.fit(dataset.train,
                    epochs=params["epochs"])
# history = history.history
history = DataFrame(history)
filename = f"../{argv[1]}_history.csv"
history.to_csv(filename,
               index=False)
filename = f"{argv[1]}_model.h5"
filename = join(params["path models"],
                filename)
model.save_weights(filename)
