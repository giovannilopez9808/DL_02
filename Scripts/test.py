from dataset import dataset_model
import matplotlib.pyplot as plt
from params import get_params


params = get_params()
dataset = dataset_model(params)
for images in dataset.train.take(1):
    for i in range(len(images)):
        ax = plt.subplot(3, 3, i+1)
        ax.imshow(images[i].numpy())
        ax.axis("off")
plt.tight_layout()
plt.show()
