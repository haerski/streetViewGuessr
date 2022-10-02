import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def show_pic(pic, title):
  plt.title(title)
  plt.imshow(pic.permute(1,2,0))

pic_name = os.listdir("pics/JP")[0]
country = pic_name[0:2]
pic_tensor = read_image(os.path.join("pics", country, pic_name))
show_pic(pic_tensor, country)

dataset = torchvision.datasets.ImageFolder("pics", T.ToTensor())
class_idx, class_map = dataset.find_classes(dataset.root)

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True)
inputs, classes = next(iter(data_loader))
show_pic(torchvision.utils.make_grid(inputs), [class_idx[i] for i in classes])