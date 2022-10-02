import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

def show_pic(pic, title):
  plt.title(title)
  plt.imshow(pic.permute(1,2,0))

pic_name = os.listdir("pics/JP")[0]
country = pic_name[0:2]
pic_tensor = read_image(os.path.join("pics", country, pic_name))
show_pic(pic_tensor, country)
