import torch
import torchvision
from torchvision import transforms
from PIL import Image

model = torch.jit.load("BR_FI_FR_JP_US_scripted.pt")
model.eval()
labels = ['BR', 'FI', 'FR', 'JP', 'US']

# Define transforms,
# We will cut each picture into five, and take the mean of all predictions
model_trans = torchvision.models.ResNet101_Weights.DEFAULT.transforms()
trans = transforms.Compose([
                transforms.FiveCrop(400),
                transforms.Lambda(lambda crops: torch.stack([model_trans(crop) for crop in crops])),
                ])

# Make a guess for each sample picture
for country in labels:
  filename = f"sample_pics/{country}.jpg"
  im = Image.open(filename)
  preds = model(trans(im)).detach()
  # Softmax the mean of the five raw scores
  probs = (preds.mean(0).softmax(0).squeeze() * 100)
  guess_string = ""
  for c, p in zip(labels, probs):
    guess_string += f"{c}: {p:.1f}%, "
  print(f"True: {country}, Guess: {labels[probs.argmax().item()]}\nProbabilities: {guess_string}\n")

