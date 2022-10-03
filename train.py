import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch import nn
import torch.optim as optim
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def show_pic(pic, title):
  plt.title(title)
  plt.imshow(pic.permute(1,2,0))

# Visualize on picture
pic_name = os.listdir("pics/train/JP")[0]
country = pic_name[0:2]
pic_tensor = read_image(os.path.join("pics/train", country, pic_name))
show_pic(pic_tensor, country)

## Get pretrained CNN weights
weights = torchvision.models.ResNet18_Weights.DEFAULT

# Define training dataloader & transforms
train_dataset = torchvision.datasets.ImageFolder("pics/train", weights.transforms())
class_idx = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)
# Visualize
inputs, classes = next(iter(train_loader))
show_pic(torchvision.utils.make_grid(inputs), [class_idx[i] for i in classes])

# Define validation dataloader & transforms
val_dataset = torchvision.datasets.ImageFolder("pics/val", weights.transforms())
val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)


## Define model
model_conv = torchvision.models.resnet18(weights=weights)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)



### Training
model = model_conv
optimizer = optimizer_conv
scheduler = exp_lr_scheduler
num_epochs = 100

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

optimizer = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # train
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


    ## Validation
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)

    print(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

## Training done!
print(f'Best val Acc: {best_acc:4f}')

# load best model weights
model.load_state_dict(best_model_wts)



### Predict new ones?
test_dataset = torchvision.datasets.ImageFolder("pics/test", transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)
# Visualize
inputs, classes = next(iter(test_loader))
_, guesses = model(weights.transforms()(inputs)).max(1)
print(f"Guessed: {[class_idx[i] for i in guesses]}\nTrue:    {[class_idx[i] for i in classes]}")
show_pic(torchvision.utils.make_grid(inputs), [class_idx[i] for i in guesses])