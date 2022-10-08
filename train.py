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
%matplotlib

def un_transform(inp):
    #Undoes the resnet transform
    inp = inp.permute(0,2,3,1)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp.permute(0,3,1,2)

def show_pic(pic, title):
  plt.title(title)
  plt.imshow(pic.permute(1,2,0))

# Visualize on picture
pic_name = os.listdir("pics120/train/FR")[0]
country = pic_name[0:2]
pic_tensor = read_image(os.path.join("pics120/train", country, pic_name))
show_pic(pic_tensor, country)

## Get pretrained CNN weights
weights = torchvision.models.ResNet50_Weights.DEFAULT

# Define training dataloader & transforms
train_dataset = torchvision.datasets.ImageFolder("pics120/train",
                transform = transforms.Compose([
                        transforms.RandomResizedCrop(300,scale=(0.5,1)),
                        weights.transforms()]),
                target_transform = transforms.Lambda(lambda c: nn.functional.one_hot(torch.tensor(c), num_classes=len(class_idx))))
class_idx = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=4)
# Visualize
inputs, classes = next(iter(train_loader))
show_pic(torchvision.utils.make_grid(un_transform(inputs)), [class_idx[i] for i in classes.argmax(1)])

# Define validation dataloader & transforms
val_dataset = torchvision.datasets.ImageFolder("pics120/val",
                transform = transforms.Compose([
                    transforms.FiveCrop(400),
                    transforms.Lambda(lambda crops: torch.stack([weights.transforms()(crop) for crop in crops])),
                ]),
                target_transform = transforms.Lambda(lambda c: nn.functional.one_hot(torch.tensor(c), num_classes=len(class_idx))))
val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=10,
                                          shuffle=True,
                                          num_workers=4)


## Define model
model_conv = torchvision.models.resnet50(weights=weights)
for param in model_conv.parameters():
    param.requires_grad = False
# Optimize last layer as well
for param in model_conv.layer4.parameters():
    param.requires_grad = True

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_idx))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.5, momentum=0.9)

# Decay LR by a factor of 0.1 every 50 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=50, gamma=0.1)



### Training
model = model_conv
optimizer = optimizer_conv
scheduler = exp_lr_scheduler
num_epochs = 500

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
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
            loss = criterion(outputs.softmax(1), labels.to(torch.float32))

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        _, label_ind = torch.max(labels, 1)
        running_corrects += torch.sum(preds == label_ind)

    scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch%10 != 0:
        continue

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
            b, cr, co, w, h = inputs.shape
            outputs = model(inputs.view(-1,co,w,h))
            outputs = outputs.view(b, cr, -1).mean(1)
            loss = criterion(outputs.softmax(1), labels.to(torch.float32))

        # statistics
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        corrects = labels.argmax(1)
        running_corrects += torch.sum(preds == corrects)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)

    print(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

## Training done!

# load best model weights
# model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), '100_BR_FI_FR_JP_US_4.pth')

# Reload model from disk if needed
model.load_state_dict(torch.load('100_BR_FI_FR_JP_US_4.pth'))
model.eval()



### Predict new ones?
# Fivecrop
test_dataset = torchvision.datasets.ImageFolder("pics120/test",
        transform = transforms.Compose([
                transforms.FiveCrop(400),
                transforms.Lambda(lambda crops: torch.stack([weights.transforms()(crop) for crop in crops])),
                ]),
        target_transform = transforms.Lambda(lambda c: nn.functional.one_hot(torch.tensor(c), num_classes=len(class_idx))))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=True)
it = iter(test_loader)
# Visualize
inputs, classes = next(it)
b, cr, co, w, h = inputs.shape
_, max_idx = torch.max(classes, 1)
guess = nn.functional.softmax(model(inputs.view(-1,co,w,h)).mean(0), dim=0)
title = f"Probs {class_idx}: {(100*guess.detach()).round()}\nTrue:    {class_idx[max_idx.item()]}"
print(title)
show_pic(torchvision.utils.make_grid(un_transform(inputs.view(-1,co,w,h))), title)

acc = 0
for inputs, classes in test_loader:
    b, cr, co, w, h = inputs.shape
    guess = model(inputs.view(-1,co,w,h))
    guess = guess.view(b, cr, -1).mean(1)
    acc += (guess.argmax(1) == classes.argmax(1)).sum()

print(f"Test accuracy {acc/len(test_dataset)}")


# No fivecrop
test_dataset = torchvision.datasets.ImageFolder("pics120/test",
        transform = weights.transforms(),
        target_transform = transforms.Lambda(lambda c: nn.functional.one_hot(torch.tensor(c), num_classes=len(class_idx))))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=True)
it = iter(test_loader)
# Visualize
inputs, classes = next(it)
max_idx = classes.argmax(1)
guess = nn.functional.softmax(model(inputs), dim=1)
title = f"Probs {class_idx}: {(100*guess.detach()).round()}\nTrue:    {class_idx[max_idx.item()]}"
print(title)
show_pic(torchvision.utils.make_grid(un_transform(inputs.view(-1,co,w,h))), title)

acc = 0
for inputs, classes in test_loader:
    guess = nn.functional.softmax(model(inputs), dim=1)
    acc += (guess.argmax(1) == classes.argmax(1)).sum()

print(f"Test accuracy {acc/len(test_dataset)}")
