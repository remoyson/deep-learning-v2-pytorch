import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from time import time


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('CUDA is available, training on GPU ...')
else:
    print('CUDA is not available, training on CPU ...')


data_dir = 'flower_photos/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)
print('Data ingeladen.')

print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))

batch_size = 20
num_workers = 0

train_loader = torch.utils.data.Dataloader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.Dataloader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True)

vgg16 = models.vgg16(pretrained=True)
print('Pretrained vgg16 ingeladen.')

for param in vgg16.features.parameters():
    param.requires_grad = False

n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer

if train_on_gpu:
    vgg16.cuda()

print('Model defined')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

print('Loss function and optimizer defined.')

# number of epochs to train the model
start_time_training = time()
n_epochs = 2
for epoch in range(1, n_epochs + 1):
    train_loss = 0.0

    for batch_number, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # forward pass
        output = vgg16(data)
        # calculate loss
        loss = criterion(output, target)
        # backward pass
        loss.backward()
        # optimization step
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        if batch_number % 20 == 19:
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_number + 1, train_loss / 20))
            train_loss = 0.0
end_time_training = time()
duration_training_in_minutes = (end_time_training - start_time_training)/60
print('Time training in minutes: ', str(duration_training_in_minutes))



