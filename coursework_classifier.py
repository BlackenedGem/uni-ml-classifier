# Tweakable constants
QUICK_MODE = False

if QUICK_MODE:
    lin_layer_features = 2000
    NUM_EPOCHS = 35
else:
    lin_layer_features = 32000
    NUM_EPOCHS = 45

# region Imports and pytorch setup

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

from functions import *

print("Using Pytorch " + torch.__version__)
print(f"{torch.cuda.device_count()} CUDA device(s) found")
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("Using CUDA " + torch.version.cuda + " - " + device_name)
else:
    print("Using CPU")
    device = torch.device('cpu')

# endregion
# region Import dataset

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
               'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
               'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
               'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
               'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm', ]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])),
    shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])),
    shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset):,}')
print(f'> Size of test dataset {len(test_loader.dataset):,}')

# endregion
# region Define a simple model

# define the model (a simple classifier)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)

        self.dropout_low = nn.Dropout2d(0.2)
        self.dropout_high = nn.Dropout2d(0.4)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=4, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.lin1 = nn.Linear(in_features=128 * 3 * 3, out_features=lin_layer_features)
        self.lin2 = nn.Linear(in_features=lin_layer_features, out_features=100)

    def forward(self, x):
        # print(x.shape)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)  # flatten input as we're using linear layers
        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        # print(x.shape)
        # print("Done")

        return x

N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters())):,}')

# initialise the optimiser
optimiser = torch.optim.SGD(N.parameters(), lr=1e-3, momentum=0.9)
epoch = 0
# liveplot = PlotLosses()

# endregion
# region Main training and testing loop

# Variables to track performance
start_time = time.time()
loop_start_time = time.time()

train_acc_graph = []
test_acc_graph = []
train_loss_graph = []
test_loss_graph = []

best_acc = 0
best_epoch = 0

while epoch < NUM_EPOCHS:
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate over some of the train dataset
    for x, t in train_loader:
        x, t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(x)
        pred = p.argmax(dim=1, keepdim=True)
        train_loss = torch.nn.functional.cross_entropy(p, t)
        train_loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, torch.mean(train_loss).item())
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    # iterate entire test dataset
    for x, t in test_loader:
        x, t = x.to(device), t.to(device)

        p = N(x)
        train_loss = torch.nn.functional.cross_entropy(p, t)
        pred = p.argmax(dim=1, keepdim=True)

        test_loss_arr = np.append(test_loss_arr, torch.mean(train_loss).item())
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    total_duration = time.time() - start_time
    loop_duration = time.time() - loop_start_time

    print(f"Epoch {epoch + 1} finished ({format_time(loop_duration)}/{format_time(total_duration)})")
    print("\tAccuracy: " + format_acc(train_acc_arr.mean(), convert_to_percentage=True))
    print("\tVal Accuracy: " + format_acc(test_acc_arr.mean(), convert_to_percentage=True))
    print("\tLoss: " + format_acc(train_loss_arr.mean()))
    print("\tVal loss: " + format_acc(test_loss_arr.mean()))

    train_acc_graph.append(train_acc_arr.mean() * 100)
    test_acc_graph.append(test_acc_arr.mean() * 100)
    train_loss_graph.append(train_loss_arr.mean())
    test_loss_graph.append(test_loss_arr.mean())

    if test_acc_arr.mean() > best_acc:
        best_acc = test_acc_arr.mean()
        best_epoch = epoch

    epoch += 1
    loop_start_time = time.time()

    # if epoch == num_epochs:
    #     try:
    #         extra_epochs = input("Number of epochs to use: ")
    #         num_epochs += int(extra_epochs)
    #     except ValueError:
    #         pass

# endregion

print(f"Best test accuracy occurred in epoch {best_epoch + 1}: " + format_acc(best_acc, convert_to_percentage=True))
create_end_graphs(train_acc_graph, test_acc_graph, train_loss_graph, test_loss_graph)
