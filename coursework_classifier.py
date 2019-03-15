"""coursework-classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/cwkx/cef73403ec60ab467f9761f822d75aa2/coursework-classifier.ipynb

%%capture
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
!pip install 'livelossplot==0.3.0'

**Main imports**
"""

# region Imports and pytorch setup

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time
from livelossplot import PlotLosses

from functions import *

print(f"{torch.cuda.device_count()} CUDA device(s) found")
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("Using CUDA - " + device_name)
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
        torchvision.transforms.ToTensor()
    ])),
    shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset):,}')
print(f'> Size of test dataset {len(test_loader.dataset):,}')

# endregion
# region View some of the test dataset

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_loader.dataset[i][0].permute(0, 2, 1).contiguous().permute(2, 1, 0), cmap=plt.cm.binary)
#     plt.xlabel(class_names[test_loader.dataset[i][1]])
# plt.show()

# endregion
# region Define a simple model

# define the model (a simple classifier)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features=3 * 32 * 32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=100))
        self.layers = layers

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input as we're using linear layers
        for m in self.layers:
            x = m(x)
        return x

N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters())):,}')

# initialise the optimiser
optimiser = torch.optim.SGD(N.parameters(), lr=0.001)
epoch = 0
# liveplot = PlotLosses()

# endregion
# region Main training and testing loop

start_time = time.time()
loop_start_time = time.time()

train_acc_graph = []
test_acc_graph = []
train_loss_graph = []
test_loss_graph = []

while epoch < 10:
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate over some of the train dataset
    for i in range(1000):
        x, t = next(train_iterator)
        x, t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(x)
        pred = p.argmax(dim=1, keepdim=True)
        train_loss = torch.nn.functional.cross_entropy(p, t)
        train_loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, train_loss.cpu().data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    # iterate entire test dataset
    for x, t in test_loader:
        x, t = x.to(device), t.to(device)

        p = N(x)
        train_loss = torch.nn.functional.cross_entropy(p, t)
        pred = p.argmax(dim=1, keepdim=True)

        test_loss_arr = np.append(test_loss_arr, train_loss.cpu().data)
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    total_duration = time.time() - start_time
    loop_duration = time.time() - loop_start_time

    print(f"Epoch {epoch + 1} finished ({format_time(loop_duration)}/{format_time(total_duration)})")
    print("\tAccuracy: " + str(train_acc_arr.mean()))
    print("\tVal Accuracy: " + str(test_acc_arr.mean()))
    print("\tLoss: " + str(train_loss_arr.mean()))
    print("\tVal loss: " + str(test_loss_arr.mean()))

    train_acc_graph.append(train_acc_arr.mean())
    test_acc_graph.append(test_acc_arr.mean())
    train_loss_graph.append(train_loss_arr.mean())
    test_loss_graph.append(test_loss_arr.mean())

    # NOTE: live plot library has dumb naming forcing our 'test' to be called 'validation'
    # liveplot.update({
    #     'accuracy': train_acc_arr.mean(),
    #     'val_accuracy': test_acc_arr.mean(),
    #     'loss': train_loss_arr.mean(),
    #     'val_loss': test_loss_arr.mean()
    # })
    # liveplot.draw()

    epoch += 1
    loop_start_time = time.time()

# endregion
# region Inference on dataset

# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     color = '#335599' if predicted_label == true_label else '#ee4433'
#
#     # noinspection PyTypeChecker
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('#ee4433')
#     thisplot[true_label].set_color('#335599')
#
# test_images, test_labels = next(test_iterator)
# test_images, test_labels = test_images.to(device), test_labels.to(device)
# test_preds = torch.softmax(N(test_images).view(test_images.size(0), len(class_names)), dim=1).data.squeeze().cpu().numpy()
#
# num_rows = 4
# num_cols = 4
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, test_preds, test_labels.cpu(), test_images.cpu().squeeze().permute(1, 3, 2, 0).contiguous().permute(3, 2, 1, 0))
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, test_preds, test_labels)
# plt.show()

# endregion

create_end_graphs(train_acc_graph, test_acc_graph, train_loss_graph, test_loss_graph)
