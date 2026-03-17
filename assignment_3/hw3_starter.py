import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from scipy.stats import entropy
from time import time

num_classes = 10
num_layers = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
dataset = CIFAR10(root="./data", download=True, transform=ToTensor())
test_dataset = CIFAR10(root="./data", train=False, transform=ToTensor())

batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size
_, val_ds = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)


class Branch(nn.Module):
    def __init__(self, in_channels, in_features):
        super(Branch, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.bn = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.in_channels = [32, 32, 64, 64, 128]
        self.in_features = [3600, 784, 784, 144, 144]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.4)

        self.branch1 = Branch(
            in_channels=self.in_channels[0], in_features=self.in_features[0]
        )
        self.branch2 = Branch(
            in_channels=self.in_channels[1], in_features=self.in_features[1]
        )
        self.branch3 = Branch(
            in_channels=self.in_channels[2], in_features=self.in_features[2]
        )
        self.branch4 = Branch(
            in_channels=self.in_channels[3], in_features=self.in_features[3]
        )
        self.branch5 = Branch(
            in_channels=self.in_channels[4], in_features=self.in_features[4]
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=128, out_features=num_classes)

        self.num_layers = num_layers

    def forward(self, tensor_after_previous_layer, exit_layer_idx=num_layers):
        if exit_layer_idx == 0:
            x = self.conv1(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn1(x)
            predicted_scores_from_layer = self.branch1(tensor_after_layer)

        elif exit_layer_idx == 1:
            x = self.conv2(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.pool1(x)
            tensor_after_layer = self.dropout1(x)
            predicted_scores_from_layer = self.branch2(tensor_after_layer)

        elif exit_layer_idx == 2:
            x = self.conv3(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn3(x)
            predicted_scores_from_layer = self.branch3(tensor_after_layer)

        elif exit_layer_idx == 3:
            x = self.conv4(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn4(x)
            x = self.pool2(x)
            tensor_after_layer = self.dropout2(x)
            predicted_scores_from_layer = self.branch4(tensor_after_layer)

        elif exit_layer_idx == 4:
            x = self.conv5(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn5(x)
            predicted_scores_from_layer = self.branch5(tensor_after_layer)

        elif exit_layer_idx == 5:
            x = self.conv6(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn6(x)
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.bn7(x)
            tensor_after_layer = self.dropout4(x)
            predicted_scores_from_layer = self.fc5(tensor_after_layer)

        else:
            ValueError(f"exit_layer_idx {exit_layer_idx} should be int within 0 to 5")

        return tensor_after_layer, predicted_scores_from_layer


model = Baseline().to(device)
model.load_state_dict(torch.load("cifar10_branchyNet_m.h5", map_location="cpu"))
model.eval()


def cutoff_exit_performance_check(cutoff, print_per_layer_performance=False):
    """
    TODO: On test data, run the model by iterating through exit layer indices.
    Decide, based on entropy, whether to exit from a particular layer or not.
    Please utilize tensors  after a layer for the next layer, if not exited.
    If print_per_layer_performance is True, please print accuracy and time
    for each layer. We want to see the printables for only one value. When
    plotting, you don't need to print these.
    """

    return overall_accuracy, total_time


def estimate_thresholds(desired_accuracy):
    """
    TODO: On validation data, for each layer, estimate entropy cutoff that
    gives the desired accuracy. Consider the samples exited and skip those
    samples when estimating the thresholds for the following layers.
    """

    return estimated_thresholds


# TODO: 1(a) For a fixed value of cutoff, show performance for all layers.


# TODO: 1(b) Plot overall accuracy vs cutoff, total time vs cutoff
# and total time vs overall accuracy.


# TODO: 2(a) On validation data, estimate threshold for each layer based on
# desired minimum accuracy. Use said list of thresholds on test data.


# TODO: 2(c) Vary the desired minimum accuracy and generate lists of
# thresholds. For the list of list of thresholds, plot total time
# vs overall accuracy.
