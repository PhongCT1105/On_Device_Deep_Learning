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
from torch.utils.data import random_split

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
    total_exits = num_layers + 1  # 5 intermediate branches + 1 final head

    exited_per_layer = [0] * total_exits
    correct_per_layer = [0] * total_exits
    time_per_layer = [0.0] * total_exits

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            current_tensor = images
            remaining_idx = torch.arange(len(images), device=device)

            for L in range(total_exits):
                t0 = time()
                feat, logits = model(current_tensor, exit_layer_idx=L)
                time_per_layer[L] += time() - t0

                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

                if L == total_exits - 1:
                    exit_mask = torch.ones_like(ent, dtype=torch.bool)
                else:
                    exit_mask = ent < cutoff

                preds = logits.argmax(dim=1)
                exited_labels = labels[remaining_idx[exit_mask]]
                exited_per_layer[L] += int(exit_mask.sum().item())
                correct_per_layer[L] += int(
                    (preds[exit_mask] == exited_labels).sum().item()
                )

                keep = ~exit_mask
                if keep.sum().item() == 0:
                    break
                current_tensor = feat[keep]
                remaining_idx = remaining_idx[keep]

    total_exited = sum(exited_per_layer)
    total_correct = sum(correct_per_layer)
    overall_accuracy = total_correct / total_exited
    total_time = sum(time_per_layer)

    if print_per_layer_performance:
        print(f"cutoff = {cutoff}")
        for L in range(total_exits):
            n = exited_per_layer[L]
            acc = correct_per_layer[L] / n if n > 0 else float("nan")
            print(
                f"  layer {L}: exited={n:5d}  acc={acc:.4f}  time={time_per_layer[L]:.4f}s"
            )
        print(f"  overall: acc={overall_accuracy:.4f}  total_time={total_time:.4f}s")

    return overall_accuracy, total_time


def estimate_thresholds(desired_accuracy):
    """
    TODO: On validation data, for each layer, estimate entropy cutoff that
    gives the desired accuracy. Consider the samples exited and skip those
    samples when estimating the thresholds for the following layers.
    """
    ent_layers = [[] for _ in range(num_layers)]
    corr_layers = [[] for _ in range(num_layers)]

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            current = images
            for L in range(num_layers):
                feat, logits = model(current, exit_layer_idx=L)
                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
                preds = logits.argmax(dim=1)
                ent_layers[L].append(ent.cpu())
                corr_layers[L].append((preds == labels).cpu())
                current = feat

    ent_layers = [torch.cat(e) for e in ent_layers]
    corr_layers = [torch.cat(c) for c in corr_layers]

    active = torch.ones_like(ent_layers[0], dtype=torch.bool)
    estimated_thresholds = []
    for L in range(num_layers):
        e = ent_layers[L][active]
        c = corr_layers[L][active].float()
        order = torch.argsort(e)
        e_s = e[order]
        c_s = c[order]
        running_acc = c_s.cumsum(0) / torch.arange(1, len(c_s) + 1, dtype=torch.float32)

        ok = running_acc >= desired_accuracy
        if ok.any():
            k = int(ok.nonzero().max().item())
            if k + 1 < len(e_s):
                tau = float(((e_s[k] + e_s[k + 1]) / 2).item())
            else:
                tau = float(e_s[k].item()) + 1e-6
        else:
            tau = -1.0
        estimated_thresholds.append(tau)

        active_idx = active.nonzero().flatten()
        exited_here = ent_layers[L][active_idx] < tau
        active[active_idx[exited_here]] = False

    return estimated_thresholds


def run_with_thresholds(thresholds, loader):
    """Evaluate on `loader` using per-layer entropy thresholds."""
    total_exits = num_layers + 1
    exited_per_layer = [0] * total_exits
    correct_per_layer = [0] * total_exits
    time_per_layer = [0.0] * total_exits

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            current_tensor = images
            remaining_idx = torch.arange(len(images), device=device)

            for L in range(total_exits):
                t0 = time()
                feat, logits = model(current_tensor, exit_layer_idx=L)
                time_per_layer[L] += time() - t0

                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

                if L == total_exits - 1:
                    exit_mask = torch.ones_like(ent, dtype=torch.bool)
                else:
                    exit_mask = ent < thresholds[L]

                preds = logits.argmax(dim=1)
                exited_labels = labels[remaining_idx[exit_mask]]
                exited_per_layer[L] += int(exit_mask.sum().item())
                correct_per_layer[L] += int(
                    (preds[exit_mask] == exited_labels).sum().item()
                )

                keep = ~exit_mask
                if keep.sum().item() == 0:
                    break
                current_tensor = feat[keep]
                remaining_idx = remaining_idx[keep]

    total_exited = sum(exited_per_layer)
    total_correct = sum(correct_per_layer)
    overall_accuracy = total_correct / total_exited
    total_time = sum(time_per_layer)
    return overall_accuracy, total_time


# TODO: 1(a) For a fixed value of cutoff, show performance for all layers.
cutoff_exit_performance_check(cutoff=0.6, print_per_layer_performance=True)


# TODO: 1(b) Plot overall accuracy vs cutoff, total time vs cutoff
# and total time vs overall accuracy.
max_entropy = np.log(num_classes)
cutoffs = np.linspace(0.0, max_entropy, 100)

sweep_accs = []
sweep_times = []
for c in cutoffs:
    acc, t = cutoff_exit_performance_check(cutoff=float(c),
                                           print_per_layer_performance=False)
    sweep_accs.append(acc)
    sweep_times.append(t)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(cutoffs, sweep_accs)
axes[0].set_xlabel("cutoff (entropy)")
axes[0].set_ylabel("overall accuracy")
axes[0].set_title("Accuracy vs Cutoff")

axes[1].plot(cutoffs, sweep_times)
axes[1].set_xlabel("cutoff (entropy)")
axes[1].set_ylabel("total time (s)")
axes[1].set_title("Total Time vs Cutoff")

axes[2].plot(sweep_times, sweep_accs, marker="o", markersize=3, linestyle="-")
axes[2].set_xlabel("total time (s)")
axes[2].set_ylabel("overall accuracy")
axes[2].set_title("Accuracy vs Total Time")

plt.tight_layout()
plt.savefig("task1b_sweep.png", dpi=150)
plt.show()

best_i = int(np.argmax(np.asarray(sweep_accs) / np.asarray(sweep_times)))
print(
    f"Best trade-off: cutoff={cutoffs[best_i]:.4f}  "
    f"acc={sweep_accs[best_i]:.4f}  time={sweep_times[best_i]:.4f}s"
)


# TODO: 2(a) On validation data, estimate threshold for each layer based on
# desired minimum accuracy. Use said list of thresholds on test data.
desired_acc_2a = 0.8
thresholds_2a = estimate_thresholds(desired_acc_2a)
print(f"\n[Task 2(a)] desired per-layer accuracy = {desired_acc_2a}")
for i, t in enumerate(thresholds_2a):
    print(f"  layer {i} threshold = {t:.4f}")

acc_2a, time_2a = run_with_thresholds(thresholds_2a, test_loader)
print(f"  test accuracy = {acc_2a:.4f}  total time = {time_2a:.4f}s")


# TODO: 2(c) Vary the desired minimum accuracy and generate lists of
# thresholds. For the list of list of thresholds, plot total time
# vs overall accuracy.
desired_accs = np.linspace(0.5, 0.95, 10)
all_thresholds = []
sweep2_accs = []
sweep2_times = []
for d in desired_accs:
    thr = estimate_thresholds(float(d))
    acc, t = run_with_thresholds(thr, test_loader)
    all_thresholds.append(thr)
    sweep2_accs.append(acc)
    sweep2_times.append(t)
    print(f"  desired={d:.2f}  test_acc={acc:.4f}  time={t:.4f}s  thresholds={[f'{x:.3f}' for x in thr]}")

plt.figure(figsize=(6, 4))
plt.plot(sweep2_times, sweep2_accs, marker="o")
for i, d in enumerate(desired_accs):
    plt.annotate(f"{d:.2f}", (sweep2_times[i], sweep2_accs[i]), fontsize=7)
plt.xlabel("total time (s)")
plt.ylabel("overall accuracy")
plt.title("Task 2(b): Inference Time vs Accuracy")
plt.tight_layout()
plt.savefig("task2b_sweep.png", dpi=150)
plt.show()

best_i = int(np.argmax(np.asarray(sweep2_accs) / np.asarray(sweep2_times)))
best_thr = all_thresholds[best_i]
best_acc_test, best_time_test = run_with_thresholds(best_thr, test_loader)
print(
    f"\n[Task 2(b)] best desired_acc = {desired_accs[best_i]:.2f}  "
    f"thresholds = {[f'{x:.3f}' for x in best_thr]}"
)
print(f"  test accuracy = {best_acc_test:.4f}  total time = {best_time_test:.4f}s")
