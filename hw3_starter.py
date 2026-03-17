from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

num_classes = 10
num_layers = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

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
            raise ValueError(
                f"exit_layer_idx {exit_layer_idx} should be int within 0 to 5"
            )

        return tensor_after_layer, predicted_scores_from_layer


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        features, target = self.dataset[sample_index]
        return features, target, sample_index


model = Baseline().to(device)
model.load_state_dict(torch.load("cifar10_branchyNet_m.h5", map_location="cpu"))
model.eval()


def _sync_device():
    if device.type == "cuda":
        torch.cuda.synchronize()


def _compute_entropy(logits):
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return entropy(probabilities, axis=1)


def _evaluate_with_thresholds(
    loader,
    thresholds,
    print_per_layer_performance=False,
    print_classification=False,
):
    thresholds = list(thresholds)
    if len(thresholds) != num_layers + 1:
        raise ValueError(
            f"Expected {num_layers + 1} thresholds, received {len(thresholds)}."
        )

    total_samples = 0
    total_correct = 0
    all_targets = []
    all_predictions = []
    per_layer_stats = [
        {"count": 0, "correct": 0, "time": 0.0} for _ in range(num_layers + 1)
    ]

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            active_x = images.to(device)
            active_y = labels.to(device)
            total_samples += labels.size(0)

            for layer_idx in range(num_layers + 1):
                if active_y.numel() == 0:
                    break

                _sync_device()
                start_time = time()
                active_x, logits = model(active_x, exit_layer_idx=layer_idx)
                _sync_device()
                per_layer_stats[layer_idx]["time"] += time() - start_time

                entropies = _compute_entropy(logits)
                predictions = logits.argmax(dim=1)

                if layer_idx == num_layers:
                    exit_mask = torch.ones_like(active_y, dtype=torch.bool)
                else:
                    exit_mask = torch.from_numpy(
                        entropies <= thresholds[layer_idx]
                    ).to(device)

                exited_count = int(exit_mask.sum().item())
                if exited_count > 0:
                    exited_predictions = predictions[exit_mask]
                    exited_labels = active_y[exit_mask]
                    exited_correct = int(
                        (exited_predictions == exited_labels).sum().item()
                    )
                    total_correct += exited_correct
                    per_layer_stats[layer_idx]["count"] += exited_count
                    per_layer_stats[layer_idx]["correct"] += exited_correct
                    all_targets.extend(exited_labels.cpu().tolist())
                    all_predictions.extend(exited_predictions.cpu().tolist())

                remaining_mask = ~exit_mask
                active_x = active_x[remaining_mask]
                active_y = active_y[remaining_mask]

    overall_accuracy = total_correct / total_samples
    total_time = sum(layer_stat["time"] for layer_stat in per_layer_stats)

    if print_per_layer_performance:
        print("Per-layer early-exit performance")
        for layer_idx, layer_stat in enumerate(per_layer_stats):
            exited_samples = layer_stat["count"]
            layer_accuracy = (
                layer_stat["correct"] / exited_samples if exited_samples > 0 else 0.0
            )
            print(
                f"Layer {layer_idx}: exited={exited_samples:5d}, "
                f"accuracy={layer_accuracy:.4f}, "
                f"time={layer_stat['time']:.4f}s"
            )
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Total inference time: {total_time:.4f}s")

    if print_classification and all_targets:
        print(classification_report(all_targets, all_predictions))

    return overall_accuracy, total_time, per_layer_stats


def _select_entropy_threshold(entropies, correctness, desired_accuracy):
    if len(entropies) == 0:
        return -np.inf

    entropies = np.asarray(entropies)
    correctness = np.asarray(correctness, dtype=np.float32)
    sorted_indices = np.argsort(entropies)
    sorted_entropies = entropies[sorted_indices]
    sorted_correctness = correctness[sorted_indices]
    cumulative_accuracy = np.cumsum(sorted_correctness) / (
        np.arange(len(sorted_correctness)) + 1
    )
    valid_indices = np.where(cumulative_accuracy >= desired_accuracy)[0]

    if len(valid_indices) == 0:
        return -np.inf

    return float(sorted_entropies[valid_indices[-1]])


def _forward_to_layer(inputs, layer_idx):
    features = inputs.to(device)
    logits = None
    for current_idx in range(layer_idx + 1):
        features, logits = model(features, exit_layer_idx=current_idx)
    return features, logits


def _gather_layer_outputs(loader, layer_idx):
    entropies = []
    correctness = []
    sample_indices = []

    with torch.no_grad():
        for inputs, targets, indices in loader:
            _, logits = _forward_to_layer(inputs, layer_idx)
            predictions = logits.argmax(dim=1).cpu().numpy()
            entropies.extend(_compute_entropy(logits))
            correctness.extend((predictions == targets.numpy()).astype(np.float32))
            sample_indices.extend(indices.numpy().tolist())

    return np.asarray(entropies), np.asarray(correctness), np.asarray(sample_indices)


def cutoff_exit_performance_check(cutoff, print_per_layer_performance=False):
    """
    On test data, run the model by iterating through exit layer indices.
    Decide, based on entropy, whether to exit from a particular layer or not.
    Please utilize tensors after a layer for the next layer, if not exited.
    If print_per_layer_performance is True, print accuracy and time for each
    layer. When plotting, printing can be disabled.
    """

    thresholds = [cutoff] * num_layers + [np.inf]
    overall_accuracy, total_time, _ = _evaluate_with_thresholds(
        test_loader,
        thresholds,
        print_per_layer_performance=print_per_layer_performance,
        print_classification=print_per_layer_performance,
    )
    return overall_accuracy, total_time


def estimate_thresholds(desired_accuracy):
    """
    On validation data, estimate an entropy cutoff for each early-exit layer
    that achieves the desired minimum accuracy. After each threshold is
    estimated, samples that would exit at that layer are removed before the
    next layer is processed.
    """

    remaining_indices = list(range(len(val_ds)))
    estimated_thresholds = []

    for layer_idx in range(num_layers):
        if not remaining_indices:
            estimated_thresholds.extend([-np.inf] * (num_layers - layer_idx))
            break

        indexed_subset = IndexedSubset(val_ds, remaining_indices)
        layer_loader = DataLoader(indexed_subset, batch_size=batch_size * 2, num_workers=0)

        entropies, correctness, sample_indices = _gather_layer_outputs(
            layer_loader, layer_idx
        )
        threshold = _select_entropy_threshold(
            entropies, correctness, desired_accuracy
        )
        estimated_thresholds.append(threshold)

        remain_mask = entropies > threshold
        remaining_indices = sample_indices[remain_mask].tolist()

    estimated_thresholds.append(np.inf)
    return estimated_thresholds


def _make_scatter_plot(x_values, y_values, xlabel, ylabel, title, output_name):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / output_name, dpi=200)
    plt.close()


def _run_fixed_cutoff_experiments():
    fixed_cutoff = 0.75
    print(f"\n1(a) Fixed-cutoff experiment with cutoff={fixed_cutoff:.2f}")
    cutoff_exit_performance_check(
        fixed_cutoff, print_per_layer_performance=True
    )

    cutoffs = np.linspace(0.0, np.log(num_classes), 12)
    accuracies = []
    times = []
    for cutoff in cutoffs:
        accuracy, total_time = cutoff_exit_performance_check(cutoff)
        accuracies.append(accuracy)
        times.append(total_time)

    _make_scatter_plot(
        cutoffs,
        accuracies,
        "Entropy cutoff",
        "Overall accuracy",
        "Overall Accuracy vs Cutoff",
        "accuracy_vs_cutoff.png",
    )
    _make_scatter_plot(
        cutoffs,
        times,
        "Entropy cutoff",
        "Total inference time (s)",
        "Total Time vs Cutoff",
        "time_vs_cutoff.png",
    )
    _make_scatter_plot(
        accuracies,
        times,
        "Overall accuracy",
        "Total inference time (s)",
        "Total Time vs Overall Accuracy",
        "time_vs_accuracy_fixed_cutoffs.png",
    )

    return cutoffs, accuracies, times


def _evaluate_threshold_list_on_test(thresholds, print_details=False):
    return _evaluate_with_thresholds(
        test_loader,
        thresholds,
        print_per_layer_performance=print_details,
        print_classification=print_details,
    )[:2]


def _run_validation_threshold_experiments():
    desired_accuracy = 0.80
    thresholds = estimate_thresholds(desired_accuracy)
    test_accuracy, test_time = _evaluate_threshold_list_on_test(
        thresholds, print_details=True
    )

    print(f"\n2(a) Validation-derived thresholds for desired accuracy={desired_accuracy}")
    print(f"Estimated thresholds: {[round(threshold, 4) for threshold in thresholds]}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test inference time: {test_time:.4f}s")

    desired_accuracies = np.arange(0.70, 0.96, 0.05)
    threshold_sets = []
    test_accuracies = []
    test_times = []

    for minimum_accuracy in desired_accuracies:
        thresholds = estimate_thresholds(minimum_accuracy)
        accuracy, total_time = _evaluate_threshold_list_on_test(thresholds)
        threshold_sets.append(thresholds)
        test_accuracies.append(accuracy)
        test_times.append(total_time)

    _make_scatter_plot(
        test_accuracies,
        test_times,
        "Overall accuracy",
        "Total inference time (s)",
        "Total Time vs Overall Accuracy (Validation Thresholds)",
        "time_vs_accuracy_threshold_lists.png",
    )

    print("\n2(c) Thresholds by desired validation accuracy")
    for minimum_accuracy, thresholds in zip(desired_accuracies, threshold_sets):
        rounded_thresholds = [round(threshold, 4) for threshold in thresholds]
        print(
            f"Desired accuracy={minimum_accuracy:.2f} -> thresholds={rounded_thresholds}"
        )

    return desired_accuracies, threshold_sets, test_accuracies, test_times


if __name__ == "__main__":
    _run_fixed_cutoff_experiments()
    _run_validation_threshold_experiments()
    print(f"\nSaved plots to: {results_dir.resolve()}")
