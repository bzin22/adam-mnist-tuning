from pathlib import Path
from urllib.request import urlretrieve
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FOLDER = Path(__file__).resolve().parent # Use the script's folder to store the MNIST dataset and results.
EPOCHS, BATCH_SIZE = 200, 128
SEEDS = [41, 42, 43]

# Each experiment specifies: learning rate, whether to decay it, and L2 strength.
SETUPS = {
    "baseline": (0.00015, False, 0.0),
    "rate_only": (0.0003, False, 0.0),
    "rate_decay": (0.0003, True, 0.0),
    "tuned": (0.0003, True, 0.0001),
}
LABELS = ["Baseline", "Higher rate", "Higher rate + decay", "Higher rate + decay + L2"]
COLORS = ["#7863a6", "#64748b", "#0072b2", "#087e8b"]


def load_mnist():
    path = FOLDER / "mnist.npz"
    if path.exists() == False: # Download the MNIST dataset if it is not already present.
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urlretrieve(url, path)
    data = np.load(path)
    arrays = []
    for split in ["train", "test"]:
        # Flatten each image and apply the same scaling to both data splits.
        images = data["x_" + split].reshape(-1, 784).astype(np.float32)
        images = (images / 255.0 - 0.1307) / 0.3081
        labels = data["y_" + split].astype(np.int64)
        arrays.extend([torch.tensor(images),
                       torch.tensor(labels)])
    data.close()
    return arrays


def make_model(seed):
    torch.manual_seed(seed)
    model = nn.Sequential( # simple FF NN w/ Dropout layers, ReLU activations, and 2 hidden layers of 1000 units each.
        nn.Dropout(0.2), nn.Linear(784, 1000), nn.ReLU(),
        nn.Dropout(0.375), nn.Linear(1000, 1000), nn.ReLU(),
        nn.Dropout(0.375), nn.Linear(1000, 10),
    )
    # Give all optimizer setups the same starting weights for a given seed.
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                weights = rng.randn(layer.out_features, layer.in_features)
                weights = (weights / math.sqrt(layer.in_features)).astype(np.float32)
                layer.weight.copy_(torch.from_numpy(weights))
                layer.bias.zero_()
    return model


def test_loss(model, images, labels):
    model.eval()  # Disable dropout when measuring test predictions.
    losses = []
    with torch.no_grad():
        for start in range(0, len(labels), 1024):
            scores = model(images[start:start + 1024])
            loss = F.cross_entropy(scores, labels[start:start + 1024],
                                   reduction="none")
            losses.append(loss.numpy())
    return np.concatenate(losses).mean(dtype=np.float64)


def train_model(name, seed, data):
    images, labels, test_images, test_labels = data
    rate, decay, l2 = SETUPS[name]
    model = make_model(seed)
    weights = [p for n, p in model.named_parameters() if n.endswith("weight")] 
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    # Apply L2 to weights; leave biases unregularized.
    optimizer = torch.optim.Adam([
        {"params": weights, "weight_decay": l2},
        {"params": biases, "weight_decay": 0.0},
    ], lr=rate)
    rng = np.random.default_rng(seed + 100000)  # Separate stream for image order.
    batches = (len(labels) + BATCH_SIZE - 1) // BATCH_SIZE
    history = []
    for epoch in range(EPOCHS):
        torch.manual_seed(seed + epoch)
        model.train()
        order = torch.tensor(rng.permutation(len(labels)))
        total_loss = 0.0
        for batch, start in enumerate(range(0, len(labels), BATCH_SIZE)):
            index = order[start:start + BATCH_SIZE]
            progress = (epoch * batches + batch) / (EPOCHS * batches - 1)
            factor = 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate * factor if decay else rate
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images[index]), labels[index])
            total_loss += loss.item() * len(index)
            loss.backward()
            optimizer.step()
        train_nll = total_loss / len(labels)  # Dropout loss without L2 penalty
        test_nll = test_loss(model, test_images, test_labels)
        history.append([train_nll, test_nll])
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"{name}, seed {seed}, epoch {epoch + 1}: "
                  f"train={train_nll:.6f}, test={test_nll:.6f}")
    return np.array(history)


def plot_baseline_comparison(results):
    paper = np.loadtxt(FOLDER / "paper_curve.csv", delimiter=",", skiprows=1)
    epochs = np.arange(1, EPOCHS + 1)
    # Average the baseline training losses across the three seeds.
    baseline = np.mean([results[f"baseline_{seed}"][:, 0] for seed in SEEDS], axis=0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, baseline, color=COLORS[0], label="Baseline (three-seed mean)")
    ax.plot(paper[:, 0], paper[:, 1], color="#d55e00", label="Paper Figure 2a (approx.)")
    ax.set(title="Baseline and published Adam curve", xlabel="Epoch",
           ylabel="Training cross-entropy", yscale="log", xlim=(1, EPOCHS))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()


def show_results(results):
    plot_baseline_comparison(results)
    epochs = np.arange(1, EPOCHS + 1)
    test_fig, test_ax = plt.subplots(figsize=(6, 4.5))
    train_fig, train_ax = plt.subplots(figsize=(6, 4.5))
    print(f"\nMean losses at epoch {EPOCHS}")
    for name, label, color in zip(SETUPS, LABELS, COLORS):
        # Average the three runs; column 0 is training loss and column 1 is test loss.
        runs = np.array([results[f"{name}_{seed}"] for seed in SEEDS])
        mean = runs.mean(axis=0)
        print(f"{label:25s} train={mean[-1, 0]:.6f}  test={mean[-1, 1]:.6f}")
        test_ax.plot(epochs, mean[:, 1], color=color, label=label)
        train_ax.plot(epochs, mean[:, 0], color=color, label=label)
    test_ax.set(title="Test loss", ylim=(0, None))
    train_ax.set(title="Training loss with dropout", yscale="log")
    for fig, ax in [(test_fig, test_ax), (train_fig, train_ax)]:
        ax.set(xlabel="Epoch", ylabel="Cross-entropy", xlim=(1, EPOCHS))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
    plt.show()


def main():
    results_path = FOLDER / "losses.npz"
    if results_path.exists():  # Plot saved losses without repeating training.
        with np.load(results_path) as saved:
            results = {name: saved[name] for name in saved.files}
    else:
        print("No saved losses found. Training all setups and seeds.")
        data = load_mnist()
        results = {}
        for name in SETUPS:
            for seed in SEEDS:
                results[f"{name}_{seed}"] = train_model(name, seed, data)
        np.savez_compressed(results_path, **results)
        print(f"Saved losses to {results_path}")
    show_results(results)


if __name__ == "__main__":
    main()