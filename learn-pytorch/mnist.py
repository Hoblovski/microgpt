import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "mnist_model.pth"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "mnist_checkpoint.pt"
CLASSES = [str(i) for i in range(10)]


def get_device():
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataloaders(data_dir, batch_size):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    training_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return training_data, test_data, train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device, max_batches=None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        # shape: [64, 1, 28, 28]

        # print(f'{x.shape=}, {y.shape=}')
        pred = model(x)
        # print(f'{pred.shape=}')
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = min((batch + 1) * len(x), size)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, max_batches=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    model.eval()
    test_loss, correct, seen = 0, 0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch >= max_batches:
                break

            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            seen += len(x)

    test_loss /= num_batches
    denominator = seen if max_batches is not None else size
    correct /= denominator
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def predict_one(model, test_data, device):
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = CLASSES[pred[0].argmax(0)], CLASSES[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def save_checkpoint(path, epoch, model, optimizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    epoch = checkpoint["epoch"]
    print(f"Resumed checkpoint from {path} at epoch {epoch}")
    return epoch


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch MNIST classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    training_data, test_data, train_dataloader, test_dataloader = make_dataloaders(
        args.data_dir, args.batch_size
    )

    for x, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    device = get_device()
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume:
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            max_batches=args.limit_train_batches,
        )
        test(test_dataloader, model, loss_fn, device, max_batches=args.limit_test_batches)
        if not args.no_save and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint_path, epoch + 1, model, optimizer)
    print("Done!")

    if not args.no_save:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"Saved PyTorch Model State to {args.model_path}")
        save_checkpoint(args.checkpoint_path, args.epochs, model, optimizer)

        loaded_model = NeuralNetwork().to(device)
        loaded_model.load_state_dict(
            torch.load(args.model_path, weights_only=True, map_location=device)
        )
        predict_one(loaded_model, test_data, device)
    else:
        predict_one(model, test_data, device)


if __name__ == "__main__":
    main()
