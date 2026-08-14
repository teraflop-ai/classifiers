import argparse

import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from map_labels import encode_labels
from models import BinaryClassifier, MultiClassifier
from trainer import Trainer


def main(
    dataset_name: str,
    batch_size: int = 64,
    save_path: str = "model.pt",
    num_labels: int = 1,
    task: str = "binary",
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
):
    train_ds = load_dataset(dataset_name, split="train")
    train_ds, label2id = encode_labels(train_ds, task)
    train_ds = train_ds.with_format("torch")

    num_labels = len(label2id)
    in_dim = len(train_ds[0]["embeddings"])

    if task == "binary":
        model = BinaryClassifier(in_dim)
        loss = nn.BCEWithLogitsLoss()
    elif task == "multilabel":
        model = MultiClassifier(in_dim, num_labels)
        loss = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        model = MultiClassifier(in_dim, num_labels)
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Use one of selected available tasks.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        loss=loss,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )
    trainer.train()
    trainer.save(path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", default="model.pt")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--task", choices=["binary", "multilabel", "multiclass"], default="binary"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(**vars(args))
