import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader

from trainer import Trainer


def main(
    dataset_name: str,
    batch_size: int = 64,
    save_path: str = "model.pt",
    num_labels: int = 1,
    binary: bool = False,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
):
    train_ds = load_dataset(dataset_name, split="train").with_format("torch")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    in_dim = len(train_ds[0]["features"])
    trainer = Trainer(
        in_dim=in_dim,
        train_loader=train_loader,
        num_labels=num_labels,
        binary=binary,
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
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(**vars(args))