import argparse

import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from map_labels import encode_labels
from models import BertClassifier
from trainer import Trainer


def main(
    dataset_name: str,
    model_name: str,
    batch_size: int = 16,
    save_path: str = "modernbert.pt",
    num_epochs: int = 3,
    lr: float = 2e-5,
    max_length: int = 8192,
    device: str = "cuda",
):
    ds = load_dataset(dataset_name, split="train").rename_column("level", "labels")
    ds, label2id = encode_labels(ds, "multiclass")

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = ds.map(
        lambda b: tok(b["content"], truncation=True, max_length=max_length),
        batched=True,
    )
    ds = ds.remove_columns(
        [
            c
            for c in ds.column_names
            if c not in ("input_ids", "attention_mask", "labels")
        ]
    )

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=DataCollatorWithPadding(tok)
    )

    model = BertClassifier(model_name, len(label2id))
    trainer = Trainer(model, loader, nn.CrossEntropyLoss(), num_epochs, lr, device)
    trainer.train()
    trainer.save(save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--save_path", default="modernbert.pt")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    main(**vars(p.parse_args()))
