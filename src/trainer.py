import torch
from torch import nn
from tqdm import tqdm

from models import LinearClassifier


class Trainer:
    def __init__(
        self,
        in_dim,
        train_loader,
        num_labels=1,
        binary=False,
        num_epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cuda",
    ):
        self.device = device
        self.base_model = LinearClassifier(in_dim, num_labels, binary).to(device)
        self.model = torch.compile(self.base_model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = nn.BCELoss() if binary else nn.BCEWithLogitsLoss()

        self.num_epochs = num_epochs
        self.train_loader = train_loader

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
            for batch in pbar:
                inputs, labels = (
                    batch["features"].to(self.device),
                    batch["labels"].to(self.device),
                )
                loss = self.loss(self.model(inputs), labels.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())

    def save(self, path="model.pt"):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path="model.pt"):
        self.base_model.load_state_dict(torch.load(path, map_location=self.device))
