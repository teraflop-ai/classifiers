import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        loss,
        num_epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cuda",
    ):
        self.device = device
        self.base_model = model.to(device)
        self.model = torch.compile(self.base_model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = loss

        self.num_epochs = num_epochs
        self.train_loader = train_loader

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
            for batch in pbar:
                inputs, labels = (
                    batch["embeddings"].to(self.device),
                    batch["labels"].to(self.device),
                )
                labels = (
                    labels.long()
                    if isinstance(self.loss, nn.CrossEntropyLoss)
                    else labels.float()
                )
                loss = self.loss(self.model(inputs), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())

    def save(self, path="model.pt"):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path="model.pt"):
        self.base_model.load_state_dict(torch.load(path, map_location=self.device))
