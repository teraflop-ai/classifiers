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
        self.model = model.to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = loss

        self.num_epochs = num_epochs
        self.train_loader = train_loader

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="epochs"):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
            for batch in pbar:
                labels = batch.pop("labels").to(self.device)
                labels = (
                    labels.long()
                    if isinstance(self.loss, nn.CrossEntropyLoss)
                    else labels.float()
                )
                if "embeddings" in batch:
                    logits = self.model(batch["embeddings"].to(self.device))
                else:
                    logits = self.model(
                        **{k: v.to(self.device) for k, v in batch.items()}
                    )
                loss = self.loss(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())

    def save(self, path="model.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
