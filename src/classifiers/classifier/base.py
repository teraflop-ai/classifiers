from abc import ABC, abstractmethod

import torch


class BaseModel(ABC, torch.nn.Module):
    pass
