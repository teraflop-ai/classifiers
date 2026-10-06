from torch import nn
from transformers import AutoModelForSequenceClassification

from classifiers.classifier.base import BaseModel


class BinaryProbe(BaseModel):
    def __init__(self, in_dim):
        super().__init__()
        self.probe = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.probe(x).squeeze(-1)


class MultilabelProbe(BaseModel):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.probe = nn.Linear(in_dim, num_labels)

    def forward(self, x):
        return self.probe(x)


class MulticlassProbe(BaseModel):
    pass


class Model2vecModel(BaseModel):
    pass


class BertClassifier(BaseModel):
    def __init__(self, name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels
        )

    def forward(self, **inputs):
        return self.model(**inputs).logits


class JevModel(BaseModel):
    pass


class XGBoostModel(BaseModel):
    pass


class Classifier(BaseModel):
    pass
