import torch

from aicontrib.model.classifier import MLPClassifier


def test_forward_shape():
    model = MLPClassifier(input_dim=256, hidden_dims=[128, 64], num_classes=3, dropout=0.2)
    x = torch.randn(8, 256)
    logits = model(x)
    assert logits.shape == (8, 3)
