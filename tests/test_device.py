import pytest
import torch

from aicontrib import device
from aicontrib.config import load_config
from aicontrib.features.embed import CodeEmbedder


def test_old_torch_is_rejected_with_the_fix_command(monkeypatch):
    # PyTorch's cu121 index stops at 2.5.1, which transformers can't load our weights with.
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
    with pytest.raises(RuntimeError) as exc:
        CodeEmbedder(load_config())
    assert "2.6" in str(exc.value) and device.TORCH_INSTALL_HINT in str(exc.value)


def test_supported_torch_passes(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu126")
    device.require_supported_torch()
