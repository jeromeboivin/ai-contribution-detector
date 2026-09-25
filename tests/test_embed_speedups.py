import numpy as np
import pytest
import torch

from aicontrib.features.embed import CodeEmbedder, plan_batches, precision_matches


def test_plan_batches_respects_the_token_budget_and_covers_everything():
    lengths = [512, 10, 300, 40, 512, 90, 7, 200]
    batches = plan_batches(lengths, budget=1024)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) * max(lengths[i] for i in batch) <= 1024 or len(batch) == 1
    firsts = [lengths[b[0]] for b in batches]
    assert firsts == sorted(firsts, reverse=True)  # longest first


def test_a_sample_longer_than_the_budget_still_gets_its_own_batch():
    assert plan_batches([5000, 3], budget=1024) == [[0], [1]]


def test_precision_check():
    ref = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
    assert precision_matches(ref, ref + 1e-5 * torch.randn(8, 256))[0]
    assert not precision_matches(ref, torch.full((8, 256), float("nan")))[0]
    assert not precision_matches(ref, torch.randn(8, 256))[0]  # drifted


def test_out_of_memory_splits_the_batch_and_keeps_order():
    emb = object.__new__(CodeEmbedder)

    class FakeTokenizer:
        def pad(self, features, return_tensors):
            return features

    emb.tokenizer = FakeTokenizer()

    def fake_embed_inputs(features):
        if len(features) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return np.array([[f["input_ids"][0]] for f in features], dtype=np.float32)

    emb._embed_inputs = fake_embed_inputs
    out = emb.embed_encoded([{"input_ids": [i]} for i in range(7)])
    assert out[:, 0].tolist() == list(range(7))


def test_out_of_memory_on_a_single_sample_is_reported():
    emb = object.__new__(CodeEmbedder)
    emb.tokenizer = type("T", (), {"pad": lambda self, f, return_tensors: f})()

    def always_oom(features):
        raise torch.cuda.OutOfMemoryError("simulated")

    emb._embed_inputs = always_oom
    with pytest.raises(torch.cuda.OutOfMemoryError):
        emb.embed_encoded([{"input_ids": [1]}])
