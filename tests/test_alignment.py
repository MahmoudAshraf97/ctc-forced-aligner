import numpy as np
import pytest
import torch
import torchaudio.functional as F

from ctc_forced_aligner import alignment_utils
from ctc_forced_aligner.alignment_utils import forced_align


@pytest.mark.parametrize(
    "logprobs_size, vocab_size, targets_size",
    [
        (l, t, int(l * ratio))
        for l in range(100, 60101, 20000)
        for t in range(30, 41, 10)
        for ratio in [0.4, 0.6]
    ],
)
def test_alignment(logprobs_size, vocab_size, targets_size):
    blank = 0
    targets = torch.randint(blank + 1, vocab_size, (1, targets_size))
    logprobs = torch.randn((1, logprobs_size, vocab_size + 1))
    logprobs = 6.5 * logprobs - 13  # same distribution as default model logits

    ctc_alignment = forced_align(logprobs.numpy(), targets.numpy())
    torch_alignment = F.forced_align(logprobs, targets)

    num_mismatches = np.sum(ctc_alignment[0] != torch_alignment[0].numpy())
    assert num_mismatches <= 1


def test_get_alignments_uses_emission_vocab_size_for_star(monkeypatch):
    class DummyTokenizer:
        pad_token_id = 1

        def get_vocab(self):
            return {
                "<blank>": 0,
                "<pad>": 1,
                "a": 2,
                "b": 3,
                "|": 4,
            }

    captured = {}

    def fake_forced_align(log_probs, targets, blank=0, **kwargs):
        captured["log_probs_shape"] = log_probs.shape
        captured["targets"] = targets.copy()
        captured["blank"] = blank
        return targets.copy(), np.zeros((targets.shape[-1],), dtype=np.float32)

    monkeypatch.setattr(alignment_utils, "forced_align", fake_forced_align)

    emissions = torch.zeros((8, 5), dtype=torch.float32)
    alignment_utils.get_alignments(emissions, ["<star>", "a"], DummyTokenizer())

    assert captured["targets"].tolist() == [[4, 2]]
    assert captured["log_probs_shape"][-1] == 5
    assert captured["blank"] == 0
