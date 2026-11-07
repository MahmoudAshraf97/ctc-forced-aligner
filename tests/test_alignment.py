import numpy as np
import pytest
import torch
import torchaudio.functional as F

from ctc_forced_aligner.alignment_utils import forced_align, get_alignments


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


class _Tokenizer:
    vocab_size = 4
    pad_token_id = 0

    def get_vocab(self):
        return {"<blank>": 0, "a": 1, "b": 2, "c": 3}


@pytest.mark.parametrize(
    "tokens, expected_path",
    [
        (["a"], [1]),
        (["a a b"], [1, 0, 1, 2]),
    ],
    ids=["single_frame", "exact_fit_with_repeat"],
)
def test_frames_that_exactly_fit_the_targets(tokens, expected_path):
    emissions = torch.full((len(expected_path), 5), -1.6)

    segments, scores, blank = get_alignments(emissions, tokens, _Tokenizer())

    path = [
        token
        for segment in segments
        for token in [segment.label] * (segment.end - segment.start + 1)
    ]
    assert path == [{0: blank, 1: "a", 2: "b", 3: "c"}[index] for index in expected_path]
    assert scores.shape == (len(expected_path),)


def test_too_few_frames_are_still_rejected():
    with pytest.raises(RuntimeError, match="too long for CTC"):
        get_alignments(torch.full((3, 5), -1.6), ["a a b"], _Tokenizer())
