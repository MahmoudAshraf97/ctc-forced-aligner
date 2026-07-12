import numpy as np
import pytest
import torch
import torchaudio.functional as F

from ctc_forced_aligner.alignment_utils import forced_align


def assert_matches_torchaudio(logprobs, targets):
    ctc_paths, ctc_scores = forced_align(logprobs.numpy(), targets.numpy())
    torch_paths, torch_scores = F.forced_align(logprobs, targets)

    np.testing.assert_array_equal(ctc_paths, torch_paths.numpy())
    np.testing.assert_array_equal(ctc_scores, torch_scores.squeeze(0).numpy())


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


@pytest.mark.parametrize(
    "frames, target_values",
    [
        (21, list(range(1, 16))),
        (1, [1]),
        (15, list(range(1, 16))),
        (7, [1, 1, 2, 2, 3]),
        (155, [(index % 30) + 1 for index in range(86)]),
        (300, [(index % 30) + 1 for index in range(80)]),
    ],
)
def test_tight_trellis_matches_torchaudio(frames, target_values):
    generator = torch.Generator().manual_seed(frames + len(target_values))
    targets = torch.tensor([target_values], dtype=torch.int64)
    logprobs = torch.randn((1, frames, 32), generator=generator).log_softmax(dim=-1)

    assert_matches_torchaudio(logprobs, targets)
