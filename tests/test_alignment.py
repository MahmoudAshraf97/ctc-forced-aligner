import torch
from ctc_forced_aligner.alignment_utils import forced_align
import torchaudio.functional as F
import numpy as np
import pytest

@pytest.mark.parametrize("logprobs_size, vocab_size, targets_size", [
    (l, t, int(l * ratio))
    for l in range(100, 60101, 20000)
    for t in range(30, 41, 10)
    for ratio in [0.4, 0.6]
])
def test_alignment(logprobs_size, vocab_size, targets_size):
    blank = 0
    targets = torch.randint(blank + 1, vocab_size, (1, targets_size))
    logprobs = torch.randn((1, logprobs_size, vocab_size + 1))

    ctc_alignment = forced_align(logprobs.numpy(), targets.numpy())
    torch_alignment = F.forced_align(logprobs, targets)

    assert np.allclose(ctc_alignment[0], torch_alignment[0].numpy())
    