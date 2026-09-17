import math

from types import SimpleNamespace

import pytest
import torch
import torchaudio.functional as F

from ctc_forced_aligner.alignment_utils import SAMPLING_FREQ, generate_emissions

RATIO = 320
WINDOW = SAMPLING_FREQ
CONTEXT = SAMPLING_FREQ // 5


class DummyCTCModel(torch.nn.Module):
    def __init__(self, fail_on_call=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(inputs_to_logits_ratio=RATIO)
        self.calls = []
        self.fail_on_call = fail_on_call

    def forward(self, values):
        self.calls.append(values.detach().clone())
        if len(self.calls) == self.fail_on_call:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        frame_count = values.size(1) // RATIO
        frames = values[:, : frame_count * RATIO].reshape(values.size(0), frame_count, RATIO)
        means = frames.mean(dim=-1)
        logits = torch.stack((means, -means, means.square() / 2), dim=-1)
        return SimpleNamespace(logits=logits)


def reference_generate_emissions(model, audio, batch_size):
    window = WINDOW
    context = CONTEXT
    context_frames = context // RATIO
    window_frames = window // RATIO

    if audio.size(0) < window:
        extension = 0
        context = 0
        input_tensor = audio.unsqueeze(0)
    else:
        extension = math.ceil(audio.size(0) / window) * window - audio.size(0)
        padded = torch.nn.functional.pad(audio, (context, context + extension))
        input_tensor = padded.unfold(0, window + 2 * context, window)

    batches = []
    with torch.inference_mode():
        for index in range(0, input_tensor.size(0), batch_size):
            batches.append(model(input_tensor[index : index + batch_size]).logits)
    emissions = torch.cat(batches)
    if context > 0:
        emissions = emissions[:, context_frames : context_frames + window_frames]
    emissions = emissions.flatten(0, 1)
    extension_frames = extension // RATIO
    if extension_frames > 0:
        emissions = emissions[:-extension_frames]
    emissions = torch.log_softmax(emissions, dim=-1)
    return torch.cat((emissions, torch.zeros(emissions.size(0), 1, device=emissions.device)), dim=1)


@pytest.mark.parametrize(
    "sample_count",
    [
        WINDOW // 2,
        WINDOW - 1,
        WINDOW,
        WINDOW + 1,
        2 * WINDOW - 321,
        2 * WINDOW - 320,
        2 * WINDOW - 319,
        2 * WINDOW - 1,
        2 * WINDOW,
        3 * WINDOW + 640,
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_generate_emissions_matches_reference(sample_count, batch_size):
    audio = torch.linspace(-1, 1, sample_count)
    expected = reference_generate_emissions(DummyCTCModel(), audio, batch_size)

    emissions, stride = generate_emissions(
        DummyCTCModel(),
        audio,
        window_length=1,
        context_length=0.2,
        batch_size=batch_size,
    )

    torch.testing.assert_close(emissions, expected, rtol=0, atol=0)
    assert emissions.dtype == torch.float32
    assert emissions.device == audio.device
    assert emissions.is_contiguous()
    assert torch.count_nonzero(emissions[:, -1]) == 0
    assert stride == 20.0


def test_subframe_extension_does_not_empty_emissions():
    audio = torch.zeros(2 * WINDOW - 1)
    emissions, _ = generate_emissions(DummyCTCModel(), audio, window_length=1, context_length=0.2)

    assert emissions.size(0) == math.ceil(audio.size(0) / RATIO)


def test_emissions_preserve_downstream_forced_alignment():
    audio = torch.linspace(-1, 1, 2 * WINDOW + RATIO)
    expected = reference_generate_emissions(DummyCTCModel(), audio, batch_size=2)
    actual, _ = generate_emissions(
        DummyCTCModel(), audio, window_length=1, context_length=0.2, batch_size=2
    )
    targets = torch.tensor([[1, 2, 1, 2]], dtype=torch.int64)

    expected_path, expected_scores = F.forced_align(expected.unsqueeze(0), targets)
    actual_path, actual_scores = F.forced_align(actual.unsqueeze(0), targets)

    torch.testing.assert_close(actual_path, expected_path, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_audio_is_processed_on_gpu_without_changing_output_device():
    audio = torch.linspace(-1, 1, 2 * WINDOW + RATIO, dtype=torch.float16)
    reference_model = DummyCTCModel().cuda().half()
    candidate_model = DummyCTCModel().cuda().half()
    expected = reference_generate_emissions(reference_model, audio.cuda(), batch_size=2).cpu()

    actual, _ = generate_emissions(
        candidate_model,
        audio,
        window_length=1,
        context_length=0.2,
        batch_size=2,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_audio_preserves_cuda_output_compatibility():
    audio = torch.linspace(-1, 1, 2 * WINDOW + RATIO, device="cuda", dtype=torch.float16)
    expected = reference_generate_emissions(DummyCTCModel().cuda().half(), audio, batch_size=2)

    actual, _ = generate_emissions(
        DummyCTCModel().cuda().half(),
        audio,
        window_length=1,
        context_length=0.2,
        batch_size=2,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cuda"


def test_oom_reduces_remaining_batches_without_recomputing_completed_windows():
    audio = torch.arange(8 * WINDOW, dtype=torch.float32)
    failing_model = DummyCTCModel(fail_on_call=2)
    expected, _ = generate_emissions(
        DummyCTCModel(), audio, window_length=1, context_length=0.2, batch_size=4
    )

    emissions, _ = generate_emissions(
        failing_model, audio, window_length=1, context_length=0.2, batch_size=4
    )

    torch.testing.assert_close(emissions, expected, rtol=0, atol=0)
    assert [batch.size(0) for batch in failing_model.calls] == [4, 4, 2, 2]
    torch.testing.assert_close(failing_model.calls[2], failing_model.calls[1][:2], rtol=0, atol=0)
    torch.testing.assert_close(failing_model.calls[3], failing_model.calls[1][2:], rtol=0, atol=0)


def test_oom_with_one_window_is_not_retried():
    model = DummyCTCModel(fail_on_call=1)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        generate_emissions(model, torch.zeros(WINDOW // 2), window_length=1)
    assert len(model.calls) == 1
