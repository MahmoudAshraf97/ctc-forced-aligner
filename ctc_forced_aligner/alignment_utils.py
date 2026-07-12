import math

from dataclasses import dataclass
from subprocess import CalledProcessError, run
from typing import Optional, Tuple

import numpy as np
import torch

from transformers import AutoModelForCTC, AutoTokenizer

from .ctc_forced_aligner import forced_align as forced_align_cpp

SAMPLING_FREQ = 16000


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def get_spans(tokens, segments, blank):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == blank
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == blank:
            continue
        assert seg.label == ltr, f"{seg.label} != {ltr}"
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == blank:
                pad_start = (
                    prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end) / 2)
                )
                span = [Segment(blank, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == blank:
                pad_end = (
                    next_seg.end
                    if (idx == len(intervals) - 1)
                    else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(blank, span[-1].end, pad_end)]
        spans.append(span)
    return spans


def load_audio(audio_file: str, dtype: torch.dtype, device: str):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    audio_file: str
        The audio file to open

    dtype: torch.dtype
        The desired data type of the returned tensor

    device: str
        The device to place the returned tensor on

    Returns
    -------
    A PyTorch tensor containing the audio waveform, in requested dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", audio_file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLING_FREQ),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise ImportError("ffmpeg not found. Please ensure ffmpeg is installed and in PATH.")

    return torch.frombuffer(out, dtype=torch.int16).flatten().to(dtype).to(device) / 32768.0


def generate_emissions(
    model,
    audio_waveform: torch.Tensor,
    window_length=30,
    context_length=2,
    batch_size=4,
):
    """Generate CTC emissions while retaining only one inference batch at a time.

    The returned tensor remains on ``audio_waveform.device`` for compatibility. Keeping
    the waveform on CPU therefore bounds accelerator memory to the model, the current
    inference batch, and its logits.
    """
    batch_size = max(batch_size, 1)
    ratio = model.config.inputs_to_logits_ratio  # 320 for wav2vec2/MMS
    window = int(window_length * SAMPLING_FREQ)
    context = int(context_length * SAMPLING_FREQ)
    assert window % ratio == 0 and context % ratio == 0, (
        f"window/context must be multiples of the model stride (ratio={ratio / SAMPLING_FREQ}s)"
    )
    context_frames = context // ratio
    window_frames = window // ratio

    if audio_waveform.size(0) < window:
        extension = 0
        total_windows = 1
        total_frames = None
    else:
        total_windows = math.ceil(audio_waveform.size(0) / window)
        extension = total_windows * window - audio_waveform.size(0)
        total_frames = total_windows * window_frames - extension // ratio

    try:
        model_parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("The alignment model must have at least one parameter") from error
    model_device = model_parameter.device
    model_dtype = model_parameter.dtype
    output_device = audio_waveform.device
    effective_batch_size = batch_size
    emissions = None
    write_offset = 0
    first_window = 0

    with torch.inference_mode():
        while first_window < total_windows:
            window_count = min(effective_batch_size, total_windows - first_window)
            input_batch = None
            logits = None
            batch_emissions = None
            try:
                if audio_waveform.size(0) < window:
                    input_batch = audio_waveform.unsqueeze(0)
                else:
                    batch_start = first_window * window - context
                    batch_end = (first_window + window_count) * window + context
                    source_start = max(0, batch_start)
                    source_end = min(audio_waveform.size(0), batch_end)
                    input_batch = torch.nn.functional.pad(
                        audio_waveform[source_start:source_end],
                        (source_start - batch_start, batch_end - source_end),
                    )
                    input_batch = input_batch.unfold(0, window + 2 * context, window)

                input_batch = input_batch.to(device=model_device, dtype=model_dtype)
                logits = model(input_batch).logits

                if audio_waveform.size(0) >= window:
                    required_frames = context_frames + window_frames
                    if logits.size(1) < required_frames:
                        raise RuntimeError(
                            "Model returned too few frames for the configured window"
                        )
                    logits = logits[:, context_frames:required_frames]

                batch_emissions = torch.log_softmax(logits.flatten(0, 1), dim=-1)
            except torch.cuda.OutOfMemoryError:
                if window_count == 1:
                    raise
                effective_batch_size = max(1, window_count // 2)
                del input_batch, logits, batch_emissions
                if model_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if total_frames is None:
                total_frames = batch_emissions.size(0)
            if emissions is None:
                emissions = torch.empty(
                    (total_frames, batch_emissions.size(-1) + 1),
                    dtype=torch.float32,
                    device=output_device,
                )
                emissions[:, -1] = 0

            frame_count = min(batch_emissions.size(0), total_frames - write_offset)
            emissions[write_offset : write_offset + frame_count, :-1].copy_(
                batch_emissions[:frame_count].to(output_device, dtype=torch.float32)
            )
            write_offset += frame_count
            first_window += window_count
            del input_batch, logits, batch_emissions

    if emissions is None or write_offset != total_frames:
        raise RuntimeError(
            f"Emission assembly wrote {write_offset} frames; expected {total_frames}"
        )

    stride = ratio * 1000 / SAMPLING_FREQ

    return emissions, stride


def forced_align(
    log_probs: np.ndarray,
    targets: np.ndarray,
    input_lengths: Optional[np.ndarray] = None,
    target_lengths: Optional[np.ndarray] = None,
    blank: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Align a CTC label sequence to an emission.
    Args:
        log_probs (NDArray): log probability of CTC emission output.
            NDArray of shape `(B, T, C)`. where `B` is the batch size, `T` is the input length,
            `C` is the number of characters in alphabet including blank.
        targets (NDArray): Target sequence. NDArray of shape `(B, L)`,
            where `L` is the target length.
        input_lengths (NDArray or None, optional):
            Lengths of the inputs (max value must each be <= `T`). 1-D NDArray of shape `(B,)`.
        target_lengths (NDArray or None, optional):
            Lengths of the targets. 1-D NDArray of shape `(B,)`.
        blank_id (int, optional): The index of blank symbol in CTC emission. (Default: 0)

    Returns:
        Tuple(NDArray, NDArray):
            NDArray: Label for each time step in the alignment path computed using forced alignment.

            NDArray: Log probability scores of the labels for each time step.

    Note:
        The sequence length of `log_probs` must satisfy:


        .. math::
            L_{\text{log\_probs}} \ge L_{\text{label}} + N_{\text{repeat}}

        where :math:`N_{\text{repeat}}` is the number of consecutively repeated tokens.
        For example, in str `"aabbc"`, the number of repeats are `2`.

    Note:
        The current version only supports ``batch_size==1``.
    """
    if blank in targets:
        raise ValueError(f"targets Tensor shouldn't contain blank index. Found {targets}.")
    if blank >= log_probs.shape[-1] or blank < 0:
        raise ValueError("blank must be within [0, log_probs.shape[-1])")
    if np.max(targets) >= log_probs.shape[-1] and np.min(targets) >= 0:
        raise ValueError("targets values must be within [0, log_probs.shape[-1])")
    assert log_probs.dtype == np.float32, "log_probs must be float32"

    paths, scores = forced_align_cpp(
        log_probs,
        targets,
        blank,
    )
    return paths, scores


def get_alignments(
    emissions: torch.Tensor,
    tokens: list,
    tokenizer,
):
    assert len(tokens) > 0, "Empty transcript"

    dictionary = tokenizer.get_vocab()
    if len(dictionary) > tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary contains more tokens than expected. "
            "Please open an issue on Github to report this and include the model name."
        )
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    dictionary["<star>"] = len(dictionary)

    # Force Alignment
    token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]

    blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)

    if not emissions.is_cpu:
        emissions = emissions.cpu()
    targets = np.asarray([token_indices], dtype=np.int64)

    path, scores = forced_align(
        emissions.unsqueeze(0).float().numpy(),
        targets,
        blank=blank_id,
    )
    path = path.squeeze().tolist()

    idx_to_token_map = {v: k for k, v in dictionary.items()}
    segments = merge_repeats(path, idx_to_token_map)
    return segments, scores, idx_to_token_map[blank_id]


def load_alignment_model(
    device: str,
    model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    attn_implementation: str = None,
    dtype: torch.dtype = torch.float32,
):
    model = AutoModelForCTC.from_pretrained(model_path, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, word_delimiter_token=None)

    return model, tokenizer
