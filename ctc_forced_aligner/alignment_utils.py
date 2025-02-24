import math

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available

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


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


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
                    prev_seg.start
                    if (idx == 0)
                    else int((prev_seg.start + prev_seg.end) / 2)
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
    waveform, audio_sf = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = torch.mean(waveform, dim=0)

    if audio_sf != SAMPLING_FREQ:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=audio_sf, new_freq=SAMPLING_FREQ
        )
    waveform = waveform.to(dtype).to(device)
    return waveform


def generate_emissions(
    model,
    audio_waveform: torch.Tensor,
    window_length=30,
    context_length=2,
    batch_size=4,
):
    batch_size = max(batch_size, 1)
    window = int(window_length * SAMPLING_FREQ)
    if audio_waveform.size(0) < window:
        extension = 0
        context = 0
        input_tensor = audio_waveform.unsqueeze(0)
    else:
        # batching the input tensor and including a context
        # before and after the input tensor
        context = int(context_length * SAMPLING_FREQ)
        extension = math.ceil(
            audio_waveform.size(0) / window
        ) * window - audio_waveform.size(0)
        padded_waveform = torch.nn.functional.pad(
            audio_waveform, (context, context + extension)
        )
        input_tensor = padded_waveform.unfold(0, window + 2 * context, window)

    # Batched Inference
    emissions_arr = []
    with torch.inference_mode():
        for i in range(0, input_tensor.size(0), batch_size):
            input_batch = input_tensor[i : i + batch_size]
            emissions_ = model(input_batch).logits
            emissions_arr.append(emissions_)

    emissions = torch.cat(emissions_arr, dim=0)
    if context > 0:
        emissions = emissions[
            :,
            time_to_frame(context_length) : -time_to_frame(context_length) + 1,
        ]  # removing the context
    emissions = emissions.flatten(0, 1)

    if time_to_frame(extension / SAMPLING_FREQ) > 0:
        emissions = emissions[: -time_to_frame(extension / SAMPLING_FREQ)]

    emissions = torch.log_softmax(emissions, dim=-1)
    emissions = torch.cat(
        [emissions, torch.zeros(emissions.size(0), 1).to(emissions.device)], dim=1
    )  # adding a star token dimension
    stride = float(audio_waveform.size(0) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, math.ceil(stride)


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
        raise ValueError(
            f"targets Tensor shouldn't contain blank index. Found {targets}."
        )
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
    dictionary = {k.lower(): v for k, v in dictionary.items()}
    dictionary["<star>"] = len(dictionary)

    # Force Alignment
    token_indices = [
        dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
    ]

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
    if attn_implementation is None:
        if version.parse(transformers_version) < version.parse("4.41.0"):
            attn_implementation = "eager"
        elif (
            is_flash_attn_2_available()
            and device == "cuda"
            and dtype in [torch.float16, torch.bfloat16]
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

    model = (
        AutoModelForCTC.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer
