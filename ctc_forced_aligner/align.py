UROMAN_PATH = "..uroman/bin"
import torch
import os
from .alignment_helpers import (
    load_alignment_model,
    load_audio,
    generate_emissions,
    get_alignments,
    get_spans,
    preprocess_text,
    postprocess_results,
)

TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", help="path of the audio file", required=True)
    parser.add_argument(
        "--text_path", help="path of the text to be aligned", required=True
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        required=True,
        help="Language in ISO 639-3 code. Identifying the input as Arabic, Belarusian,"
        " Bulgarian, English, Farsi, German, Ancient Greek, Modern Greek, Pontic Greek"
        ", Hebrew, Kazakh, Kyrgyz, Latvian, Lithuanian, North Macedonian, Russian, "
        "Serbian, Turkish, Ukrainian, Uyghur, Mongolian, Thai, Javanese or Yiddish "
        "will improve romanization for those languages, No effect for other languages.",
    )

    parser.add_argument(
        "--romanize",
        action="store_false",
        default=False,
        help="Enable romanization for non-latin scripts. "
        "Use if you are using a model that doesn't support your language vocabulary.",
    )

    parser.add_argument(
        "--split_size",
        type=str,
        default="word",
        choices=["sentence", "word", "char"],
        help="Whether to align on a sentence, word, or character level.",
    )
    parser.add_argument(
        "--star_frequency",
        type=str,
        default="edges",
        choices=["segment", "edges"],
        help="The frequency of the <star> token in the text."
        "Star token increases the accuracy of the alignment but also increases segment fragmentation."
        "segment adds <star> token after each segment."
        "edges adds <star> token at the start and end of the text."
        "use --merge_threshold to merge segments that are closer than the threshold.",
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.00,
        help="merge segments that are closer than the threshold."
        "used to remove very small time differences between segments.",
    )

    parser.add_argument(
        "--alignment_model",
        default="MahmoudAshraf/mms-300m-1130-forced-aligner",
        help="Name of the CTC (Wav2Vec2/HuBERT/MMS) model to use for alignment,"
        " you can choose a language-specific model or an "
        "english model along with --romanize flag to support all languages."
        " accepts Huggingface model names or local pathes.",
    )

    # compute related arguments
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for alignment model inference. Helps with speed and memory usage.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for  inference.",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="ًWindow size in seconds to chunk the audio file for alignment.",
    )

    parser.add_argument(
        "--context_size",
        type=int,
        default=2,
        help="ًOverlab between chunks in seconds.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for the model.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    args = parser.parse_args()

    model, tokenizer, dictionary = load_alignment_model(
        args.device,
        args.alignment_model,
        args.attn_implementation,
        TORCH_DTYPES[args.compute_dtype],
    )

    audio_waveform = load_audio(args.audio_path, model.dtype, model.device)
    emissions, stride = generate_emissions(
        model, audio_waveform, args.window_size, args.context_size, args.batch_size
    )

    with open(args.text_path, "r") as f:
        lines = f.readlines()
    text = "".join(line for line in lines).replace("\n", " ").strip()

    tokens_starred, text_starred = preprocess_text(
        text, args.split_size, args.language, args.romanize, args.star_frequency
    )

    segments, blank_id = get_alignments(
        emissions,
        tokens_starred,
        dictionary,
    )

    spans = get_spans(tokens_starred, segments, tokenizer.decode(blank_id))

    results = postprocess_results(text_starred, spans, stride, args.merge_threshold)

    # write the results to a file
    with open(f"{os.path.splitext(args.audio_path)[0]}.txt", "w") as f:
        for result in results:
            f.write(f"{result['start']}-{result['end']}: {result['text']}\n")


if __name__ == "__main__":
    cli()
