<h1 align="center">Forced Alignment with Hugging Face CTC Models</h1>

[![FreePalestine.Dev](https://freepalestine.dev/header/1)](https://freepalestine.dev)


<p align="center">
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/actions/workflows/test_build.yml">
    <img src="https://github.com/MahmoudAshraf97/ctc-forced-aligner/actions/workflows/CI.yml/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/stargazers">
    <img src="https://img.shields.io/github/stars/MahmoudAshraf97/ctc-forced-aligner.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/issues">
        <img src="https://img.shields.io/github/issues/MahmoudAshraf97/ctc-forced-aligner.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/MahmoudAshraf97/ctc-forced-aligner.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2FMahmoudAshraf97%2Fctc-forced-aligner">
  <img src="https://img.shields.io/twitter/url/https/github.com/MahmoudAshraf97/ctc-forced-aligner.svg?style=social" alt="Twitter">
  </a> 
  </a>
 
</p>

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

This Python package provides an efficient way to perform forced alignment between text and audio using Hugging Face's pretrained models. It leverages the power of Wav2Vec2, HuBERT, and MMS models for accurate alignment, making it a powerful tool for creating speech corpuses.

### Features
- **Atleast 5X less memory usage:** Improved implementation to use much less memory than TorchAudio forced alignment API.
- **Wide range of language support:** Works with multiple languages including English, Arabic, Russian, German, and 1126 more languages.
- **Flexibility in alignment granularity:** Choose between aligning on a sentence, word, or character level.
- **Customizable alignment parameters:** Control the frequency of `<star>` token insertion, merge threshold for segment merging, and more.
- **Integration with Hugging Face's models:** Leverage the power of pretrained Wav2Vec2, HuBERT, and MMS models for accurate alignment.
- **GPU acceleration:** Utilize your GPU for faster inference.
- **Output in JSON format:** Provides clear and structured alignment results for easy analysis and integration.


### Installation

#### Latest version from GitHub
```bash
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```
#### Installing locally from source
```bash
git clone https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
cd ctc-forced-aligner
pip install -e .[dev]
```

### Usage

```bash
ctc-forced-aligner --audio_path "path/to/audio.wav" --text_path "path/to/text.txt" --language "eng" --romanize
```

<details>
<summary>Terminal Usage</summary>


### Arguments

| Argument | Description | Default |
|---|---|---|
| `--audio_path` | Path to the audio file | Required |
| `--text_path` | Path to the text file | Required |
| `--language` | Language in ISO 639-3 code | Required |
| `--romanize` | Enable romanization for non-latin scripts or for multilingual models regardless of the language, required when using the default model| False |
| `--split_size` | Alignment granularity: "sentence", "word", or "char" | "word" |
| `--star_frequency` | Frequency of `<star>` token: "segment" or "edges" | "edges" |
| `--merge_threshold` | Merge threshold for segment merging | 0.00 |
| `--alignment_model` | Name of the alignment model | [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) |
| `--compute_dtype` | Compute dtype for inference | "float32" |
| `--batch_size` | Batch size for inference | 4 |
| `--window_size` | Window size in seconds for audio chunking | 30 |
| `--context_size` | Overlap between chunks in seconds | 2 |
| `--attn_implementation` | Attention implementation | "eager" |
| `--device` | Device to use for inference: "cuda" or "cpu" | "cuda" if available, else "cpu" |

### Examples

```bash
# Align an English audio file with the text file
ctc-forced-aligner --audio_path "english_audio.wav" --text_path "english_text.txt" --language "eng" --romanize

# Align a Russian audio file with romanized text
ctc-forced-aligner --audio_path "russian_audio.wav" --text_path "russian_text.txt" --language "rus" --romanize

# Align on a sentence level
ctc-forced-aligner --audio_path "audio.wav" --text_path "text.txt" --language "eng" --split_size "sentence" --romanize

# Align using a model with native vocabulary
ctc-forced-aligner --audio_path "audio.wav" --text_path "text.txt" --language "ara" --alignment_model "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
```

</details>


<details>
<summary>Python Usage</summary>
  
### Python Usage
```python
import torch
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

audio_path = "your/audio/path"
text_path = "your/text/path"
language = "iso" # ISO-639-3 Language code
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16


alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)


with open(text_path, "r") as f:
    lines = f.readlines()
text = "".join(line for line in lines).replace("\n", " ").strip()

emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=batch_size
)

tokens_starred, text_starred = preprocess_text(
    text,
    romanize=True,
    language=language,
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)
```

</details>

### Output

The alignment results will be saved to a file containing the following information in JSON format:

- **`text`:** The aligned text.
- **`segments`:** A list of segments, each containing the start and end time of the corresponding text segment.
<details>
<summary>JSON</summary>

```json
{
  "text": "This is a sample text to be aligned with the audio.",
  "segments": [
    {
      "start": 0.000,
      "end": 1.234,
      "text": "This"
    },
    {
      "start": 1.234,
      "end": 2.567,
      "text": "is"
    },
    {
      "start": 2.567,
      "end": 3.890,
      "text": "a"
    },
    {
      "start": 3.890,
      "end": 5.213,
      "text": "sample"
    },
    {
      "start": 5.213,
      "end": 6.536,
      "text": "text"
    },
    {
      "start": 6.536,
      "end": 7.859,
      "text": "to"
    },
    {
      "start": 7.859,
      "end": 9.182,
      "text": "be"
    },
    {
      "start": 9.182,
      "end": 10.405,
      "text": "aligned"
    },
    {
      "start": 10.405,
      "end": 11.728,
      "text": "with"
    },
    {
      "start": 11.728,
      "end": 13.051,
      "text": "the"
    },
    {
      "start": 13.051,
      "end": 14.374,
      "text": "audio."
    }
  ]
}
```
</details>



### Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

### License

This project is licensed under the BSD License, note that the default model has CC-BY-NC 4.0 License, so make sure to use a different model for commercial usage.

### Acknowledgements

This project is based on the work of FAIR MMS team.

