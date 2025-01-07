import os
import re
import subprocess
import unicodedata

import numpy as np

from .norm_config import norm_config

UROMAN_PATH = os.path.join(os.path.dirname(__file__), "uroman", "bin")


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations,
    removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code : ISO 639-3 code of the language
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += r"]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + r"]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases:
    #   a)text starts with a number
    #   b) a number is present somewhere in the middle of the text
    #   c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded
    # and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:
        digits_pattern = r"[" + config["digit_set"]

        digits_pattern += r"]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode

        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


# iso codes with specialized rules in uroman
special_isos_uroman = [
    "ara",
    "bel",
    "bul",
    "deu",
    "ell",
    "eng",
    "fas",
    "grc",
    "ell",
    "eng",
    "heb",
    "kaz",
    "kir",
    "lav",
    "lit",
    "mkd",
    "mkd2",
    "oss",
    "pnt",
    "pus",
    "rus",
    "srp",
    "srp2",
    "tur",
    "uig",
    "ukr",
    "yid",
]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts, iso=None):
    input_text = "\n".join(norm_transcripts) + "\n"

    assert os.path.exists(os.path.join(UROMAN_PATH, "uroman.pl")), "uroman not found"

    assert not subprocess.call(
        ["perl", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ), (
        "Please ensure that a valid perl installation exists,"
        " you can verify by running `perl --version` in your terminal"
    )

    cmd = ["perl", os.path.join(UROMAN_PATH, "uroman.pl")]
    if iso in special_isos_uroman:
        cmd.extend(["-l", iso])

    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    output_text = result.stdout

    outtexts = []
    for line in output_text.splitlines():
        line = " ".join(line.strip())
        line = re.sub(r"\s+", " ", line).strip()
        outtexts.append(line)

    assert len(outtexts) == len(norm_transcripts)

    uromans = [normalize_uroman(ot) for ot in outtexts]

    return uromans


def split_text(text: str, split_size: str = "word"):
    if split_size == "sentence":
        from nltk.tokenize import PunktSentenceTokenizer

        sentence_checker = PunktSentenceTokenizer()
        sentences = sentence_checker.sentences_from_text(text)
        return sentences

    elif split_size == "word":
        return text.split()
    elif split_size == "char":
        return list(text)


def preprocess_text(
    text, romanize, language, split_size="word", star_frequency="segment"
):
    assert split_size in [
        "sentence",
        "word",
        "char",
    ], "Split size must be sentence, word, or char"
    assert star_frequency in [
        "segment",
        "edges",
    ], "Star frequency must be segment or edges"
    if language in ["jpn", "chi"]:
        split_size = "char"
    text_split = split_text(text, split_size)
    norm_text = [text_normalize(line.strip(), language) for line in text_split]

    if romanize:
        tokens = get_uroman_tokens(norm_text, language)
    else:
        tokens = [" ".join(list(word)) for word in norm_text]

    # add <star> token to the tokens and text
    # it's used extensively here but I found that it produces more accurate results
    # and doesn't affect the runtime
    if star_frequency == "segment":
        tokens_starred = []
        [tokens_starred.extend(["<star>", token]) for token in tokens]

        text_starred = []
        [text_starred.extend(["<star>", chunk]) for chunk in text_split]

    elif star_frequency == "edges":
        tokens_starred = ["<star>"] + tokens + ["<star>"]
        text_starred = ["<star>"] + text_split + ["<star>"]

    return tokens_starred, text_starred


def merge_segments(segments, threshold=0.00):
    for i in range(len(segments) - 1):
        if segments[i + 1]["start"] - segments[i]["end"] < threshold:
            segments[i + 1]["start"] = segments[i]["end"]


def postprocess_results(
    text_starred: list,
    spans: list,
    stride: float,
    scores: np.ndarray,
    merge_threshold: float = 0.0,
):
    results = []

    for i, t in enumerate(text_starred):
        if t == "<star>":
            continue
        span = spans[i]
        seg_start_idx = span[0].start
        seg_end_idx = span[-1].end

        audio_start_sec = seg_start_idx * (stride) / 1000
        audio_end_sec = seg_end_idx * (stride) / 1000
        score = scores[seg_start_idx:seg_end_idx].sum()
        sample = {
            "start": audio_start_sec,
            "end": audio_end_sec,
            "text": t,
            "score": score.item(),
        }
        results.append(sample)

    merge_segments(results, merge_threshold)
    return results
