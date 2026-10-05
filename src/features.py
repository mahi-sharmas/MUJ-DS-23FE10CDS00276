"""Step 2 of the pipeline: turn text into numbers that describe writing STYLE
(sentence length, punctuation, function words) and mask character names."""
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

# NLTK's sentence splitter needs this model; download it once if missing
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Small, topic-independent words that authors use unconsciously
FUNCTION_WORDS = """the of and to a in that it is was he she i you for with as his her
but not be on at by this had have from or which they we all an were so my me
are their there what if would been one no will when who him them could some then
upon shall thus yet also very should may might must such own than only those these
whom where while though because nor""".split()

PUNCTUATION = {"comma": ",", "semicolon": ";", "colon": ":", "exclaim": "!",
               "question": "?", "dash": "--", "paren": "(", "quote": '"', "apostrophe": "'"}

WORD_RE = re.compile(r"[A-Za-z']+")

# Curly quotes / long dashes depend on who typed up the e-book, not on the author
PUNCT_MAP = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                           "\u2014": "--", "\u2013": "--"})


def normalize_punctuation(text):
    """Turn typographic quotes/dashes into plain ASCII so every book is measured the same way."""
    return text.translate(PUNCT_MAP)


def style_features(text):
    """Return ~90 style measurements for one piece of text (rates are per 100 words)."""
    words = WORD_RE.findall(text.lower())
    n = max(len(words), 1)
    sent_lens = [len(s.split()) for s in sent_tokenize(text)] or [0]
    feats = {
        "avg_sent_len": float(np.mean(sent_lens)),
        "std_sent_len": float(np.std(sent_lens)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "type_token_ratio": len(set(words)) / n,
    }
    for name, mark in PUNCTUATION.items():
        feats[f"p_{name}"] = text.count(mark) / n * 100
    counts = Counter(words)
    for w in FUNCTION_WORDS:
        feats[f"fw_{w}"] = counts[w] / n * 100
    return feats


def style_matrix(texts):
    """Apply style_features to many texts -> one row per text."""
    return pd.DataFrame([style_features(t) for t in texts])


def find_names(texts, min_count=3, cap_ratio=0.9):
    """Words that are (almost) always capitalised = probably names/places."""
    total, capital = Counter(), Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-z]+", t):
            total[w.lower()] += 1
            capital[w.lower()] += w[0].isupper()
    return {w for w, c in total.items()
            if c >= min_count and len(w) > 1 and capital[w] / c > cap_ratio}


def mask_names(text, names):
    """Replace every name-like word with 'Xxx' so the model can't use them."""
    return re.sub(r"[A-Za-z]+", lambda m: "Xxx" if m.group().lower() in names else m.group(), text)


if __name__ == "__main__":
    from src.utils import load_config
    cfg = load_config()
    df = pd.read_csv("data/chunks.csv")
    X = style_matrix(df["text"])
    print("Style feature matrix:", X.shape)

    names = find_names(df.loc[df.split == "train", "text"],
                       cfg["features"]["name_min_count"], cfg["features"]["name_cap_ratio"])
    print(f"{len(names)} name-like words, e.g.", sorted(names)[:15])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    cols = ["avg_sent_len", "avg_word_len", "p_semicolon", "p_dash", "p_exclaim", "fw_upon", "fw_very"]
    print(X[cols].groupby(df["author"]).mean().round(2))
