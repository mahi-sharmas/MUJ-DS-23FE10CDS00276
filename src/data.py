"""Step 1 of the pipeline: download books, clean them, cut them into chunks,
and split into train/test (one whole unseen book per author for testing)."""
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.features import normalize_punctuation
from src.utils import ensure_dir, load_config


def download_book(book_id, url_template, raw_dir):
    """Download one Gutenberg book (skips the download if already saved)."""
    path = Path(raw_dir) / f"{book_id}.txt"
    if not path.exists():
        url = url_template.format(id=book_id)
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
        time.sleep(1)                      # be polite to Gutenberg's servers
    return path.read_text(encoding="utf-8", errors="ignore")


def get_title(raw_text):
    """Read the 'Title:' line from the Gutenberg header (used as a sanity check)."""
    match = re.search(r"^Title:\s*(.+)$", raw_text, flags=re.M)
    return match.group(1).strip() if match else "UNKNOWN"


def clean_text(raw_text):
    """Keep only the book itself and remove formatting that isn't the author's style."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw_text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw_text)
    text = raw_text[start.end() if start else 0: end.start() if end else len(raw_text)]
    text = re.sub(r"\[Illustration[^\]]*\]", " ", text)                   # picture captions
    text = re.sub(r"^\s*(CHAPTER|Chapter|BOOK|Book|PART|Part)\b.*$", " ", text, flags=re.M)  # headings
    text = text.replace("_", "")                                            # _italics_ markers
    return normalize_punctuation(text)


def chunk_words(text, size):
    """Cut text into consecutive chunks of exactly `size` words (leftover dropped)."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size + 1, size)]


def build_dataset(cfg):
    """Download every book in the config and return one row per chunk."""
    d = cfg["data"]
    raw_dir = ensure_dir(d["raw_dir"])
    rows = []
    for author, info in d["authors"].items():
        books = [(b, "train") for b in info["train"]] + [(info["test"], "test")]
        for book_id, split in books:
            raw = download_book(book_id, d["source_url"], raw_dir)
            chunks = chunk_words(clean_text(raw), d["chunk_words"])
            print(f"{author:10s} {split:5s} {book_id:>6} {len(chunks):4d} chunks  {get_title(raw)[:45]}")
            rows += [{"text": c, "author": author, "book_id": book_id, "split": split, "position": i}
                     for i, c in enumerate(chunks)]
    return pd.DataFrame(rows)


def cap_chunks(df, cfg):
    """Randomly keep at most N chunks per train book and per author's test book."""
    d, seed = cfg["data"], cfg["project"]["random_seed"]
    parts = []
    for _, g in df[df.split == "train"].groupby("book_id"):
        parts.append(g.sample(min(len(g), d["max_train_chunks_per_book"]), random_state=seed))
    for _, g in df[df.split == "test"].groupby("author"):
        parts.append(g.sample(min(len(g), d["max_test_chunks_per_author"]), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


if __name__ == "__main__":
    cfg = load_config()
    df_all = build_dataset(cfg)
    df = cap_chunks(df_all, cfg)
    out = ensure_dir("data") / "chunks.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} chunks to {out}\n")
    print(pd.crosstab(df["author"], df["split"], margins=True))
