# -*- coding: utf-8 -*-
"""
step03: sentence embeddings (SBERT)
Input:  data/output/sentences.csv
Output: data/output/embeddings.npy (n_sentences x 384 matrix)
Run:    python step03_embed.py
Note: the first run downloads all-MiniLM-L6-v2 (~90 MB) automatically.
      If the download fails on a restricted network, set HF_ENDPOINT to a
      mirror in config.py or edit the line below.
"""
import os
# Must be set BEFORE importing sentence_transformers, otherwise the mirror
# has no effect; with unrestricted network access, delete this line or point
# it to https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import OUT_DIR, EMBED_MODEL


def main():
    csv = OUT_DIR / "sentences.csv"
    if not csv.exists():
        print("!! sentences.csv not found — run step02 first.")
        return

    df = pd.read_csv(csv)
    print(f"Loaded {len(df)} sentences; loading model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    # Batch encoding: each sentence -> 384-dim vector (semantically similar
    # sentences have nearby vectors)
    emb = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    out = OUT_DIR / "embeddings.npy"
    np.save(out, emb)
    print(f"Done: embedding matrix shape {emb.shape}, saved to {out}")
    print("Next: run step04_choose_k.py")


if __name__ == "__main__":
    main()
