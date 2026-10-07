# -*- coding: utf-8 -*-
"""
Global configuration: all step scripts read paths and parameters from here.
To change parameters, edit this file only.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------- Directory paths ----------
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"        # place the full-text PDFs here
TEXT_DIR = DATA_DIR / "texts"      # step01 output: one .md per article
OUT_DIR = DATA_DIR / "output"      # all outputs of step02-08

# ---------- step02 sentence filtering ----------
MIN_WORDS = 6      # drop sentences shorter than 6 words (headings/fragments)
MAX_WORDS = 300    # drop sentences longer than 300 words (reference paper value)

# ---------- step03 embedding model ----------
# Same as the reference paper: all-MiniLM-L6-v2, 384-dim sentence vectors
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# If the model download fails (e.g., restricted network), uncomment the line
# below to use a HuggingFace mirror:
# import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- step04 k selection ----------
K_RANGE = range(2, 11)     # candidate cluster counts: 2 to 10
SILHOUETTE_SAMPLE = 4000   # max sentences sampled for silhouette (speed)

# ---------- step05 final clustering ----------
# After step04, fill in the chosen k here (e.g., 6), then run step05
FINAL_K = 5

# ---------- General ----------
RANDOM_SEED = 42           # fixed seed for reproducibility (reference paper practice)

# ---------- step06 stability validation ----------
BOOTSTRAP_ITERS = 1000     # 1000 bootstrap iterations, same as reference paper
LOPO = True                # leave-one-paper-out validation

# ---------- step07 representative sentences ----------
N_TOP_SENTENCES = 80       # top-80 centroid-closest sentences per cluster
                           # (>=5% of every cluster at k=5; reference paper used 30)

# ---------- step08 UMAP ----------
UMAP_DIMS = 2
