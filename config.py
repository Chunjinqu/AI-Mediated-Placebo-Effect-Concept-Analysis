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

# ---------- step06b sub-clustering (pre-specified, data-triggered) ----------
# Concept-relevant clusters as interpreted from the step05 k=5 solution
# (C2 Technology Acceptance, C3 Mental Health Chatbots, C4 GenAI/Placebo).
CONCEPT_CLUSTERS = [2, 3, 4]
SUB_MIN_SIZE = 800           # only clusters with >= 800 sentences are eligible
SUB_K_RANGE = (3, 4)         # candidate sub-cluster counts (silhouette picks)
SUB_ARI_GATE = 0.6           # adopt only if bootstrap mean ARI >= 0.6 ...
SUB_JACCARD_GATE = 0.5       # ... and every sub-cluster Jaccard >= 0.5

# ---------- step08 UMAP ----------
UMAP_DIMS = 2
