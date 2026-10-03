# AI-Mediated Placebo Effect in Healthcare — NLP Pipeline for Concept Analysis

Rodgers' evolutionary concept analysis integrated with NLP, in a dual-track
full-corpus design (manual review and computational clustering both cover all
included articles; findings are integrated afterwards).

Methodology replicated from: Lee et al., *npj Digital Medicine* (2026),
DOI: 10.1038/s41746-026-03300-2.

## Requirements

- Python 3.11 (the reference paper used 3.11.2)
- Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

0. Place the full-text PDFs of the included articles in `data/pdfs/`
   (ASCII file names only).
1. `python step01_pdf_to_text.py` — PDF to Markdown text → `data/texts/`
2. `python step02_clean_segment.py` — cleaning + sentence segmentation
   → `data/output/sentences.csv`
3. `python step03_embed.py` — SBERT sentence embeddings (downloads the ~90 MB
   model on first run) → `data/output/embeddings.npy`
4. `python step04_choose_k.py` — elbow + silhouette diagnostics
   → `data/output/k_selection.png`; then fill the chosen k into `FINAL_K`
   in `config.py`
5. `python step05_cluster.py` — K-means clustering
   → `data/output/clusters.csv`
6. `python step06_stability.py` — bootstrap (1000 iterations) + LOPO
   stability validation → `data/output/stability.txt`
7. `python step07_representative.py` — top-30 centroid-closest representative
   sentences per cluster → `data/output/representative_sentences.md`
8. `python step08_umap.py` — UMAP visualization
   → `data/output/umap_clusters.png`

All parameters (paths, word-count filters, model name, k range, seed,
bootstrap iterations, number of representative sentences) are centralized in
`config.py`.

## Differences from the reference paper

- **Corpus and design**: 21 articles, all analyzed by both manual review and
  NLP (dual-track full-corpus design). NLP is positioned as a computational
  audit / methodological triangulation of the full corpus rather than a
  scale extension.
- **Cleaning**: one explicit addition to the reference pipeline — structural
  truncation of the References section (`drop_backmatter` in step02). The
  reference paper does not report how bibliography noise was handled; we
  truncate reference lists and copyright/ISSN metadata lines explicitly,
  which removed ~29% of non-content sentences and prevented bibliography
  junk clusters in our corpus.
- **Single-layer clustering**: sub-clustering is designed as data-triggered
  (pre-registered sentence-count and stability thresholds); if the thresholds
  are not met, the single-layer solution is retained and reported as such.
- **Bootstrap iterations**: 1000, identical to the reference paper
  (`BOOTSTRAP_ITERS` in `config.py`).

## Data availability

The full-text articles analyzed in this study are not publicly available due
to publisher copyright restrictions; they are accessible through the
respective journals and databases. The extracted text corpora and derived
analytical outputs (e.g., sentence embeddings and clustering results) are
available from the corresponding author on reasonable request. Accordingly,
`data/` is excluded from this repository via `.gitignore`.

## Code availability

This repository contains the full set of custom scripts for all analytic
steps: preprocessing, sentence embedding generation, clustering, stability
validation, representative-sentence extraction, and UMAP visualization.
