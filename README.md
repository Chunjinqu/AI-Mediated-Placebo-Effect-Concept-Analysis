# AI-Mediated Placebo Effect in Healthcare — NLP Pipeline for Concept Analysis

Rodgers’ evolutionary concept analysis integrated with natural language processing (NLP), using a dual-track, full-corpus design: all included articles undergo both manual concept analysis and computational clustering, followed by an integrated interpretation of the findings.

Methodological reference: Lee et al. npj Digital Medicine (2026). https://doi.org/10.1038/s41746-026-03300-2.

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
7. `python step06b_subcluster.py` — sub-clustering of concept-relevant
   clusters with the same stability validation and pre-specified adoption
   gates → `data/output/subclusters.csv`, `subcluster_stability.txt`
8. `python step08_umap.py` — UMAP visualization
   → `data/output/umap_clusters.png`
9. `python step08b_umap_subclusters.py` — UMAP panels of the sub-cluster
   structure within each concept-relevant cluster
   → `data/output/umap_subclusters.png`

All parameters (paths, word-count filters, model name, k range, seed,
bootstrap iterations, sub-clustering gates) are centralized in
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
- **Single-layer clustering**: sub-clustering is designed as data-triggered; if the thresholds
  are not met, the single-layer solution is retained and reported as such.
- **Bootstrap iterations**: 1000, identical to the reference paper
  (`BOOTSTRAP_ITERS` in `config.py`).

## Data availability

The full-text articles analyzed in this study are subject to the copyright and access conditions of their respective publishers and databases. Full-text PDFs and extracted full-text corpora are therefore not included in this repository.

The repository provides the analysis code and configuration files. Aggregate analytical outputs and other derived materials may be shared where permitted by applicable copyright restrictions and licensing terms. Materials containing copyrighted text will not be distributed unless such sharing is legally permitted.

## Code availability

This repository contains the full set of custom scripts for all analytic
steps: preprocessing, sentence embedding generation, clustering, stability
validation, sub-cluster analysis, and UMAP visualization.
