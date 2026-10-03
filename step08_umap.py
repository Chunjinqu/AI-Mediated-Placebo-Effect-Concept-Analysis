# -*- coding: utf-8 -*-
"""
step08: UMAP 2-D visualization (figure for the paper)
Input:  embeddings.npy, clusters.csv
Output: data/output/umap_clusters.png
Run:    python step08_umap.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from config import OUT_DIR, RANDOM_SEED, UMAP_DIMS


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")

    reducer = umap.UMAP(n_components=UMAP_DIMS, random_state=RANDOM_SEED)
    xy = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(df["cluster"].unique()):
        mask = (df["cluster"] == c).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.5,
                   label=f"Cluster {c}")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("UMAP projection of sentence embeddings")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_clusters.png", dpi=300)
    print(f"Done: {OUT_DIR / 'umap_clusters.png'}")
    print("All steps complete. Result files are in data/output/.")


if __name__ == "__main__":
    main()
