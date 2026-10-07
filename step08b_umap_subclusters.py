# -*- coding: utf-8 -*-
"""
step08b: UMAP visualization of sub-clusters (figure for the paper)
Input:  data/output/embeddings.npy, clusters.csv, subclusters.csv
Output: data/output/umap_subclusters.png (one panel per concept-relevant
        parent cluster; same full-corpus UMAP projection as step08,
        sub-cluster points coloured, remaining corpus in light grey)
Run:    python step08b_umap_subclusters.py  (after step06b)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from config import OUT_DIR, RANDOM_SEED, UMAP_DIMS, CONCEPT_CLUSTERS


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")
    sub = pd.read_csv(OUT_DIR / "subclusters.csv")
    if emb.shape[0] != len(df):
        raise ValueError("Row mismatch between embeddings.npy and "
                         "clusters.csv. Rerun step05 first.")

    reducer = umap.UMAP(n_components=UMAP_DIMS, random_state=RANDOM_SEED)
    xy = reducer.fit_transform(emb)

    parents = [c for c in CONCEPT_CLUSTERS
               if c in sub["cluster"].unique()]
    fig, axes = plt.subplots(1, len(parents),
                             figsize=(6 * len(parents), 5.5))
    if len(parents) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, c in zip(axes, parents):
        ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.15, color="lightgrey")
        g = sub[sub["cluster"] == c]
        for s in sorted(g["subcluster"].unique()):
            si = g.loc[g["subcluster"] == s, "sent_id"].to_numpy()
            ax.scatter(xy[si, 0], xy[si, 1], s=4, alpha=0.6,
                       color=cmap(s % 10), label=f"Sub-cluster {s}")
        ax.legend(markerscale=3, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Cluster {c} sub-clusters (n={len(g)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_subclusters.png", dpi=300)
    print(f"Done: {OUT_DIR / 'umap_subclusters.png'}")


if __name__ == "__main__":
    main()
