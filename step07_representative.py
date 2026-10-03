# -*- coding: utf-8 -*-
"""
step07: extract representative sentences per cluster (ranked by cosine
        similarity to the cluster centroid)
Input:  embeddings.npy, clusters.csv, centroids.npy
Output: data/output/representative_sentences.csv
        data/output/representative_sentences.md (grouped by cluster, for
        manual Rodgers coding)
Run:    python step07_representative.py

Manual coding guidance (Rodgers' three slots):
- attribute: is this sentence about what the concept IS (defining features)?
- antecedent: is it about conditions that precede or enable the concept?
- consequence: is it about outcomes that follow from the concept?
Note: a cluster is NOT a conceptual category; clustering only brings
      same-topic sentences to you — slot assignment is a human
      theoretical judgment.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import OUT_DIR, N_TOP_SENTENCES


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    cent = np.load(OUT_DIR / "centroids.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")

    reps = []
    for c in sorted(df["cluster"].unique()):
        idx = df.index[df["cluster"] == c].to_numpy()
        sims = cosine_similarity(emb[idx], cent[[c]])[:, 0]
        order = np.argsort(-sims)[:N_TOP_SENTENCES]
        for rank, j in enumerate(order, 1):
            row = df.iloc[idx[j]]
            reps.append({"cluster": c, "rank": rank,
                         "similarity": round(float(sims[j]), 4),
                         "article": row["article"], "text": row["text"]})

    out = pd.DataFrame(reps)
    out.to_csv(OUT_DIR / "representative_sentences.csv",
               index=False, encoding="utf-8-sig")

    # Markdown version for printing/reading
    lines = []
    for c, g in out.groupby("cluster"):
        n_in_cluster = (df["cluster"] == c).sum()
        lines.append(f"\n# Cluster {c} ({n_in_cluster} sentences in total; "
                     f"top representatives below)\n")
        for _, r in g.iterrows():
            lines.append(f"{r['rank']:>2}. [{r['article']}] {r['text']}")
    (OUT_DIR / "representative_sentences.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(f"Done: {len(df['cluster'].unique())} clusters x {N_TOP_SENTENCES} "
          f"sentences = {len(out)} representative sentences")
    print(f"Open {OUT_DIR / 'representative_sentences.md'} to start manual coding.")
    print("Next: run step08_umap.py to generate the figure.")


if __name__ == "__main__":
    main()
