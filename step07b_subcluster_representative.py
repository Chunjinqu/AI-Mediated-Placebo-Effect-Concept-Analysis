# -*- coding: utf-8 -*-
"""
step07b: representative sentences per sub-cluster + Rodgers coding basis
Input:  data/output/embeddings.npy, clusters.csv, subclusters.csv,
        subcluster_stability.txt
Output: data/output/subcluster_representative.csv / .md
        data/output/rodgers_basis.csv / .md
Run:    python step07b_subcluster_representative.py  (after step06b)

Three products:

1. Sub-cluster representative sentences (transparency material). For each
   sub-cluster, sentences are ranked by cosine similarity to the sub-cluster
   centroid in the 384-dim SBERT embedding space, and at least SUB_TOP_PROP
   (5%) of the sub-cluster's sentences are exported (rounded up), each with
   its source article.

2. Rodgers coding basis. The adoption gates recorded in
   subcluster_stability.txt decide the basis per concept cluster:
   - ADOPTED: the cluster's sub-cluster representative sentences;
   - NOT ADOPTED (single-layer retained): the SUB_FALLBACK_TOP_N (80)
     centroid-closest sentences of the whole cluster, which is more than 5%
     of each concept cluster.

3. Saturation check. For every concept cluster, SUB_RANDOM_CHECK_N (40)
   additional sentences are drawn by seeded stratified random sampling
   (four centroid-similarity quartiles of the sentences not already in the
   basis, 10 per stratum). These are coded alongside the basis; whether they
   introduce new concept elements is reported as a formal sensitivity
   analysis of the centroid-based sampling rule.
"""
import math
import re

import numpy as np
import pandas as pd

from config import (OUT_DIR, CONCEPT_CLUSTERS, RANDOM_SEED,
                    SUB_TOP_PROP, SUB_FALLBACK_TOP_N, SUB_RANDOM_CHECK_N)


def read_adoption():
    """Parse subcluster_stability.txt: cluster id -> adopted (bool)."""
    txt = (OUT_DIR / "subcluster_stability.txt").read_text(encoding="utf-8")
    out = {}
    for chunk in re.split(r"(?====== Cluster )", txt):
        m = re.match(r"===== Cluster (\d+)", chunk)
        if not m:
            continue
        c = int(m.group(1))
        if re.search(r"->\s+NOT ADOPTED", chunk):
            out[c] = False
        elif re.search(r"->\s+ADOPTED", chunk):
            out[c] = True
    return out


def cluster_sims(emb, sent_idx):
    """Cosine similarity of every sentence to its cluster centroid."""
    e = emb[sent_idx]
    cen = e.mean(axis=0)
    return e @ cen / (np.linalg.norm(e, axis=1) * np.linalg.norm(cen))


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")
    sub = pd.read_csv(OUT_DIR / "subclusters.csv")
    if emb.shape[0] != len(df):
        raise ValueError("Row mismatch between embeddings.npy and "
                         "clusters.csv. Rerun step05 first.")
    adopted = read_adoption()
    for c in CONCEPT_CLUSTERS:
        if c not in adopted:
            raise ValueError(f"Cluster {c}: no adoption decision found in "
                             "subcluster_stability.txt. Rerun step06b first.")

    # ---------- 1. sub-cluster representative sentences ----------
    rows = []
    for (c, s), g in sub.groupby(["cluster", "subcluster"]):
        sent_idx = g["sent_id"].to_numpy()  # == row index in clusters.csv
        take = min(math.ceil(SUB_TOP_PROP * len(g)), len(g))
        sim = cluster_sims(emb, sent_idx)
        for rank, li in enumerate(np.argsort(-sim)[:take], 1):
            rows.append({
                "cluster": c, "subcluster": s, "rank": rank,
                "similarity": round(float(sim[li]), 4),
                "article": g["article"].iloc[li],
                "n_subcluster": len(g),
                "sent_id": int(sent_idx[li]),
                "text": df["text"].iloc[sent_idx[li]],
            })
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT_DIR / "subcluster_representative.csv", index=False)
    with open(OUT_DIR / "subcluster_representative.md", "w",
              encoding="utf-8") as f:
        for (c, s), g in rep.groupby(["cluster", "subcluster"]):
            n = g["n_subcluster"].iloc[0]
            f.write(f"\n## Cluster {c} / Sub-cluster {s} "
                    f"({len(g)} of {n} sentences, "
                    f"{len(g)/n:.1%})\n\n")
            for _, r in g.iterrows():
                f.write(f"{r['rank']}. [{r['article']}] {r['text']}\n\n")

    # ---------- 2. Rodgers coding basis + 3. saturation check ----------
    basis_rows = []
    for c in CONCEPT_CLUSTERS:
        sent_idx = np.where(df["cluster"].to_numpy() == c)[0]
        sim = cluster_sims(emb, sent_idx)
        if adopted[c]:
            g = rep[rep["cluster"] == c]
            used = set(g["sent_id"])
            for _, r in g.iterrows():
                basis_rows.append({
                    "cluster": c,
                    "basis": f"sub-cluster {r['subcluster']}",
                    "rank": r["rank"],
                    "similarity": r["similarity"],
                    "article": r["article"],
                    "n_cluster": len(sent_idx),
                    "sent_id": r["sent_id"],
                    "text": r["text"],
                })
        else:
            take = min(SUB_FALLBACK_TOP_N, len(sent_idx))
            order = np.argsort(-sim)[:take]
            used = set(int(sent_idx[li]) for li in order)
            for rank, li in enumerate(order, 1):
                si = sent_idx[li]
                basis_rows.append({
                    "cluster": c,
                    "basis": "cluster (single-layer fallback)",
                    "rank": rank,
                    "similarity": round(float(sim[li]), 4),
                    "article": df["article"].iloc[si],
                    "n_cluster": len(sent_idx),
                    "sent_id": int(si),
                    "text": df["text"].iloc[si],
                })
        # seeded stratified random saturation check over the remaining
        # sentences: four similarity quartiles, equal draws per stratum
        rng = np.random.default_rng(RANDOM_SEED + c)
        rem = np.array([li for li in range(len(sent_idx))
                        if int(sent_idx[li]) not in used])
        rem = rem[np.argsort(-sim[rem])]  # peripheral coverage by quartiles
        strata = np.array_split(rem, 4)
        per = SUB_RANDOM_CHECK_N // len(strata)
        picked = np.concatenate([
            rng.choice(s, min(per, len(s)), replace=False) for s in strata])
        picked = picked[np.argsort(-sim[picked])]
        for rank, li in enumerate(picked, 1):
            si = sent_idx[li]
            basis_rows.append({
                "cluster": c,
                "basis": "random saturation check",
                "rank": rank,
                "similarity": round(float(sim[li]), 4),
                "article": df["article"].iloc[si],
                "n_cluster": len(sent_idx),
                "sent_id": int(si),
                "text": df["text"].iloc[si],
            })
    basis = pd.DataFrame(basis_rows)
    basis.to_csv(OUT_DIR / "rodgers_basis.csv", index=False)
    with open(OUT_DIR / "rodgers_basis.md", "w", encoding="utf-8") as f:
        for (c, b), g in basis.groupby(["cluster", "basis"]):
            n = g["n_cluster"].iloc[0]
            f.write(f"\n## Cluster {c} — {b} "
                    f"({len(g)} of {n} sentences, {len(g)/n:.1%})\n\n")
            for _, r in g.iterrows():
                f.write(f"{r['rank']}. [{r['article']}] {r['text']}\n\n")

    print(f"Sub-cluster representatives: {len(rep)} sentences across "
          f"{rep.groupby(['cluster', 'subcluster']).ngroups} sub-clusters "
          f"(>= {SUB_TOP_PROP:.0%} each)")
    for c in CONCEPT_CLUSTERS:
        g = basis[basis["cluster"] == c]
        n_fb = int((g["basis"] != "random saturation check").sum())
        n_rc = int((g["basis"] == "random saturation check").sum())
        print(f"Cluster {c}: adoption={adopted[c]}, "
              f"Rodgers basis = {n_fb} sentences + {n_rc} random-check")
    print(f"Saved to {OUT_DIR / 'subcluster_representative.csv'} and "
          f"{OUT_DIR / 'rodgers_basis.csv'}")
    print("Next: run step08_umap.py and step08b_umap_subclusters.py")


if __name__ == "__main__":
    main()
