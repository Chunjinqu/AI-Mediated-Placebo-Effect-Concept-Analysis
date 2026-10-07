# -*- coding: utf-8 -*-
"""
step06b: sub-cluster analysis of the concept-relevant clusters
Input:  data/output/embeddings.npy, clusters.csv, sentences.csv
Output: data/output/subclusters.csv (sentence-level sub-cluster labels)
        data/output/subcluster_stability.txt (bootstrap + LOPO per parent
        cluster, with pre-specified adoption gates)
        data/output/subcluster_representative.csv / .md (top SUB_N_TOP
        centroid-closest sentences per sub-cluster, with source article)
Run:    python step06b_subcluster.py  (after step06)

Pre-specified design (decided before inspection of sub-cluster results):
- Only concept-relevant clusters (CONCEPT_CLUSTERS in config.py) with
  >= SUB_MIN_SIZE sentences are eligible.
- Sub-cluster count k is chosen within SUB_K_RANGE by the silhouette
  coefficient on the 384-dim SBERT embeddings of the parent cluster
  (the reference paper likewise sub-clustered in the embedding space).
- Stability uses the same canonical clusterboot design as step06
  (papers resampled with replacement, duplicates kept, refit centroids
  mapped back to ALL sentences of the parent cluster, Hungarian-aligned
  ARI and sub-cluster-wise Jaccard) plus leave-one-paper-out (LOPO).
- Adoption gates: the sub-cluster solution is adopted only if bootstrap
  mean ARI >= SUB_ARI_GATE AND every sub-cluster Jaccard >=
  SUB_JACCARD_GATE. Otherwise the single-layer solution is retained and
  the outcome is reported transparently.
- Representative sentences: top SUB_N_TOP per sub-cluster by cosine
  similarity to the sub-cluster centroid in the 384-dim embedding space.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from config import (OUT_DIR, RANDOM_SEED, BOOTSTRAP_ITERS,
                    CONCEPT_CLUSTERS, SUB_MIN_SIZE, SUB_K_RANGE,
                    SUB_ARI_GATE, SUB_JACCARD_GATE, SUB_N_TOP)
from step06_stability import aligned_jaccards, summarize


def pick_k(emb_sub):
    """Choose sub-cluster count within SUB_K_RANGE by silhouette."""
    scores = {}
    for k in SUB_K_RANGE:
        lab = KMeans(n_clusters=k, random_state=RANDOM_SEED,
                     n_init=10).fit_predict(emb_sub)
        scores[k] = silhouette_score(emb_sub, lab)
    best = max(scores, key=scores.get)
    return best, scores


def stability(emb_sub, ref, articles_sub, k, tag, lines):
    """Canonical clusterboot + LOPO within one parent cluster."""
    uniq = np.unique(articles_sub)
    art_ids = {a: np.where(articles_sub == a)[0] for a in uniq}
    n_art = len(art_ids)
    rng = np.random.default_rng(RANDOM_SEED)

    def fit_and_map(fit_idx):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(emb_sub[fit_idx])
        return km.predict(emb_sub)

    aris, all_jac = [], []
    for it in range(BOOTSTRAP_ITERS):
        picked = rng.choice(n_art, n_art, replace=True)
        idx = np.concatenate([art_ids[uniq[i]] for i in picked])
        new = fit_and_map(idx)
        aris.append(adjusted_rand_score(ref, new))
        all_jac.append(aligned_jaccards(ref, new, k))
        if (it + 1) % 200 == 0:
            print(f"  {tag} bootstrap {it+1}/{BOOTSTRAP_ITERS} ...")
    summarize(f"{tag} Bootstrap", aris, all_jac, k, lines)

    aris_l, all_jac_l = [], []
    for a in uniq:
        new = fit_and_map(np.where(articles_sub != a)[0])
        aris_l.append(adjusted_rand_score(ref, new))
        all_jac_l.append(aligned_jaccards(ref, new, k))
    summarize(f"{tag} LOPO", aris_l, all_jac_l, k, lines)

    return float(np.mean(aris)), np.nanmean(np.asarray(all_jac), axis=0)


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")
    if emb.shape[0] != len(df):
        raise ValueError("Row mismatch between embeddings.npy and "
                         "clusters.csv. Rerun step05 first.")

    lines = [f"concept_clusters={CONCEPT_CLUSTERS}  "
             f"min_size={SUB_MIN_SIZE}  k_range={SUB_K_RANGE}  "
             f"bootstrap_iters={BOOTSTRAP_ITERS}",
             "design=clusterboot within each parent cluster "
             "(compare on full parent cluster)"]
    label_rows, rep_rows = [], []

    for c in CONCEPT_CLUSTERS:
        idx = np.where(df["cluster"].to_numpy() == c)[0]
        lines.append("")
        lines.append(f"===== Cluster {c} (n={len(idx)}) =====")
        if len(idx) < SUB_MIN_SIZE:
            lines.append(f"skipped: below pre-specified size gate "
                         f"({len(idx)} < {SUB_MIN_SIZE})")
            continue

        emb_sub = emb[idx]
        art_sub = df["article"].to_numpy()[idx]

        k, sil = pick_k(emb_sub)
        lines.append("k selection (silhouette): "
                     + ", ".join(f"k={kk}: {v:.3f}" for kk, v in sil.items())
                     + f"  ->  k={k}")

        ref = KMeans(n_clusters=k, random_state=RANDOM_SEED,
                     n_init=10).fit_predict(emb_sub)
        sizes = np.bincount(ref, minlength=k)
        lines.append("sub-cluster sizes: "
                     + ", ".join(f"S{i}={n}" for i, n in enumerate(sizes)))

        mean_ari, jac = stability(emb_sub, ref, art_sub, k, f"C{c}", lines)

        adopted = (mean_ari >= SUB_ARI_GATE
                   and np.all(jac >= SUB_JACCARD_GATE))
        lines.append(f"gates: mean ARI {mean_ari:.2f} "
                     f"(>= {SUB_ARI_GATE}) and min Jaccard {jac.min():.2f} "
                     f"(>= {SUB_JACCARD_GATE})  ->  "
                     f"{'ADOPTED' if adopted else 'NOT ADOPTED'}")
        if not adopted:
            lines.append("single-layer solution retained for this cluster; "
                         "sub-cluster results reported for transparency only")

        label_rows.append(pd.DataFrame({
            "sent_id": df["sent_id"].to_numpy()[idx],
            "article": art_sub,
            "cluster": c,
            "subcluster": ref,
        }))

        # representative sentences (cosine to centroid in 384-dim space)
        for s in range(k):
            loc = np.where(ref == s)[0]
            e = emb_sub[loc]
            cen = e.mean(axis=0)
            sim = e @ cen / (np.linalg.norm(e, axis=1)
                             * np.linalg.norm(cen))
            take = min(SUB_N_TOP, len(loc))
            for rank, li in enumerate(np.argsort(-sim)[:take], 1):
                gi = idx[loc[li]]
                rep_rows.append({
                    "cluster": c, "subcluster": s, "rank": rank,
                    "similarity": round(float(sim[li]), 4),
                    "article": df["article"].iloc[gi],
                    "text": df["text"].iloc[gi],
                })

    if label_rows:
        pd.concat(label_rows).to_csv(OUT_DIR / "subclusters.csv",
                                     index=False)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT_DIR / "subcluster_representative.csv", index=False)
    with open(OUT_DIR / "subcluster_representative.md", "w",
              encoding="utf-8") as f:
        for (c, s), g in rep.groupby(["cluster", "subcluster"]):
            f.write(f"\n## Cluster {c} / Sub-cluster {s} "
                    f"(top {len(g)})\n\n")
            for _, r in g.iterrows():
                f.write(f"{r['rank']}. [{r['article']}] {r['text']}\n\n")

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "subcluster_stability.txt").write_text(report,
                                                      encoding="utf-8")
    print(f"\nSaved to {OUT_DIR / 'subcluster_stability.txt'}")
    print("Next: run step07_representative.py / step08_umap.py if needed")


if __name__ == "__main__":
    main()
