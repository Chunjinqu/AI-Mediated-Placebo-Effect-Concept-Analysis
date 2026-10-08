# -*- coding: utf-8 -*-
"""
step06: cluster stability validation (bootstrap + leave-one-paper-out)
Input:  data/output/embeddings.npy, clusters.csv
Output: data/output/stability.txt (overall ARI mean/median with 95% CI
        and IQR; cluster-wise Jaccard mean/median; valid iteration counts)
Run:    python step06_stability.py

Canonical resampling design (Hennig-style clusterboot):
- Bootstrap: papers are resampled WITH replacement; a paper drawn m times
  contributes its sentences m times to the fitting set (no de-duplication).
- After refitting K-means on the resampled (or LOPO-reduced) fitting set,
  the new centroids assign labels to ALL sentences of the original corpus
  (nearest centroid). Agreement metrics are therefore always computed on
  the same fixed comparison set: the full original corpus. This gives
  Jaccard = 0 an unambiguous meaning: the original cluster dissolved.
 - ARI is invariant to cluster-label permutation and is computed directly.
  Cluster-wise Jaccard is computed after matching refit clusters to the
  reference solution using the Hungarian algorithm.
- Interpretation thresholds (ARI >= 0.6, Jaccard >= 0.5) are commonly used
  rules of thumb, not formal statistical tests.
- The reported 95% CI is the 2.5th-97.5th percentile range of the metric
  across resampling iterations (percentile bootstrap interval).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from config import OUT_DIR, FINAL_K, RANDOM_SEED, BOOTSTRAP_ITERS, LOPO


def aligned_jaccards(orig, new, k):
    """Align labels with the Hungarian algorithm, then compute per-cluster
    Jaccard. Both label vectors must cover the same sentences."""
    conf = np.zeros((k, k), dtype=int)
    for a, b in zip(orig, new):
        conf[a, b] += 1
    row, col = linear_sum_assignment(-conf)
    out = []
    for i in range(k):
        j = col[np.where(row == i)[0][0]]
        inter = conf[i, j]
        union = (orig == i).sum() + (new == j).sum() - inter
        out.append(inter / union if union else 0.0)
    return out


def fit_and_map(emb, fit_idx):
    """Fit K-means on the fitting subset, then assign every sentence of the
    full corpus to the nearest new centroid (fixed comparison set)."""
    km = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
    km.fit(emb[fit_idx])
    return km.predict(emb)


def summarize(tag, aris, all_jac, k, lines):
    aris = np.asarray(aris, dtype=float)
    if len(aris) == 0:
        lines.append(f"[{tag}] no valid iterations")
        return
    lo, hi = np.percentile(aris, [2.5, 97.5])
    q1, q3 = np.percentile(aris, [25, 75])
    lines.append(f"[{tag}] valid iterations = {len(aris)}")
    lines.append(f"[{tag}] mean ARI = {aris.mean():.2f}, "
                 f"95% CI [{lo:.2f}, {hi:.2f}]")
    lines.append(f"[{tag}] median ARI = {np.median(aris):.2f}, "
                 f"IQR [{q1:.2f}, {q3:.2f}]")
    jac_arr = np.asarray(all_jac, dtype=float)
    jac = np.nanmean(jac_arr, axis=0)
    lines.append(f"[{tag}] cluster-wise Jaccard = "
                 + ", ".join(f"C{i}={v:.2f}" for i, v in enumerate(jac)))
    jac_med = np.nanmedian(jac_arr, axis=0)
    lines.append(f"[{tag}] cluster-wise Jaccard (median) = "
                 + ", ".join(f"C{i}={v:.2f}" for i, v in enumerate(jac_med)))


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")
    if emb.shape[0] != len(df):
        raise ValueError(
            f"Row mismatch: embeddings.npy has {emb.shape[0]} rows but "
            f"clusters.csv has {len(df)}. Rerun step05 before step06.")
    if not np.isfinite(emb).all():
        raise ValueError("embeddings.npy contains NaN or Inf values.")
    if not (isinstance(FINAL_K, int) and 2 <= FINAL_K < len(df)):
        raise ValueError(f"FINAL_K={FINAL_K} is not a valid cluster count "
                         f"for {len(df)} sentences.")

    orig = df["cluster"].to_numpy()
    articles = df["article"].to_numpy()
    uniq_arts = np.unique(articles)
    art_ids = {a: np.where(articles == a)[0] for a in uniq_arts}
    n_art = len(art_ids)

    lines = [f"FINAL_K={FINAL_K}  bootstrap_iters={BOOTSTRAP_ITERS}  "
             f"n_articles={n_art}  n_sentences={len(df)}",
             "design=clusterboot (fit on resample with duplicates; "
             "compare on full corpus)"]
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------- bootstrap: resample articles WITH replacement ----------
    aris, all_jac = [], []
    for it in range(BOOTSTRAP_ITERS):
        picked = rng.choice(n_art, n_art, replace=True)
        # duplicates kept: a paper drawn m times enters the fit m times
        idx = np.concatenate([art_ids[uniq_arts[i]] for i in picked])
        new = fit_and_map(emb, idx)
        aris.append(adjusted_rand_score(orig, new))
        all_jac.append(aligned_jaccards(orig, new, FINAL_K))
        if (it + 1) % 100 == 0:
            print(f"bootstrap {it+1}/{BOOTSTRAP_ITERS} ...")
    summarize("Bootstrap", aris, all_jac, FINAL_K, lines)

    # ---------- LOPO: leave one article out at a time ----------
    if LOPO:
        aris_l, all_jac_l = [], []
        for a, idx_a in art_ids.items():
            idx = np.where(articles != a)[0]
            new = fit_and_map(emb, idx)
            aris_l.append(adjusted_rand_score(orig, new))
            all_jac_l.append(aligned_jaccards(orig, new, FINAL_K))
        summarize("LOPO", aris_l, all_jac_l, FINAL_K, lines)

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "stability.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to {OUT_DIR / 'stability.txt'}")
    print("Interpretation (rules of thumb, not formal tests): ARI>=0.6 "
          "suggests a stable solution; clusters with Jaccard<0.5 should be "
          "interpreted cautiously and reported in the limitations.")
    print("Next: run step06b_subcluster.py")


if __name__ == "__main__":
    main()
