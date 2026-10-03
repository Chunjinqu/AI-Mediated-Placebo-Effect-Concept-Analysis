# -*- coding: utf-8 -*-
"""
step06: cluster stability validation (bootstrap + leave-one-paper-out)
Input:  data/output/embeddings.npy, clusters.csv
Output: data/output/stability.txt (overall ARI mean, 95% CI; cluster-wise Jaccard)
Run:    python step06_stability.py

Metrics (as in the reference paper):
- ARI (adjusted Rand index): agreement between two cluster assignments,
  corrected for chance; closer to 1 = more stable. Values >0.6 are generally
  considered stable, 0.4-0.6 moderate.
- Jaccard: per-cluster membership overlap (|intersection| / |union|).
- K-means cluster labels are arbitrary; resampled labels are aligned to the
  original solution with the Hungarian algorithm before comparison.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from config import OUT_DIR, FINAL_K, RANDOM_SEED, BOOTSTRAP_ITERS, LOPO


def aligned_jaccards(orig, new, k):
    """Align labels with the Hungarian algorithm, then compute per-cluster
    Jaccard. Returns a list of length k."""
    mask = ~(pd.isna(orig) | pd.isna(new))
    o, nw = orig[mask].to_numpy(), new[mask].to_numpy()
    conf = np.zeros((k, k), dtype=int)
    for a, b in zip(o, nw):
        conf[a, b] += 1
    row, col = linear_sum_assignment(-conf)
    out = []
    for i in range(k):
        j = col[np.where(row == i)[0][0]]
        inter = conf[i, j]
        union = (o == i).sum() + (nw == j).sum() - inter
        out.append(inter / union if union else np.nan)
    return out


def fit_on_subset(emb, idx):
    km = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
    km.fit(emb[idx])
    return km.labels_


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "clusters.csv")
    orig = df["cluster"].to_numpy()
    articles = df["article"].to_numpy()
    uniq_arts = np.unique(articles)
    art_ids = {a: np.where(articles == a)[0] for a in uniq_arts}
    n_art = len(art_ids)

    lines = [f"FINAL_K={FINAL_K}  bootstrap_iters={BOOTSTRAP_ITERS}  "
             f"n_articles={n_art}  n_sentences={len(df)}"]
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------- bootstrap: resample articles with replacement ----------
    aris, all_jac = [], []
    for it in range(BOOTSTRAP_ITERS):
        picked = rng.choice(n_art, n_art, replace=True)
        idx = np.concatenate([art_ids[uniq_arts[i]] for i in picked])
        idx = np.unique(idx)
        if len(np.unique(orig[idx])) < 2:
            continue
        new = fit_on_subset(emb, idx)
        aris.append(adjusted_rand_score(orig[idx], new))
        all_jac.append(aligned_jaccards(
            pd.Series(orig[idx]), pd.Series(new), FINAL_K))
        if (it + 1) % 20 == 0:
            print(f"bootstrap {it+1}/{BOOTSTRAP_ITERS} ...")
    aris = np.array(aris)
    ci = np.percentile(aris, [2.5, 97.5])
    lines.append(f"[Bootstrap] mean ARI = {aris.mean():.2f}, "
                 f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    jac = np.nanmean(np.array(all_jac), axis=0)
    lines.append("[Bootstrap] cluster-wise Jaccard = "
                 + ", ".join(f"C{i}={v:.2f}" for i, v in enumerate(jac)))

    # ---------- LOPO: leave one article out at a time ----------
    if LOPO:
        aris_l, all_jac_l = [], []
        for a, _ in art_ids.items():
            mask = articles != a
            idx = np.where(mask)[0]
            new = fit_on_subset(emb, idx)
            aris_l.append(adjusted_rand_score(orig[mask], new))
            all_jac_l.append(aligned_jaccards(
                pd.Series(orig[mask]), pd.Series(new), FINAL_K))
        aris_l = np.array(aris_l)
        ci = np.percentile(aris_l, [2.5, 97.5])
        lines.append(f"[LOPO] mean ARI = {aris_l.mean():.2f}, "
                     f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
        jac_l = np.nanmean(np.array(all_jac_l), axis=0)
        lines.append("[LOPO] cluster-wise Jaccard = "
                     + ", ".join(f"C{i}={v:.2f}" for i, v in enumerate(jac_l)))

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "stability.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to {OUT_DIR / 'stability.txt'}")
    print("Interpretation: ARI>=0.6 indicates a stable solution; for any "
          "cluster with Jaccard<0.5, interpret its findings cautiously and "
          "report this in the limitations (cf. Sub-cluster 3 = 0.52 in the "
          "reference paper).")
    print("Next: run step07_representative.py")


if __name__ == "__main__":
    main()
