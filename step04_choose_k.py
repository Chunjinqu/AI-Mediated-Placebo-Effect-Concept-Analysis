# -*- coding: utf-8 -*-
"""
step04: choose the optimal number of clusters k (elbow method + silhouette)
Input:  data/output/embeddings.npy
Output: data/output/k_selection.png (two-panel figure)
        data/output/k_selection.csv (inertia and silhouette per k)
Run:    python step04_choose_k.py
Reading: pick a k whose silhouette is a local peak and whose inertia curve
         has entered the flat "elbow" region; fill it into FINAL_K in
         config.py, then run step05.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import OUT_DIR, K_RANGE, SILHOUETTE_SAMPLE, RANDOM_SEED


def main():
    emb = np.load(OUT_DIR / "embeddings.npy")
    n = len(emb)
    print(f"Embedding matrix: {emb.shape}")

    # For large corpora, silhouette is computed on a random subsample for
    # speed (elbow inertia still uses the full matrix)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = (rng.choice(n, SILHOUETTE_SAMPLE, replace=False)
                  if n > SILHOUETTE_SAMPLE else np.arange(n))

    inertias, sils = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(emb)
        s = silhouette_score(emb[sample_idx], labels[sample_idx])
        inertias.append(km.inertia_)
        sils.append(s)
        print(f"k={k:2d}  inertia={km.inertia_:.0f}  silhouette={s:.3f}")

    pd.DataFrame({"k": list(K_RANGE), "inertia": inertias,
                  "silhouette": sils}).to_csv(
        OUT_DIR / "k_selection.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(list(K_RANGE), inertias, marker="o")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia (elbow method)")
    ax[0].set_title("Elbow")
    ax[1].plot(list(K_RANGE), sils, marker="o", color="tab:orange")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("Silhouette")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_selection.png", dpi=200)

    best = int(np.array(list(K_RANGE))[np.argmax(sils)])
    print(f"\nHighest silhouette at k = {best} (decide jointly with the elbow "
          f"criterion, fill FINAL_K in config.py, then run step05)")
    print(f"Figure saved: {OUT_DIR / 'k_selection.png'}")


if __name__ == "__main__":
    main()
