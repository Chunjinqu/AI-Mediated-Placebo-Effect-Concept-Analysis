# -*- coding: utf-8 -*-
"""
step05: final K-means clustering (single layer)
Input:  data/output/embeddings.npy
Output: data/output/clusters.csv (sentences.csv + cluster label column)
        data/output/centroids.npy (cluster centroids)
Prerequisite: set FINAL_K in config.py to the k chosen in step04
Run:    python step05_cluster.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import OUT_DIR, FINAL_K, RANDOM_SEED


def main():
    if not FINAL_K:
        print("!! Run step04 first and fill the chosen k into FINAL_K in config.py.")
        return

    emb = np.load(OUT_DIR / "embeddings.npy")
    df = pd.read_csv(OUT_DIR / "sentences.csv")

    km = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(emb)

    df["cluster"] = labels
    df.to_csv(OUT_DIR / "clusters.csv", index=False, encoding="utf-8-sig")
    np.save(OUT_DIR / "centroids.npy", km.cluster_centers_)

    print(f"Done: k={FINAL_K}, sentences per cluster:")
    print(df["cluster"].value_counts().sort_index().to_string())
    print("\nSkim the sentences of each cluster in clusters.csv to confirm "
          "no cluster consists entirely of bibliography/table debris.")
    print("Next: run step06_stability.py (time-consuming)")


if __name__ == "__main__":
    main()
