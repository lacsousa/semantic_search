import argparse
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

SEED = 42
DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "healthqa_processed.parquet"
EMBEDDINGS = DATA_DIR / "healthqa_embeddings.npy"  # linhas alinhadas com PROCESSED
EMBED_MODEL = "bge-m3"  # multilíngue; `ollama pull bge-m3`
BATCH_SIZE = 32
CLUSTERS = DATA_DIR / "healthqa_kmeans.npz"  # labels + centroids
DEFAULT_K = 50


def load_healthqa() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED)


def embed_texts(texts: list[str], model: str = EMBED_MODEL, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Embeddings L2-normalizados (float32), de modo que produto interno == cosseno."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"embedding ({model})"):
        out.extend(ollama.embed(model=model, input=texts[i : i + batch_size])["embeddings"])
    vecs = np.asarray(out, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def load_or_create_embeddings(hq: pd.DataFrame, model: str = EMBED_MODEL) -> np.ndarray:
    if EMBEDDINGS.exists():
        vecs = np.load(EMBEDDINGS)
        if len(vecs) == len(hq):
            return vecs
    vecs = embed_texts(hq["questão"].tolist(), model)
    np.save(EMBEDDINGS, vecs)
    return vecs


def load_or_create_kmeans(vecs: np.ndarray, k: int = DEFAULT_K) -> tuple[np.ndarray, np.ndarray]:
    """K-means sobre embeddings normalizados. Retorna (labels, centroides L2-normalizados)."""
    if CLUSTERS.exists():
        saved = np.load(CLUSTERS)
        if saved["centroids"].shape[0] == k and len(saved["labels"]) == len(vecs):
            return saved["labels"], saved["centroids"]
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(vecs)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    np.savez(CLUSTERS, labels=km.labels_, centroids=centroids)
    return km.labels_, centroids


def describe_clusters(hq: pd.DataFrame, vecs: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> None:
    """Tamanho de cada cluster e a questão mais próxima do centroide."""
    for c in np.argsort(-np.bincount(labels)):
        idx = np.flatnonzero(labels == c)
        rep = idx[np.argmax(vecs[idx] @ centroids[c])]
        print(f"cluster {c:>3} | n={len(idx):>4} | {hq['questão'].iloc[rep][:100]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="número de clusters")
    args = parser.parse_args()

    hq = load_healthqa()
    vecs = load_or_create_embeddings(hq, args.model)
    print(f"embeddings: {vecs.shape} -> {EMBEDDINGS}")

    labels, centroids = load_or_create_kmeans(vecs, args.k)
    describe_clusters(hq, vecs, labels, centroids)


if __name__ == "__main__":
    main()
