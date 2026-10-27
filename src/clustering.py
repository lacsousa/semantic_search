"""Etapa 3 (parte 1): seleção de k (cotovelo + silhouette), HDBSCAN e avaliação externa no Enare.

Uso: .venv/bin/python src/clustering.py
Ajusta nas questões únicas (sem duplicatas por chave_dup) para as duplicatas não pesarem em dobro.
"""
from pathlib import Path

import hdbscan
import matplotlib
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
EMB = Path("embeddings")
RES = Path("resultados")
FIG = Path("figuras")
KS = list(range(4, 31))


def carregar(nome: str, versao: str):
    X = np.load(EMB / f"{versao}_{nome}.npy")
    meta = pd.read_parquet(EMB / f"meta_{nome}.parquet")
    unico = ~meta["chave_dup"].duplicated().values  # primeira ocorrência de cada enunciado
    return X, meta, unico


def varrer_k(X):
    """Inércia e silhouette para cada k (K-Means, seed fixa)."""
    linhas = []
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        linhas.append({"k": k, "inercia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(linhas)


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    fig, eixos = plt.subplots(2, 2, figsize=(11, 7))
    todos = []

    # 1) Revalida: varredura de k para as versões (a) e (b), sem e com PCA(50)
    for j, versao in enumerate("ab"):
        X, meta, un = carregar("revalida", versao)
        Xu = X[un]
        print(f"[revalida/{versao}] únicas: {un.sum()} de {len(X)}")
        for tag, M in [("raw", Xu), ("pca50", PCA(50, random_state=SEED).fit_transform(Xu))]:
            df = varrer_k(M).assign(versao=versao, entrada=tag)
            todos.append(df)
            eixos[0, j].plot(df.k, df.inercia, label=tag)
            eixos[1, j].plot(df.k, df.silhouette, label=tag)
        eixos[0, j].set_title(f"Cotovelo – versão ({versao})")
        eixos[1, j].set_title(f"Silhouette – versão ({versao})")
        for i in (0, 1):
            eixos[i, j].set_xlabel("k"); eixos[i, j].legend()
    fig.tight_layout(); fig.savefig(FIG / "kmeans_k_revalida.png", dpi=130)
    varredura = pd.concat(todos)
    varredura.to_csv(RES / "kmeans_varredura_k.csv", index=False)
    print("\nTop 3 k por silhouette (cada versão/entrada):")
    print(varredura.sort_values("silhouette", ascending=False).groupby(["versao", "entrada"]).head(3)
          .sort_values(["versao", "entrada", "silhouette"], ascending=[True, True, False]).round(4).to_string(index=False))

    # 2) HDBSCAN (padrão: reduzir com UMAP 10D antes, pois densidade em 768D degenera)
    print("\n== HDBSCAN sobre UMAP(10D, cosseno) do Revalida ==")
    for versao in "ab":
        X, meta, un = carregar("revalida", versao)
        Z = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(X[un])
        for mcs in (10, 20, 40):
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(Z)
            nc = len(set(lab)) - (1 if -1 in lab else 0)
            print(f"  versão {versao} min_cluster_size={mcs}: {nc} clusters, ruído {np.mean(lab == -1):.1%}")

    # 3) Avaliação externa no Enare RM (rótulos reais): ARI/NMI de K-Means vs macroárea e vs group
    print("\n== K-Means no Enare RM vs rótulos reais (ARI / NMI) ==")
    linhas = []
    for versao in "ab":
        X, meta, un = carregar("enare", versao)
        for alvo, k in [("macroarea", 6), ("group", 37)]:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X[un])
            y = meta.loc[un, alvo].values
            linhas.append({"versao": versao, "alvo": alvo, "k": k,
                           "ARI": adjusted_rand_score(y, km.labels_), "NMI": normalized_mutual_info_score(y, km.labels_)})
    av = pd.DataFrame(linhas)
    av.to_csv(RES / "kmeans_enare_ari_nmi.csv", index=False)
    print(av.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
