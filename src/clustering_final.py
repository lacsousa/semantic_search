"""Etapa 3 (parte 2): K-Means final (k=8, versão (a), PCA 50), nomes dos clusters e UMAP 2D.

Uso: .venv/bin/python src/clustering_final.py
Saídas: resultados/clusters_revalida.parquet, resultados/cluster_nomes.csv, resultados/modelos/*.joblib,
        figuras/umap_clusters_macroarea.png
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from areas import NOMES_CLUSTERS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
K = 8
VERSAO = "a"
EMB, RES, FIG = Path("embeddings"), Path("resultados"), Path("figuras")

# stopwords pt-BR + vocabulário de enunciado clínico que não distingue tema
STOP = set("""a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre entre até ao aos à às
e ou mas se que como quando qual quais quem onde já não nem também mais menos muito muitos pouco só ser é são foi era
está estão estava estavam tem têm tinha há ter seu sua seus suas ele ela eles elas isso isto este esta estes estas
esse essa esses essas aquele aquela lhe lhes me te nos vos cujo cuja num numa pelo pela pelos pelas ainda após antes
durante desde apenas cerca então assim porém contudo onde dia dias mês meses ano anos idade
paciente pacientes caso quadro conduta médico médica assinale alternativa correta correto melhor diagnóstico
provável principal seguinte seguintes indicado indicada apresenta apresentou procura procurou atendimento
exame físico exames realizado realizada realizados feito consulta história relata refere queixa
masculino feminino sexo homem mulher criança menino menina anos de idade""".split())


def main():
    (RES / "modelos").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    X = np.load(EMB / f"{VERSAO}_revalida.npy")
    meta = pd.read_parquet(EMB / "meta_revalida.parquet")
    Xe = np.load(EMB / f"{VERSAO}_enare.npy")
    meta_e = pd.read_parquet(EMB / "meta_enare.parquet")
    un = ~meta["chave_dup"].duplicated().values

    # ajuste só nas questões únicas; depois todas (inclusive duplicatas) recebem cluster
    pca = PCA(50, random_state=SEED).fit(X[un])
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(pca.transform(X[un]))
    joblib.dump(pca, RES / "modelos/pca50_a.joblib")
    joblib.dump(km, RES / "modelos/kmeans_k8_a.joblib")
    meta["cluster_id"] = km.predict(pca.transform(X))
    print("tamanho dos clusters (todas as 1.777):", meta["cluster_id"].value_counts().sort_index().to_dict())

    # nomes: TF-IDF com cada cluster como um "documento" (termos frequentes no cluster e raros nos outros)
    docs = [" ".join(meta.loc[meta.cluster_id == c, "enunciado"]) for c in range(K)]
    tf = TfidfVectorizer(lowercase=True, stop_words=list(STOP), ngram_range=(1, 2), min_df=1, max_df=0.5,
                         token_pattern=r"(?u)\b[^\W\d_]{4,}\b", sublinear_tf=True)
    M = tf.fit_transform(docs)
    termos = np.array(tf.get_feature_names_out())
    Z = pca.transform(X)
    nomes = []
    for c in range(K):
        top = termos[np.argsort(M[c].toarray()[0])[::-1][:6]]
        idx = np.where(meta.cluster_id.values == c)[0]
        d = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
        perto = idx[np.argsort(d)[:3]]
        nome = " / ".join(top[:3])
        nomes.append({"cluster_id": c, "n": len(idx), "nome_auto": nome, "termos": ", ".join(top)})
        print(f"\n== Cluster {c} (n={len(idx)}) — {nome}\n   termos: {', '.join(top)}")
        for i in perto:
            print("   •", meta.at[i, "enunciado"][:140].replace("\n", " "))
    pd.DataFrame(nomes).to_csv(RES / "cluster_nomes.csv", index=False)
    # nomes manuais aprovados (src/areas.py); se o cluster não tiver nome manual, cai no nome automático
    manual = {r["cluster_id"]: NOMES_CLUSTERS.get(r["cluster_id"], r["nome_auto"]) for r in nomes}
    meta["cluster_nome"] = meta.cluster_id.map(manual)
    meta[["id", "cluster_id", "cluster_nome"]].to_parquet(RES / "clusters_revalida.parquet", index=False)

    # o que os clusters do Revalida significam nas macroáreas reais do Enare (sanidade externa)
    meta_e["cluster_id"] = km.predict(pca.transform(Xe))
    print("\nEnare RM atribuído aos clusters do Revalida (% de cada macroárea por cluster):")
    print((pd.crosstab(meta_e.macroarea, meta_e.cluster_id, normalize="index") * 100).round(0).astype(int).to_string())

    # UMAP 2D conjunto (Revalida únicas + Enare únicas); cores: cluster | macroárea
    un_e = ~meta_e["chave_dup"].duplicated().values
    U = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(
        np.vstack([X[un], Xe[un_e]]))
    nr = un.sum()
    Ur, Ue = U[:nr], U[nr:]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    sc = ax[0].scatter(Ur[:, 0], Ur[:, 1], c=meta.cluster_id.values[un], cmap="tab10", s=8)
    ax[0].set_title("Revalida – clusters K-Means (k=8, versão a)")
    ax[0].legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax[1].scatter(Ur[:, 0], Ur[:, 1], c="lightgray", s=5, label="Revalida (sem rótulo)")
    for i, m in enumerate(sorted(meta_e.macroarea.dropna().unique())):
        s = meta_e.loc[un_e, "macroarea"].values == m
        ax[1].scatter(Ue[s, 0], Ue[s, 1], s=6, color=plt.cm.tab10(i), label=m)
    ax[1].set_title("Enare RM por macroárea (Revalida em cinza) – mesmo UMAP")
    ax[1].legend(fontsize=8, markerscale=2)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "umap_clusters_macroarea.png", dpi=130)
    print("\nfigura salva: figuras/umap_clusters_macroarea.png")


if __name__ == "__main__":
    main()
