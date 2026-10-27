"""Etapa 5.2 (variante B): kNN sobre embeddings e Ridge com PCA, com o MESMO split de src/regressao.py.

A ideia do kNN: a dificuldade de uma questão é a média da dificuldade das questões mais parecidas.
Uso: .venv/bin/python src/regressao_knn.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from regressao import EMB, RES, SEED, metricas  # reaproveita métricas e constantes


def main():
    meta = pd.read_parquet(EMB / "meta_revalida.parquet")
    alvo = pd.read_parquet("data/dificuldade.parquet")[["id", "dificuldade"]]
    df = meta.merge(alvo, on="id")
    pos = meta.reset_index().set_index("id")["index"]
    df["_i"] = df["id"].map(pos)
    un = df.drop_duplicates("chave_dup").reset_index(drop=True)
    y, g = un["dificuldade"].values, un["chave_dup"].values
    E = {v: np.load(EMB / f"{v}_revalida.npy")[un["_i"].values] for v in "ab"}

    # split idêntico ao de regressao.py
    bins = pd.qcut(y, 4, labels=False, duplicates="drop")
    idx_dev, idx_teste = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, bins, g))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(idx_dev, bins[idx_dev], g[idx_dev]))

    cand = {}
    for v in "ab":
        for k in (5, 10, 20, 40, 80, 160):
            cand[f"kNN k={k} [emb {v}]"] = (v, lambda k=k: KNeighborsRegressor(k, metric="cosine", weights="distance"))
        for n in (16, 32, 64, 128):
            cand[f"PCA{n}+Ridge [emb {v}]"] = (v, lambda n=n: make_pipeline(
                PCA(n, random_state=SEED), RidgeCV(alphas=np.logspace(-2, 3, 10))))

    linhas = []
    for nome, (v, fab) in cand.items():
        X = E[v]
        ms = []
        for tr, va in folds:
            tr, va = idx_dev[tr], idx_dev[va]
            ms.append(metricas(y[va], np.clip(fab().fit(X[tr], y[tr]).predict(X[va]), 0, 1)))
        r = pd.DataFrame(ms).mean().to_dict()
        linhas.append({"modelo": nome, **r})
        print(f"  {nome:24s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} R2 {r['R2']:+.3f} ρ {r['Spearman']:+.3f}")
    cv = pd.DataFrame(linhas).sort_values("R2", ascending=False)
    cv.to_csv(RES / "regressao_knn_cv.csv", index=False)
    melhor = cv.iloc[0]["modelo"]
    v, fab = cand[melhor]
    m = fab().fit(E[v][idx_dev], y[idx_dev])
    t = metricas(y[idx_teste], np.clip(m.predict(E[v][idx_teste]), 0, 1))
    print(f"\nmelhor no CV: {melhor} (R2 {cv.iloc[0]['R2']:+.3f})")
    print("TESTE FINAL:", {k: round(float(x), 3) for k, x in t.items()})


if __name__ == "__main__":
    main()
