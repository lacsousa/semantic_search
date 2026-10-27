"""Dispersão UMAP 2D conjunta Enare + Revalida: por fonte, por cluster temático e por macroárea.

Uso: .venv/bin/python src/visualizacao.py
Coordenadas ficam em cache (resultados/umap2d_{versao}.parquet) para reexecuções baratas.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import umap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
VERSAO = "a"  # mesmo espaço em que os clusters foram ajustados
EMB, RES, FIG = Path("embeddings"), Path("resultados"), Path("figuras")


def coordenadas() -> pd.DataFrame:
    cache = RES / f"umap2d_{VERSAO}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rev = pd.read_parquet(EMB / "meta_revalida.parquet").assign(fonte="Revalida")
    ena = pd.read_parquet(EMB / "meta_enare.parquet").assign(fonte="Enare RM")
    X = np.vstack([np.load(EMB / f"{VERSAO}_revalida.npy"), np.load(EMB / f"{VERSAO}_enare.npy")])
    df = pd.concat([rev, ena], ignore_index=True)
    # UMAP ajustado nas questões únicas; as duplicatas herdam a posição do enunciado igual
    un = ~df["chave_dup"].duplicated().values
    modelo = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit(X[un])
    P = modelo.transform(X)
    df["x"], df["y"] = P[:, 0], P[:, 1]
    out = df[["id", "fonte", "x", "y"]]
    out.to_parquet(cache, index=False)
    return out


def main():
    c = coordenadas()
    rev = pd.read_parquet(RES / "area_revalida.parquet")
    ena = pd.read_parquet(EMB / "meta_enare.parquet")[["id", "macroarea"]]
    r = c[c.fonte == "Revalida"].merge(rev, on="id")
    e = c[c.fonte == "Enare RM"].merge(ena, on="id")

    fig, ax = plt.subplots(1, 3, figsize=(24, 7.5))

    # 1) por fonte: quão separados estão os dois domínios?
    ax[0].scatter(e.x, e.y, s=5, alpha=0.5, color="tab:orange", label=f"Enare RM (n={len(e)})")
    ax[0].scatter(r.x, r.y, s=5, alpha=0.5, color="tab:blue", label=f"Revalida (n={len(r)})")
    ax[0].set_title("Por fonte")
    ax[0].legend(markerscale=3)

    # 2) Revalida por cluster temático (k=8); Enare ao fundo em cinza
    ax[1].scatter(e.x, e.y, s=4, color="lightgray", label="Enare RM")
    cm = plt.get_cmap("tab10")
    for cid, g in r.groupby("cluster_id"):
        ax[1].scatter(g.x, g.y, s=6, color=cm(int(cid)), label=f"{cid} – {g.cluster_nome.iloc[0]}")
        ax[1].annotate(str(cid), (g.x.median(), g.y.median()), fontsize=14, weight="bold",
                       bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax[1].set_title("Revalida por cluster temático (K-Means, k=8)")
    ax[1].legend(fontsize=7, markerscale=2.5, loc="best")

    # 3) macroárea: real no Enare (círculos), prevista no Revalida (cruzes)
    areas = sorted(e.macroarea.unique())
    for i, a in enumerate(areas):
        s = e[e.macroarea == a]
        ax[2].scatter(s.x, s.y, s=6, alpha=0.5, color=cm(i), label=f"{a} (Enare, real)")
        s = r[r.area_final == a]
        ax[2].scatter(s.x, s.y, s=14, marker="x", linewidths=0.7, color=cm(i), label=f"{a} (Revalida, prevista)")
    ax[2].set_title("Macroárea: Enare real (●) e Revalida prevista (×)")
    ax[2].legend(fontsize=6, markerscale=1.5, ncol=1, loc="best")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("UMAP 2D dos embeddings (e5-base, versão a) – Enare RM e Revalida no mesmo espaço", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "dispersao_enare_revalida.png", dpi=130)
    print("salvo figuras/dispersao_enare_revalida.png")


if __name__ == "__main__":
    main()
