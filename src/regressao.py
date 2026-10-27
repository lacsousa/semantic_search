"""Etapa 5.2: regressão da dificuldade estimada (alvo gerado pelo painel de LLMs) a partir do texto da questão.

Regras: NENHUMA saída do painel entra como feature (vazamento). Só embedding + metadados do enunciado.
Protocolo: treina só nas questões únicas; 20% por grupo (chave_dup) = teste final; resto = StratifiedGroupKFold(5).
Uso: .venv/bin/python src/regressao.py
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
EMB, RES, FIG = Path("embeddings"), Path("resultados"), Path("figuras")
UNIDADES = r"mg/dl|g/dl|mmhg|mmol|meq|/mm3|/mm³|u/l|ui/l|mg/l|ng/ml|bpm|irpm|°c|µ|mcg|ml/min|mosm"


def metadados(df: pd.DataFrame) -> pd.DataFrame:
    """Features baratas do texto do enunciado/alternativas (nada vindo do painel)."""
    alts = df[[f"alt_{L}" for L in "ABCDE"]]
    t = df["enunciado"].fillna("")
    return pd.DataFrame({
        "tam_enunciado": t.str.len(),
        "tam_alt_media": alts.apply(lambda c: c.fillna("").str.len()).sum(axis=1) / alts.notna().sum(axis=1),
        "n_alternativas": alts.notna().sum(axis=1),
        "n_numeros": t.str.count(r"\d+[.,]?\d*"),
        "n_unidades_lab": t.str.lower().str.count(UNIDADES),
        "tem_tabela": (t.str.contains(r"\||\t|refer[êe]ncia", case=False, regex=True)
                       | (t.str.lower().str.count(UNIDADES) >= 4)).astype(int),
        "pergunta_negativa": t.str.contains(r"\bexceto\b|\bincorret|\bn[ãa]o\s+(?:é|se|est)", case=False, regex=True).astype(int),
    })


def metricas(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p), "R2": r2_score(y, p),
            "Spearman": spearmanr(y, p)[0] if np.std(p) > 0 else 0.0}


def modelos():
    return {
        "Baseline (média)": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4.5, 10))),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.3,
                                                      n_jobs=-1, random_state=SEED),
        "XGBoost": lambda: XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                        colsample_bytree=0.3, n_jobs=8, random_state=SEED, tree_method="hist"),
    }


def main():
    meta = pd.read_parquet(EMB / "meta_revalida.parquet")
    alvo = pd.read_parquet("data/dificuldade.parquet")[["id", "dificuldade", "taxa_erro"]]
    area = pd.read_parquet(RES / "area_revalida.parquet")[["id", "area_final", "area_confianca", "cluster_id", "cluster_nome"]]
    df = meta.merge(alvo, on="id").merge(area, on="id")
    emb = {v: np.load(EMB / f"{v}_revalida.npy") for v in "ab"}
    pos = meta.reset_index().set_index("id")["index"]
    df["_i"] = df["id"].map(pos)

    un = df.drop_duplicates("chave_dup").reset_index(drop=True)  # modelagem só com questões únicas
    md = metadados(un)
    A = pd.get_dummies(un["area_final"], prefix="area").astype(float)
    E = {v: emb[v][un["_i"].values] for v in "ab"}
    y = un["dificuldade"].values
    g = un["chave_dup"].values
    print(f"{len(un)} questões únicas | alvo: média {y.mean():.3f}, dp {y.std():.3f}")

    conjuntos = {
        "só metadados": md.values,
        "emb a": E["a"],
        "emb b": E["b"],
        "emb b + metadados": np.hstack([E["b"], md.values]),
        "emb b + metadados + área": np.hstack([E["b"], md.values, A.values]),
        "metadados + área": np.hstack([md.values, A.values]),
    }

    bins = pd.qcut(y, 4, labels=False, duplicates="drop")
    idx_dev, idx_teste = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, bins, g))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(idx_dev, bins[idx_dev], g[idx_dev]))
    print(f"dev {len(idx_dev)} | teste {len(idx_teste)} | grupos em comum: {len(set(g[idx_dev]) & set(g[idx_teste]))}")

    linhas = []
    for cn, X in conjuntos.items():
        for mn, fab in modelos().items():
            if mn == "Baseline (média)" and cn != "só metadados":
                continue
            ms = []
            for tr, va in folds:
                tr, va = idx_dev[tr], idx_dev[va]
                m = fab().fit(X[tr], y[tr])
                ms.append(metricas(y[va], np.clip(m.predict(X[va]), 0, 1)))
            r = pd.DataFrame(ms).mean().to_dict()
            linhas.append({"features": cn if mn != "Baseline (média)" else "—", "modelo": mn, **r})
            print(f"  {linhas[-1]['features']:26s} {mn:17s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} R2 {r['R2']:+.3f} ρ {r['Spearman']:+.3f}")
    cv = pd.DataFrame(linhas).sort_values("R2", ascending=False)
    cv.to_csv(RES / "regressao_cv.csv", index=False)
    top = cv[cv.modelo != "Baseline (média)"].iloc[0]
    print(f"\nmelhor no CV: {top.modelo} + {top.features} (R2 {top.R2:+.3f}, ρ {top.Spearman:+.3f})")

    # TESTE FINAL (uma vez): baseline e melhor combinação
    Xb = conjuntos[top.features]
    teste = []
    for nome, fab, X in [("Baseline (média)", modelos()["Baseline (média)"], Xb), (f"{top.modelo} + {top.features}", modelos()[top.modelo], Xb)]:
        m = fab().fit(X[idx_dev], y[idx_dev])
        teste.append({"modelo": nome, **metricas(y[idx_teste], np.clip(m.predict(X[idx_teste]), 0, 1))})
    teste = pd.DataFrame(teste)
    teste.to_csv(RES / "regressao_teste.csv", index=False)
    print("\n== TESTE FINAL ==\n", teste.round(3).to_string(index=False))

    # previsões fora-da-amostra (OOF) para TODAS as únicas, e modelo final treinado em tudo
    oof = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 2).split(Xb, bins, g):
        oof[va] = np.clip(modelos()[top.modelo]().fit(Xb[tr], y[tr]).predict(Xb[va]), 0, 1)
    print("OOF (todas as únicas):", {k: round(v, 3) for k, v in metricas(y, oof).items()})
    final = modelos()[top.modelo]().fit(Xb, y)
    joblib.dump({"modelo": final, "features": top.features, "colunas_meta": list(md.columns),
                 "colunas_area": list(A.columns), "nome": top.modelo}, RES / "modelos/regressor_dificuldade.joblib")
    un["dificuldade_prevista"] = oof
    out = df[["id", "chave_dup"]].merge(un[["chave_dup", "dificuldade_prevista"]], on="chave_dup")
    out[["id", "dificuldade_prevista"]].to_parquet(RES / "dificuldade_prevista_revalida.parquet", index=False)

    # análises de dificuldade por área, cluster e ano (alvo real do painel)
    print("\n== dificuldade média (alvo do painel) por área prevista ==")
    print(un.groupby("area_final")["dificuldade"].agg(["mean", "std", "size"]).round(3).to_string())
    print("\n== por cluster temático ==")
    print(un.groupby("cluster_nome")["dificuldade"].agg(["mean", "std", "size"]).round(3).sort_values("mean").to_string())
    print("\n== por ano ==")
    print(un.groupby("year")["dificuldade"].agg(["mean", "size"]).round(3).T.to_string())
    print("\n== correlação do alvo com metadados (Spearman) ==")
    for c in md.columns:
        print(f"  {c:18s} {spearmanr(md[c], y)[0]:+.3f}")

    un["prev"] = oof
    print("\n== 5 mais FÁCEIS (alvo real) ==")
    for r in un.nsmallest(5, "dificuldade").itertuples():
        print(f"  [{r.dificuldade:.2f} | prev {r.prev:.2f}] {r.enunciado[:110]!r}")
    print("== 5 mais DIFÍCEIS (alvo real) ==")
    for r in un.nlargest(5, "dificuldade").itertuples():
        print(f"  [{r.dificuldade:.2f} | prev {r.prev:.2f}] {r.enunciado[:110]!r}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(y, oof, s=6, alpha=0.4)
    ax[0].plot([0, 1], [0, 1], "r--", lw=1)
    ax[0].set(xlabel="dificuldade real (painel)", ylabel="prevista (OOF)",
              title=f"{top.modelo} + {top.features}\nρ={spearmanr(y, oof)[0]:.2f}")
    ordem = un.groupby("area_final")["dificuldade"].mean().sort_values().index
    ax[1].boxplot([un.loc[un.area_final == a, "dificuldade"] for a in ordem], tick_labels=[a[:14] for a in ordem])
    ax[1].tick_params(axis="x", rotation=35); ax[1].set_title("Dificuldade por área prevista")
    ordem = un.groupby("cluster_id")["dificuldade"].mean().sort_values().index
    ax[2].boxplot([un.loc[un.cluster_id == c, "dificuldade"] for c in ordem], tick_labels=[f"cl {c}" for c in ordem])
    ax[2].set_title("Dificuldade por cluster temático")
    fig.tight_layout(); fig.savefig(FIG / "regressao_dificuldade.png", dpi=130)
    print("\nfigura: figuras/regressao_dificuldade.png")


if __name__ == "__main__":
    main()
