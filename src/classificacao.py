"""Etapa 4: classificação de macroárea. Treina SÓ no Enare RM (rótulo real) e prevê o Revalida.

Protocolo (definido antes de qualquer ajuste):
  * 20% do Enare (por grupos de chave_dup, estratificado por macroárea) fica como TESTE FINAL, intocado.
  * Os outros 80% entram em StratifiedGroupKFold(5) para comparar modelos/embeddings.
  * Duplicatas (mesmo enunciado) ficam sempre do mesmo lado do split -> sem vazamento.
Uso: .venv/bin/python src/classificacao.py
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
EMB, RES, FIG = Path("embeddings"), Path("resultados"), Path("figuras")


class Modelo:
    """Interface única: fit(F, y) / predict_proba(F), onde F = {'txt': textos, 'a': emb_a, 'b': emb_b}."""

    def __init__(self, entrada, fabrica):
        self.entrada, self.fabrica = entrada, fabrica

    def fit(self, F, y):
        self.m = self.fabrica().fit(F[self.entrada], y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(F[self.entrada])


class Media:
    """Média das probabilidades de vários modelos (ensemble simples)."""

    def __init__(self, *modelos):
        self.modelos = modelos

    def fit(self, F, y):
        for m in self.modelos:
            m.fit(F, y)
        self.classes_ = self.modelos[0].classes_
        return self

    def predict_proba(self, F):
        return np.mean([m.predict_proba(F) for m in self.modelos], axis=0)


def tfidf_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
                         LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=SEED))


def logreg():
    return LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=SEED)


def svm():  # LinearSVC não dá probabilidade: calibramos para obter a confiança
    return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=SEED), cv=3)


def rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def candidatos():
    return {
        "TF-IDF+LogReg (baseline)": lambda: Modelo("txt", tfidf_lr),
        "LogReg [emb a]": lambda: Modelo("a", logreg),
        "SVM linear [emb a]": lambda: Modelo("a", svm),
        "RandomForest [emb a]": lambda: Modelo("a", rf),
        "LogReg [emb b]": lambda: Modelo("b", logreg),
        "SVM linear [emb b]": lambda: Modelo("b", svm),
        "RandomForest [emb b]": lambda: Modelo("b", rf),
        "Média TF-IDF + LogReg [emb b]": lambda: Media(Modelo("txt", tfidf_lr), Modelo("b", logreg)),
    }


def fatia(F, idx):
    return {k: v[idx] for k, v in F.items()}


def main():
    RES.mkdir(exist_ok=True)
    (RES / "modelos").mkdir(exist_ok=True)
    meta = pd.read_parquet(EMB / "meta_enare.parquet")
    y = meta["macroarea"].to_numpy(dtype=object)
    grupos = meta["chave_dup"].to_numpy(dtype=object)
    F = {"txt": meta["enunciado"].to_numpy(dtype=object), "a": np.load(EMB / "a_enare.npy"), "b": np.load(EMB / "b_enare.npy")}

    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    idx_dev, idx_teste = next(sgk.split(F["a"], y, grupos))
    print(f"dev {len(idx_dev)} | teste {len(idx_teste)} | grupos em comum: {len(set(grupos[idx_dev]) & set(grupos[idx_teste]))}")
    print("teste por classe:", pd.Series(y[idx_teste]).value_counts().to_dict())
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
                 .split(idx_dev, y[idx_dev], grupos[idx_dev]))

    cands = candidatos()
    linhas = []
    for nome, fab in cands.items():
        f1s, accs = [], []
        for tr, va in folds:
            tr, va = idx_dev[tr], idx_dev[va]
            m = fab().fit(fatia(F, tr), y[tr])
            p = m.classes_[m.predict_proba(fatia(F, va)).argmax(1)]
            f1s.append(f1_score(y[va], p, average="macro")); accs.append(accuracy_score(y[va], p))
        linhas.append({"modelo": nome, "acc": np.mean(accs), "f1_macro": np.mean(f1s), "f1_dp": np.std(f1s)})
        print(f"  {nome:32s} acc {np.mean(accs):.3f}  F1macro {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    cvdf = pd.DataFrame(linhas).sort_values("f1_macro", ascending=False)
    cvdf.to_csv(RES / "classificacao_cv.csv", index=False)
    melhor = cvdf.iloc[0]["modelo"]
    melhor_emb = cvdf[cvdf.modelo.str.contains("emb")].iloc[0]["modelo"]
    print("\nmelhor no CV:", melhor, "| melhor só-embedding:", melhor_emb)

    # TESTE FINAL: usado uma única vez, só para baseline, melhor só-embedding e melhor geral
    resumo = []
    for nome in dict.fromkeys(["TF-IDF+LogReg (baseline)", melhor_emb, melhor]):
        m = cands[nome]().fit(fatia(F, idx_dev), y[idx_dev])
        p = m.classes_[m.predict_proba(fatia(F, idx_teste)).argmax(1)]
        print(f"\n===== TESTE FINAL: {nome} =====")
        print(classification_report(y[idx_teste], p, digits=3, zero_division=0))
        resumo.append({"modelo": nome, "acc": accuracy_score(y[idx_teste], p), "f1_macro": f1_score(y[idx_teste], p, average="macro")})
        if nome == melhor:
            ConfusionMatrixDisplay.from_predictions(y[idx_teste], p, xticks_rotation=40, colorbar=False,
                                                    normalize="true", values_format=".2f")
            plt.title(f"Matriz de confusão (normalizada) – {nome}", fontsize=9); plt.tight_layout()
            plt.savefig(FIG / "confusao_classificador.png", dpi=130); plt.close()
    pd.DataFrame(resumo).to_csv(RES / "classificacao_teste.csv", index=False)

    # modelo final: treina em TODO o Enare (dev + teste) e rotula o Revalida
    clf = cands[melhor]().fit(F, y)
    joblib.dump({"clf": clf, "nome": melhor}, RES / "modelos/classificador_area.joblib")
    rev = pd.read_parquet(EMB / "meta_revalida.parquet")
    Fr = {"txt": rev["enunciado"].to_numpy(dtype=object), "a": np.load(EMB / "a_revalida.npy"), "b": np.load(EMB / "b_revalida.npy")}
    P = clf.predict_proba(Fr)
    rev["area_final"] = clf.classes_[P.argmax(1)]
    rev["area_origem"] = "prevista"
    rev["area_confianca"] = P.max(1)
    cl = pd.read_parquet(RES / "clusters_revalida.parquet")[["id", "cluster_id", "cluster_nome"]]
    rev = rev.merge(cl, on="id", how="left")
    rev[["id", "area_final", "area_origem", "area_confianca", "cluster_id", "cluster_nome"]].to_parquet(
        RES / "area_revalida.parquet", index=False)
    print(f"\n== Revalida rotulado com: {melhor} ==")
    print(rev.area_final.value_counts().to_string())
    print("\nconfiança: média %.2f | <0,5: %.1f%% | >=0,8: %.1f%%" % (
        rev.area_confianca.mean(), 100 * (rev.area_confianca < 0.5).mean(), 100 * (rev.area_confianca >= 0.8).mean()))
    print("\n== área prevista x cluster temático (%, por cluster) ==")
    print((pd.crosstab(rev.cluster_id, rev.area_final, normalize="index") * 100).round(0).astype(int).to_string())

    # amostra para revisão manual (10 por área prevista) – checa o domain shift Enare -> Revalida
    am = pd.concat([d.sample(min(len(d), 10), random_state=SEED) for _, d in rev.groupby("area_final")])
    am[["id", "area_final", "area_confianca", "enunciado"]].assign(area_correta_manual="").to_csv(
        RES / "amostra_revisao_manual.csv", index=False)
    print(f"\namostra p/ revisão manual: resultados/amostra_revisao_manual.csv ({len(am)} linhas)")


if __name__ == "__main__":
    main()
