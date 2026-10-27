"""Re-exporta o classificador de área final importando src/classificacao.py como módulo.

Motivo: o joblib.dump feito dentro de classificacao.py (executado como __main__) grava as classes como
`__main__.Modelo`, que outro programa (o sistema) não consegue desserializar.
Uso: .venv/bin/python src/exportar_classificador.py
"""
import joblib
import numpy as np
import pandas as pd

import classificacao as C

SEED = C.SEED


def main():
    melhor = pd.read_csv(C.RES / "classificacao_cv.csv").iloc[0]["modelo"]
    meta = pd.read_parquet(C.EMB / "meta_enare.parquet")
    y = meta["macroarea"].to_numpy(dtype=object)
    F = {"txt": meta["enunciado"].to_numpy(dtype=object), "a": np.load(C.EMB / "a_enare.npy"), "b": np.load(C.EMB / "b_enare.npy")}
    clf = C.candidatos()[melhor]().fit(F, y)
    joblib.dump({"clf": clf, "nome": melhor}, C.RES / "modelos/classificador_area.joblib")
    print("exportado:", melhor, "| classes:", list(clf.classes_))


if __name__ == "__main__":
    main()
