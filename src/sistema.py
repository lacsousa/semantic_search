"""Etapa 6: sistema integrado de busca de questões do Revalida.

buscar(pergunta, k) -> embedding da pergunta -> área prevista da pergunta -> top-k questões parecidas,
cada uma com enunciado, alternativas, área (real/prevista), cluster temático, dificuldade (faixa) e similaridade.

Uso rápido: .venv/bin/python src/sistema.py "paciente com dor torácica e suspeita de infarto"
"""
import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))  # permite desserializar o classificador (módulo classificacao)

EMB, RES = Path("embeddings"), Path("resultados")
MODELO_EMB = "intfloat/multilingual-e5-base"
FAIXAS = ["Fácil", "Média", "Difícil"]
ALTS = ["A", "B", "C", "D", "E"]


class Sistema:
    def __init__(self):
        meta = pd.read_parquet(EMB / "meta_revalida.parquet")
        area = pd.read_parquet(RES / "area_revalida.parquet")[["id", "area_final", "area_origem", "area_confianca", "cluster_id", "cluster_nome"]]
        dif = pd.read_parquet(RES / "dificuldade_prevista_revalida.parquet")
        self.meta = meta.merge(area, on="id").merge(dif, on="id").reset_index(drop=True)
        # faixas de dificuldade por tercis da PREVISÃO (nas questões únicas); alvo real do painel não é mostrado
        unicas = self.meta.drop_duplicates("chave_dup")["dificuldade_prevista"]
        self.cortes = unicas.quantile([1 / 3, 2 / 3]).values
        self.meta["dificuldade_faixa"] = pd.cut(self.meta["dificuldade_prevista"], [-np.inf, *self.cortes, np.inf],
                                                labels=FAIXAS).astype(str)
        # índice FAISS exato; vetores normalizados -> produto interno = cosseno
        X = np.load(EMB / "a_revalida.npy")
        assert len(X) == len(self.meta) and (meta["id"].values == self.meta["id"].values).all(), "índice desalinhado"
        self.indice = faiss.IndexFlatIP(X.shape[1])
        self.indice.add(X)
        self.emb = SentenceTransformer(MODELO_EMB)
        self.emb.max_seq_length = 512
        self.clf = joblib.load(RES / "modelos/classificador_area.joblib")["clf"]
        self.areas = list(self.clf.classes_)

    def prever_area(self, pergunta: str) -> pd.Series:
        """Probabilidade de cada macroárea para a pergunta do usuário.
        A pergunta é embutida como 'passage:' (mesmo espaço dos dados de treino); como ela não traz alternativas,
        o vetor da versão (b) é só uma aproximação."""
        v = self.emb.encode(["passage: " + pergunta], normalize_embeddings=True).astype(np.float32)
        p = self.clf.predict_proba({"txt": np.array([pergunta], dtype=object), "a": v, "b": v})[0]
        return pd.Series(p, index=self.areas).sort_values(ascending=False)

    def buscar(self, pergunta: str, k: int = 5, areas: list[str] | None = None, faixas: list[str] | None = None,
               candidatos: int = 400) -> dict:
        """Top-k questões do Revalida. `areas`/`faixas` filtram DEPOIS da busca vetorial (entre os `candidatos` mais próximos)."""
        q = self.emb.encode(["query: " + pergunta], normalize_embeddings=True).astype(np.float32)
        sim, pos = self.indice.search(q, min(candidatos, self.indice.ntotal))
        r = self.meta.iloc[pos[0]].copy()
        r.insert(0, "similaridade", sim[0])
        r = r.drop_duplicates("chave_dup")  # mesmo enunciado em cadernos diferentes
        if areas:
            r = r[r["area_final"].isin(areas)]
        if faixas:
            r = r[r["dificuldade_faixa"].isin(faixas)]
        r = r.head(k)
        return {"area_pergunta": self.prever_area(pergunta), "resultados": r}


def formatar_alternativas(linha, destacar=True) -> str:
    """(A)...(E) com a correta marcada por ✔."""
    out = []
    for L in ALTS:
        t = linha[f"alt_{L}"]
        if pd.isna(t):
            continue
        ok = str(linha["answer"]).strip().upper() == L
        out.append(f"{'✔ ' if ok and destacar else '   '}({L}) {str(t).strip()}")
    return "\n".join(out)


def main():
    pergunta = " ".join(sys.argv[1:]) or "paciente com dor torácica e suspeita de infarto agudo do miocárdio"
    s = Sistema()
    res = s.buscar(pergunta, k=3)
    print("Pergunta:", pergunta)
    print("Área prevista da pergunta:", res["area_pergunta"].round(2).to_dict())
    for r in res["resultados"].itertuples():
        print(f"\n[{r.similaridade:.3f}] {r.year} | {r.area_final} ({r.area_origem}, conf. {r.area_confianca:.2f}) | "
              f"{r.cluster_nome} | dificuldade: {r.dificuldade_faixa} ({r.dificuldade_prevista:.2f})")
        print(r.enunciado[:300].replace("\n", " "), "...")
        print(formatar_alternativas(r._asdict()))


if __name__ == "__main__":
    main()
