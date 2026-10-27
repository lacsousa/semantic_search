"""Etapa 1: carregamento, escopo e validação do dataset processado.

Uso: .venv/bin/python src/validacao.py
Para trocar a fonte dos dados, altere apenas CAMINHO_DADOS / o mapeamento COLS.
"""
import re
from pathlib import Path

import pandas as pd

from areas import GROUP_PARA_MACRO

CAMINHO_DADOS = Path("data/healthqa_processed.parquet")
SAIDA = Path("resultados")
# Colunas reais do parquet do colega -> nomes usados no projeto
COLS = {"questão": "enunciado", "A": "alt_A", "B": "alt_B", "C": "alt_C", "D": "alt_D", "E": "alt_E"}
ALTS = ["alt_A", "alt_B", "alt_C", "alt_D", "alt_E"]
# Escopo: Revalida (produto) + Enare Residência Médica (treino do classificador)
ESCOPO = ["Revalida", "Enare Residência Médica"]

MIN_ENUNCIADO = 40   # caracteres
MIN_ALT = 2
# marcadores de alternativa que não deveriam sobrar no texto
RE_MARCADOR_INICIO = re.compile(r"^\s*['\"(\[]?[A-Ea-e]['\")\]]?\s*[:.)\-]\s*\S|^\s*\([A-Ea-e]\)|^\s*['\"][A-E]['\"]\s*:")
RE_MARCADOR_MEIO = re.compile(r"(?:^|\s)(?:\([A-E]\)|['\"][A-E]['\"]\s*:|[A-E]\))\s+\S")


def carregar() -> pd.DataFrame:
    df = pd.read_parquet(CAMINHO_DADOS).rename(columns=COLS)
    return df


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma linha por (id, problema)."""
    achados = []

    def add(mask, problema):
        for i in df.index[mask]:
            achados.append({"id": df.at[i, "id"], "source": df.at[i, "source"], "problema": problema})

    enun = df["enunciado"].fillna("")
    add(enun.str.strip().eq(""), "enunciado_vazio")
    add(enun.str.strip().str.len().between(1, MIN_ENUNCIADO - 1), "enunciado_curto")
    add(enun.str.contains(r"(?:^|\n)\s*\(?[A-E][\).:]\s+\S", regex=True), "enunciado_contem_alternativa")

    for a in ALTS[:4]:  # A–D são obrigatórias
        t = df[a].fillna("")
        add(t.str.strip().str.len() < MIN_ALT, f"{a}_vazia_ou_curta")
    for a in ALTS:
        t = df[a].fillna("")
        tem = df[a].notna() & t.str.strip().ne("")
        add(tem & t.apply(lambda s: bool(RE_MARCADOR_INICIO.search(s))), f"{a}_marcador_no_inicio")
        add(tem & t.apply(lambda s: bool(RE_MARCADOR_MEIO.search(s))), f"{a}_grudada_em_outra")

    # alternativas repetidas entre si
    def repetida(r):
        v = [str(r[a]).strip() for a in ALTS if pd.notna(r[a]) and str(r[a]).strip()]
        return len(v) != len(set(v))
    add(df.apply(repetida, axis=1), "alternativas_repetidas")

    # answer inválida ou apontando para alternativa nula
    ans = df["answer"].fillna("").str.strip().str.upper()
    add(~ans.isin(list("ABCDE")), "answer_invalida")
    for L in "ABCDE":
        a = f"alt_{L}"
        add(ans.eq(L) & (df[a].isna() | df[a].fillna("").str.strip().eq("")), "answer_aponta_alt_nula")

    # E nula deve ser consistente (4 alternativas) – E preenchida mas D vazia, etc.
    add(df["alt_E"].notna() & df["alt_D"].isna(), "E_preenchida_com_D_nula")
    return pd.DataFrame(achados, columns=["id", "source", "problema"])


def main():
    SAIDA.mkdir(exist_ok=True)
    bruto = carregar()
    print("df.columns:", list(pd.read_parquet(CAMINHO_DADOS).columns))
    print("shape bruto:", bruto.shape, "| ids únicos:", bruto["id"].is_unique)
    print("`question` == `enunciado`:", (bruto["question"] == bruto["enunciado"]).mean().round(3))

    df = bruto[bruto["source"].isin(ESCOPO)].copy()
    print("\nEscopo:", df["source"].value_counts().to_dict())

    df["n_alternativas"] = df[ALTS].notna().sum(axis=1)
    df["tam_enunciado"] = df["enunciado"].str.len()
    print("\nn_alternativas por source:\n", pd.crosstab(df["source"], df["n_alternativas"]))
    print("\ntamanho do enunciado (chars) por source:\n", df.groupby("source")["tam_enunciado"].describe().round(0))

    prob = validar(df)
    prob.to_csv(SAIDA / "ids_problematicos.csv", index=False)
    print("\n== Problemas de validação (escopo) ==")
    if prob.empty:
        print("nenhum")
    else:
        print(prob.groupby(["problema", "source"]).size().unstack(fill_value=0))
        print("ids distintos com algum problema:", prob["id"].nunique(), "de", len(df))

    print("\n== group nulo por source ==")
    print(df.assign(nulo=df["group"].isna()).groupby("source")["nulo"].agg(["sum", "size"]))
    print("\n== group nulo por source x year (só nulos) ==")
    print(df[df["group"].isna()].groupby(["source", "year"]).size())
    print("\n== source x year ==")
    print(pd.crosstab(df["year"], df["source"]))

    print("\n== valores de group (Enare RM) ==")
    g = df["group"].value_counts()
    print(g.to_string())
    g.rename_axis("group").reset_index(name="n").to_csv(SAIDA / "group_frequencias.csv", index=False)

    # macroárea (só existe onde há group, ou seja, no Enare) e chave para agrupar duplicatas nos splits
    df["macroarea"] = df["group"].map(GROUP_PARA_MACRO)
    df["chave_dup"] = df["enunciado"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df.to_parquet("data/escopo.parquet", index=False)
    print("\nsalvo: data/escopo.parquet, resultados/ids_problematicos.csv, resultados/group_frequencias.csv")


if __name__ == "__main__":
    main()
