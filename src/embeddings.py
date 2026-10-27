"""Etapa 2: gera embeddings (e5) das versões (a) enunciado e (b) enunciado + alternativas.

Uso: .venv/bin/python src/embeddings.py
Gera embeddings/{a,b}_{revalida,enare}.npy e embeddings/meta_{revalida,enare}.parquet
(linhas alinhadas por posição com os .npy). Reexecução reaproveita os arquivos existentes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-base"
DADOS = Path("data/escopo.parquet")
SAIDA = Path("embeddings")
SEED = 42
ALTS = ["alt_A", "alt_B", "alt_C", "alt_D", "alt_E"]


def texto_a(df: pd.DataFrame) -> list[str]:
    """Versão (a): só o enunciado (prefixo 'passage: ' exigido pelo e5)."""
    return ("passage: " + df["enunciado"].str.strip()).tolist()


def texto_b(df: pd.DataFrame) -> list[str]:
    """Versão (b): enunciado + alternativas formatadas (A)...(E)."""
    def monta(r):
        alts = [f"({L}) {r['alt_' + L].strip()}" for L in "ABCDE" if pd.notna(r["alt_" + L])]
        return "passage: " + r["enunciado"].strip() + "\n" + "\n".join(alts)
    return df.apply(monta, axis=1).tolist()


def main():
    torch.manual_seed(SEED)
    SAIDA.mkdir(exist_ok=True)
    df = pd.read_parquet(DADOS)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SentenceTransformer(MODELO, device=dispositivo)
    modelo.max_seq_length = 512
    if dispositivo == "cuda":
        modelo.half()
    tok = modelo.tokenizer

    for nome, sub in [("revalida", df[df.source == "Revalida"]), ("enare", df[df.source != "Revalida"])]:
        sub = sub.reset_index(drop=True)
        sub.to_parquet(SAIDA / f"meta_{nome}.parquet", index=False)
        for versao, fn in [("a", texto_a), ("b", texto_b)]:
            arq = SAIDA / f"{versao}_{nome}.npy"
            if arq.exists():
                print("já existe, pulando:", arq)
                continue
            textos = fn(sub)
            ntok = np.array([len(x) for x in tok(textos, add_special_tokens=True)["input_ids"]])
            print(f"{nome}/{versao}: {len(textos)} textos | tokens média {ntok.mean():.0f}, "
                  f"p95 {np.percentile(ntok, 95):.0f}, truncados (>512): {(ntok > 512).sum()}")
            emb = modelo.encode(textos, batch_size=32, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            np.save(arq, emb.astype(np.float32))
            print("  salvo", arq, emb.shape)


if __name__ == "__main__":
    main()
