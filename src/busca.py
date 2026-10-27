"""Etapa 2: índice FAISS (cosseno) sobre as questões do Revalida e função buscar().

Os vetores já são normalizados, então produto interno == similaridade de cosseno.
"""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-base"
EMB = Path("embeddings")
_modelo = None


def _get_modelo():
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(MODELO)
        _modelo.max_seq_length = 512
    return _modelo


def carregar_indice(versao: str = "a"):
    """Devolve (índice FAISS, metadados) alinhados por posição."""
    vetores = np.load(EMB / f"{versao}_revalida.npy")
    meta = pd.read_parquet(EMB / "meta_revalida.parquet")
    indice = faiss.IndexFlatIP(vetores.shape[1])
    indice.add(vetores)
    return indice, meta


def buscar(texto: str, k: int = 5, versao: str = "a", indice=None, meta=None) -> pd.DataFrame:
    """KNN: embedding da dúvida (prefixo 'query: ') -> top-k questões do Revalida."""
    if indice is None:
        indice, meta = carregar_indice(versao)
    q = _get_modelo().encode(["query: " + texto], normalize_embeddings=True).astype(np.float32)
    # busca k*3 e remove duplicatas (mesmo enunciado em cadernos diferentes) antes de cortar em k
    sim, pos = indice.search(q, min(k * 3, indice.ntotal))
    res = meta.iloc[pos[0]].copy()
    res.insert(0, "similaridade", sim[0])
    return res.drop_duplicates("chave_dup").head(k)
