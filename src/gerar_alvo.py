"""Etapa 5.1: combina o painel de LLMs no alvo de dificuldade e roda os testes de sanidade.

  p_correta_<modelo> : probabilidade que o modelo deu à letra correta
  dificuldade        : 1 - média das p_correta (alvo principal, contínuo em [0, 1])
  taxa_erro          : proporção de modelos que erraram (alvo alternativo, discreto)

As questões duplicadas (mesma chave_dup) herdam os valores da primeira ocorrência.
Uso: .venv/bin/python src/gerar_alvo.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAINEL, FIG = Path("resultados/painel"), Path("figuras")
MODELOS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "phi-3.5-mini"]
SAIDA = Path("data/dificuldade.parquet")


def main():
    meta = pd.read_parquet("embeddings/meta_revalida.parquet")[["id", "chave_dup", "answer", "alt_E"]]
    meta["n_alt"] = 4 + meta["alt_E"].notna().astype(int)

    # uma linha por questão única, uma coluna p_correta/acertou por modelo
    base = None
    usados = []
    for m in MODELOS:
        arq = PAINEL / f"{m}.parquet"
        if not arq.exists():
            print(f"[aviso] sem resultado de {m}, ignorado")
            continue
        d = pd.read_parquet(arq)[["id", "p_correta", "acertou", "letra_escolhida", "truncada"]]
        d = d.rename(columns={"p_correta": f"p_correta_{m}", "acertou": f"acertou_{m}",
                              "letra_escolhida": f"letra_{m}", "truncada": f"truncada_{m}"})
        base = d if base is None else base.merge(d, on="id", how="outer")
        usados.append(m)
    assert usados, "nenhum resultado de painel encontrado"
    print("modelos usados:", usados)

    unicas = meta.drop_duplicates("chave_dup").merge(base, on="id", how="left")
    faltando = unicas[[f"p_correta_{m}" for m in usados]].isna().any(axis=1).sum()
    print(f"questões únicas: {len(unicas)} | com algum modelo faltando: {faltando}")
    unicas = unicas[~unicas[[f"p_correta_{m}" for m in usados]].isna().any(axis=1)]

    pc = unicas[[f"p_correta_{m}" for m in usados]]
    ac = unicas[[f"acertou_{m}" for m in usados]].astype(float)
    unicas["dificuldade"] = 1 - pc.mean(axis=1)
    unicas["taxa_erro"] = 1 - ac.mean(axis=1)

    # propaga às duplicatas
    cols = [c for c in unicas.columns if c not in ("id", "answer", "alt_E", "n_alt")]
    alvo = meta[["id", "chave_dup", "n_alt"]].merge(unicas[cols], on="chave_dup", how="inner")
    alvo.to_parquet(SAIDA, index=False)
    print(f"salvo {SAIDA}: {len(alvo)} questões (de {len(meta)})")

    # ---- testes de sanidade ----
    print("\n== 1) acurácia de cada modelo vs. acaso ==")
    acaso = (1 / unicas["n_alt"]).mean()
    for m in usados:
        a = unicas[f"acertou_{m}"].astype(float).mean()
        letras = unicas[f"letra_{m}"].value_counts(normalize=True).round(2).to_dict()
        print(f"  {m:14s} acurácia {a:.3f} (acaso {acaso:.3f}) | truncadas {int(unicas[f'truncada_{m}'].sum())} "
              f"| letras escolhidas {letras}")
    print("  letras corretas (gabarito):", unicas["answer"].value_counts(normalize=True).round(2).to_dict())

    print("\n== 2) correlação entre modelos (p_correta) ==")
    print(" Pearson:\n", pc.corr().round(3).to_string())
    print(" Spearman:\n", pc.corr(method="spearman").round(3).to_string())

    print("\n== 3) distribuição do alvo ==")
    print(unicas[["dificuldade", "taxa_erro"]].describe().round(3).to_string())
    print("correlação dificuldade x taxa_erro (Spearman): %.3f" % unicas["dificuldade"].corr(unicas["taxa_erro"], method="spearman"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(unicas["dificuldade"], bins=30, color="tab:blue")
    ax[0].set(title="Dificuldade (1 − média de p_correta)", xlabel="dificuldade", ylabel="questões")
    for m in usados:
        ax[1].hist(unicas[f"p_correta_{m}"], bins=20, alpha=0.5, label=m)
    ax[1].set(title="p_correta por modelo", xlabel="p_correta"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "dificuldade_hist.png", dpi=130)
    print("figura: figuras/dificuldade_hist.png")


if __name__ == "__main__":
    main()
