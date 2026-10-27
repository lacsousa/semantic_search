"""Etapa 5.1: painel de LLMs locais respondendo às questões do Revalida (lendo logits, sem amostragem).

Para cada questão e modelo salva a probabilidade de cada letra (softmax só sobre as letras existentes),
a letra escolhida e se acertou, em resultados/painel/{modelo}.parquet (salvamento incremental + retomada).

Exemplos:
  .venv/bin/python src/painel_dificuldade.py --teste --modelos qwen2.5-0.5b
  .venv/bin/python src/painel_dificuldade.py --modelos qwen2.5-0.5b qwen2.5-1.5b
  .venv/bin/python src/painel_dificuldade.py --amostra 300 --modelos qwen2.5-7b
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SEED = 42
SAIDA = Path("resultados/painel")
MAX_TOKENS = 1536          # limite do prompt; acima disso o enunciado é truncado
LETRAS = list("ABCDE")

# nome curto -> (repositório HF, precisão)
MODELOS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "fp16"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "fp16"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "fp16"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "4bit"),
    "phi-3.5-mini": ("microsoft/Phi-3.5-mini-instruct", "4bit"),
}
SISTEMA = "Você é um médico experiente respondendo a uma prova de residência/revalidação médica."


def carregar_questoes(caminho: str, teste: bool, amostra: int | None) -> pd.DataFrame:
    """Questões únicas do Revalida (duplicatas por chave_dup herdam o resultado depois, em gerar_alvo)."""
    df = pd.read_parquet(caminho)
    if "source" in df.columns:
        df = df[df["source"] == "Revalida"]
    df = df.drop_duplicates("chave_dup").reset_index(drop=True)
    n = 50 if teste else amostra
    if n:
        df = df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def monta_prompt(tok, enunciado: str, alts: dict[str, str]) -> str:
    corpo = enunciado.strip() + "\n\n" + "\n".join(f"({L}) {t.strip()}" for L, t in alts.items())
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": corpo + "\n\nResponda apenas com a letra da alternativa correta."}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # "Resposta: (" -> o próximo token é a letra. Sem o "(" o modelo preferia " (" e só ~38% da massa ia às letras.
    return base + "Resposta: ("


def prompt_seguro(tok, enunciado: str, alts: dict[str, str]):
    """Trunca o INÍCIO do enunciado se o prompt passar de MAX_TOKENS (mantém a pergunta final). Devolve (prompt, truncou)."""
    p = monta_prompt(tok, enunciado, alts)
    n = len(tok(p, add_special_tokens=False)["input_ids"])
    if n <= MAX_TOKENS:
        return p, False
    ids = tok(enunciado, add_special_tokens=False)["input_ids"]
    corte = n - MAX_TOKENS + 16
    return monta_prompt(tok, tok.decode(ids[corte:]), alts), True


def ids_letras(tok) -> dict[str, list[int]]:
    """Ids das variantes de tokenização de cada letra: 'A' e ' A'."""
    out = {}
    for L in LETRAS:
        ids = set()
        for v in (L, " " + L):
            t = tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
        # SentencePiece (Phi/Llama): tok("A") dá "▁A", mas depois de "(" o modelo emite o "A" cru -> pega os dois
        for v in (L, "▁" + L, "Ġ" + L):
            i = tok.convert_tokens_to_ids(v)
            if i is not None and i != tok.unk_token_id:
                ids.add(i)
        out[L] = sorted(ids)
    return out


def carregar_modelo(nome: str):
    repo, prec = MODELOS[nome]
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": {"": 0}}
    if prec == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["dtype"] = torch.float16
    modelo = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
    return tok, modelo


@torch.no_grad()
def rodar_modelo(nome: str, df: pd.DataFrame, lote: int):
    arq = SAIDA / f"{nome}.parquet"
    feito = pd.read_parquet(arq) if arq.exists() else pd.DataFrame()
    ja = set(feito["id"]) if len(feito) else set()
    pend = df[~df["id"].isin(ja)]
    print(f"\n== {nome}: {len(ja)} já feitas, {len(pend)} pendentes ==")
    if pend.empty:
        return
    tok, modelo = carregar_modelo(nome)
    ids = ids_letras(tok)
    print("ids das letras:", ids, "| VRAM em uso (GB): %.1f" % (torch.cuda.memory_allocated() / 1e9))

    # prepara prompts e ordena por tamanho (menos padding por lote)
    itens = []
    for r in pend.itertuples():
        alts = {L: getattr(r, "alt_" + L) for L in LETRAS if pd.notna(getattr(r, "alt_" + L))}
        p, trunc = prompt_seguro(tok, r.enunciado, alts)
        itens.append({"id": r.id, "answer": r.answer.strip().upper(), "letras": list(alts), "prompt": p,
                      "truncada": trunc, "n_tokens": len(tok(p, add_special_tokens=False)["input_ids"])})
    itens.sort(key=lambda d: d["n_tokens"])
    print(f"truncadas: {sum(d['truncada'] for d in itens)} de {len(itens)}")

    novos, t0 = [], time.time()
    for i in tqdm(range(0, len(itens), lote), desc=nome):
        b = itens[i:i + lote]
        enc = tok([d["prompt"] for d in b], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        logits = modelo(**enc, logits_to_keep=1).logits[:, -1, :].float()  # só a última posição (poupa VRAM)
        probs_vocab = torch.softmax(logits, dim=-1)
        for j, d in enumerate(b):
            # soma das probabilidades das variantes de cada letra, renormalizada só entre as letras existentes
            pl = np.array([probs_vocab[j, ids[L]].sum().item() for L in d["letras"]])
            massa = pl.sum()
            pl = pl / massa if massa > 0 else np.full(len(pl), 1 / len(pl))
            linha = {"id": d["id"], "n_tokens": d["n_tokens"], "truncada": d["truncada"], "massa_letras": massa,
                     "letra_escolhida": d["letras"][int(pl.argmax())], "resposta": d["answer"]}
            for L in LETRAS:
                linha["p_" + L] = pl[d["letras"].index(L)] if L in d["letras"] else np.nan
            linha["acertou"] = linha["letra_escolhida"] == d["answer"]
            linha["p_correta"] = linha.get("p_" + d["answer"], np.nan)
            novos.append(linha)
        if (i // lote) % 10 == 9:  # salvamento incremental
            pd.concat([feito, pd.DataFrame(novos)]).to_parquet(arq, index=False)
    seg = time.time() - t0
    final = pd.concat([feito, pd.DataFrame(novos)])
    final.to_parquet(arq, index=False)
    nv = pd.DataFrame(novos)
    print(f"{nome}: {seg:.0f}s no total | {seg / len(nv):.3f} s/questão | acurácia {nv.acertou.mean():.3f} "
          f"| p_correta média {nv.p_correta.mean():.3f} | massa média nas letras {nv.massa_letras.mean():.2f} "
          f"| pico de VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del modelo, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", action="store_true", help="roda só 50 questões")
    ap.add_argument("--amostra", type=int, help="roda N questões aleatórias")
    ap.add_argument("--modelos", nargs="+", default=["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"],  # opção C aprovada
                    choices=list(MODELOS))
    ap.add_argument("--dataset", default="data/escopo.parquet")
    ap.add_argument("--lote", type=int, default=8)
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    df = carregar_questoes(a.dataset, a.teste, a.amostra)
    print(f"{len(df)} questões únicas do Revalida")
    for nome in a.modelos:
        rodar_modelo(nome, df, a.lote)


if __name__ == "__main__":
    main()
