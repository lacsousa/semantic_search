"""Interface Streamlit do sistema de busca de questões médicas (Revalida).

Uso: .venv/bin/streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from sistema import FAIXAS, Sistema, formatar_alternativas  # noqa: E402

st.set_page_config(page_title="Busca de questões do Revalida", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Carregando modelo de embeddings e índice...")
def carregar() -> Sistema:
    return Sistema()


sis = carregar()

st.title("🩺 Busca de questões do Revalida")
st.caption("Digite uma dúvida ou um caso clínico. O sistema devolve questões de provas passadas semelhantes "
           "(busca vetorial por similaridade de cosseno).")

pergunta = st.text_area("Sua dúvida", height=110,
                        placeholder="Ex.: paciente com dor torácica e suspeita de infarto agudo do miocárdio")

c1, c2, c3 = st.columns([1, 2, 2])
k = c1.slider("Nº de questões", 1, 15, 5)
areas = c2.multiselect("Filtrar por área (opcional)", sis.areas)
faixas = c3.multiselect("Filtrar por dificuldade estimada (opcional)", FAIXAS,
                        help="Faixas são tercis da dificuldade prevista. Ex.: marque só 'Difícil' para as mais difíceis.")

if st.button("Buscar", type="primary", disabled=not pergunta.strip()):
    with st.spinner("Buscando..."):
        res = sis.buscar(pergunta.strip(), k=k, areas=areas or None, faixas=faixas or None)

    pa = res["area_pergunta"]
    st.subheader("Área prevista da sua pergunta")
    st.write(f"**{pa.index[0]}** ({pa.iloc[0]:.0%}) · " + " · ".join(f"{a}: {p:.0%}" for a, p in pa.iloc[1:3].items()))
    if pa.iloc[0] < 0.5:
        st.info("Confiança baixa na área: a pergunta é curta ou ambígua. Os resultados abaixo não dependem desse palpite.")

    r = res["resultados"]
    st.subheader(f"{len(r)} questões encontradas")
    if r.empty:
        st.warning("Nenhuma questão nos candidatos mais próximos atende aos filtros. Tente remover algum filtro.")
    for n, (_, q) in enumerate(r.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"**{n}. Similaridade {q.similaridade:.3f}** · Revalida {q.year}")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"**Área:** {q.area_final}  \n_{q.area_origem}, confiança {q.area_confianca:.0%}_")
            m2.markdown(f"**Cluster temático:** {q.cluster_nome}")
            m3.markdown(f"**Dificuldade estimada:** {q.dificuldade_faixa}  \n_índice {q.dificuldade_prevista:.2f}_")
            st.write(q.enunciado)
            for L in "ABCDE":
                t = q.get(f"alt_{L}")
                if t is None or t != t:
                    continue
                if str(q.answer).strip().upper() == L:
                    st.markdown(f":green[**({L}) {str(t).strip()}  ✔ gabarito**]")
                else:
                    st.markdown(f"({L}) {str(t).strip()}")

with st.expander("Limitações"):
    st.markdown(
        "- A **área** das questões do Revalida é **prevista** por um classificador treinado em questões do Enare "
        "(o Revalida não tem rótulo de área); acurácia ≈ 77% no Enare, não medida no Revalida.\n"
        "- A **dificuldade** é estimada a partir da dificuldade para um painel de LLMs (proxy, não a dificuldade para estudantes) "
        "e o modelo explica só ~5–7% da variação. Use a faixa como indicação aproximada.\n"
        "- Filtros de área e dificuldade são aplicados entre as 400 questões mais próximas da busca."
    )
