# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

School project (course: agentes inteligentes): semantic search over a database of questions. Given a topic/query, find the most relevant questions. The deliverable is a **single Jupyter notebook** that will be ported to **Google Colab**.

Current state: [semantic_search.ipynb](semantic_search.ipynb) only has setup and dataset download/caching cells. Everything under "Planned design" is intent from the project owner, not existing code — update this file as the notebook takes shape. The `graft/` directory is tooling, not project code.

## Dataset: ENEM questions (2022–2024)

Source: Hugging Face `maritaca-ai/enem`, loaded with `load_dataset("maritaca-ai/enem", "<year>")` for `2022`, `2023`, `2024` (one config per year, a single `train` split, 180 questions each → 540 total). The notebook caches the concatenated result at `data/enem.parquet` (gitignored; regenerated if missing).

Columns: `id` (`questao_01`…`questao_180`, unique only within a year), `exam` (year), `IU`, `ledor` (string flags `'True'`/`'False'` in the raw data), `question` (Portuguese text, sometimes English/Spanish source text, may start with `[[placeholder]]` or a `## title` markdown header), `description` (list of image descriptions, often empty), `alternatives` (list of 5 strings), `label` (correct letter A–E, or `'Anulado'` for one annulled question per year), `figures` (list of image URLs, often empty).

Things to know:

- **There is no subject/area column and no difficulty column.** `label` is the answer key, not a topic. Any topic label must be derived (e.g. from question position: ENEM order is roughly Linguagens 1–45, Ciências Humanas 46–90, Ciências da Natureza 91–135, Matemática 136–180 — spot checks of 2022–2024 matched, but verify before relying on it) or from clustering.
- Question `id` is not globally unique; use `(exam, id)` as the key.
- Many questions depend on figures (`[[placeholder]]`, `figures`); text-only embeddings miss that content. Decide explicitly whether to append `description` text to the embedded text.

## Dataset: HealthQA-BR

Source: `load_dataset("Larxel/healthqa-br")` (public; no HF login was needed), single `train` split, 5,632 Portuguese health-exam questions. Cached at `data/healthqa.parquet`.

Columns: `id` (unique), `source` (`Enare Residência Médica` 2,691 / `Revalida` 1,777 / `Enare Multiprofissional` 1,164), `year` (2011–2025, but 4,038 of the rows are 2024), `group` (medical specialty, 52 distinct values), `question` (statement with alternatives embedded in the text; mean ~860 chars), `answer` (letter A–E).

Things to know:

- `group` is **null for all 1,777 Revalida rows** and populated for every Enare row — usable as a classification label only on the 3,855 Enare rows (and note it is a mix of overlapping names, e.g. "Saúde coletiva" vs "Medicina de Família, Preventiva e Saúde Coletiva").
- Unlike ENEM, alternatives are not a separate column; they are inside `question`.
- It is not decided whether this dataset replaces or complements ENEM; confirm with the owner before merging them into one index.

## Assignment requirements

The homework (originally in Portuguese) asks for a functionality that combines all three ML task types:

- **Prediction (regression)**
- **Classification (supervised learning)**
- **Clustering (unsupervised learning)**

It should preferably serve the group's *Projeto Aplicado* (PA) and address demands that will also appear in the end-of-course delivery. The notebook must therefore demonstrably contain all three tasks, not just clustering-based search.

Proposed mapping onto semantic search (not confirmed with the owner — verify before building around it):

- **Clustering**: k-means over question embeddings (topic discovery + search routing, see below).
- **Classification**: predict a question's topic/category (or difficulty) from its embedding, using cluster labels or existing dataset labels as targets; can also be used to filter search results by predicted topic.
- **Regression**: predict a numeric attribute of a question (e.g. difficulty score, expected answer time/length, or relevance score for re-ranking), if the dataset has such a column.

If the dataset lacks labels or numeric targets, say so and ask the owner before fabricating them.

## Hard constraints

- **Colab compatibility is mandatory.** The notebook must run top-to-bottom on a fresh Colab runtime. Install dependencies in a `!pip install` cell; do not assume anything is preinstalled on the host.
- **No system-installed services or containers.** Databases/indexes must be plain files (e.g. SQLite, FAISS/NumPy index files, parquet/npy, ChromaDB persistent dir) — no Postgres, Docker, etc. The only accepted exceptions are few and deliberate (e.g. a local LLM via llama / llama.cpp).
- Use paths that work both locally and in Colab (relative paths, or a single configurable `DATA_DIR`; Drive mounting must be optional).
- Keep the notebook reproducible: fix random seeds (k-means, sampling), and keep outputs of heavy steps cacheable to files so reruns are cheap.

## Planned design

1. **Ingestion**: load the questions, embed them, and store text + embeddings + index in file-based storage.
2. **Search function**: `search(query, k)` returning the top-k questions for a topic query.
3. **Baseline approach — k-means**: cluster the embeddings, route a query to the nearest centroid(s), and rank only within those clusters.
4. **Alternative approaches to compare** (this is a school project; the comparison/analysis is part of the grade): e.g. brute-force exact cosine similarity (ground truth), FAISS flat/IVF/HNSW, BM25/TF-IDF lexical search, hybrid/re-ranking. Keep every approach behind the same `search` interface so they can be benchmarked interchangeably.
5. **Analysis**: compare approaches on recall@k against exact search, query latency, index build time/memory, and qualitative result inspection; include plots and short written conclusions in markdown cells.

## Working with the notebook

- Edit `.ipynb` files through the NotebookEdit tool (or `jupytext`-style round-tripping if introduced), not by hand-editing raw JSON.
- Keep cells ordered so "Restart & Run All" works; no hidden state between cells.
- Before finishing a change, verify the notebook executes end-to-end (e.g. `jupyter nbconvert --to notebook --execute <name>.ipynb`) once the notebook exists.
