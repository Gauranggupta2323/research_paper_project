# Research Paper Recommendation & Subject Area Prediction — Interview Master Sheet

*My own words. Quick to re-read before the call.*

---

## 1. Explain your project (why I chose it)

> "I built a system that does two things: recommends similar research papers and predicts a paper's subject area. Researchers waste a lot of time manually searching for related work, and keyword search misses papers that use different words for the same idea. So I used **Sentence Transformers** to turn paper titles into semantic embeddings and rank them with **cosine similarity** to get the Top-5 most relevant papers. For subject prediction, I built an **MLP with sigmoid + binary cross-entropy** on top of TF-IDF bigram features from the abstract, because one paper can belong to multiple subject areas at once. I cleaned and deduplicated **35,000+ arXiv records** as my working dataset and wrapped both pipelines in a Streamlit app."

**Why this project:** practical NLP + deep learning use case, real dataset, and it let me combine embeddings, similarity search, and multi-label classification in one app.

---

## 2. Complete Architecture & Flow

```
arXiv dataset (titles, summaries, terms)
        ↓
Clean data → drop duplicates → filter rare labels
        ↓
   ┌───────────────────────────┬─────────────────────────────┐
   │  RECOMMENDATION PIPELINE   │  SUBJECT PREDICTION PIPELINE │
   ├───────────────────────────┼─────────────────────────────┤
   │ Title                     │ Abstract                     │
   │ Sentence Transformer      │ Keras TextVectorization      │
   │ (all-MiniLM-L6-v2)        │ (TF-IDF, bigrams)             │
   │ → dense embedding         │ → sparse feature vector       │
   │ Cosine similarity vs      │ MLP + Dropout                 │
   │ all stored embeddings     │ Sigmoid output (multi-label)  │
   │ torch.topk() → Top-5      │ multi-hot → label names       │
   └───────────────────────────┴─────────────────────────────┘
        ↓
Saved artifacts (pickle / .h5) loaded once at startup
        ↓
     Streamlit UI → user enters title + abstract → results shown
```

**Walk-through I say out loud:**
1. User enters a paper title and an abstract.
2. Title → embedding via `all-MiniLM-L6-v2` → compared with pre-computed embeddings of ~35K papers → cosine similarity → top-5 via `torch.topk()`.
3. Abstract → vectorized with the saved `TextVectorization` config → passed into the trained Keras MLP → sigmoid probabilities → rounded → converted from multi-hot back to subject names via `invert_multi_hot()`.
4. Both results render in the same Streamlit page.

---

## 3. Tech Stack & Database

| Layer | Tech | Why I picked it |
|---|---|---|
| Language | Python | standard for ML prototyping |
| Data handling | Pandas, NumPy | cleaning 35K+ rows, dedup, label parsing |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) | good semantic quality, lightweight, fast inference |
| Similarity | Cosine similarity + `torch.topk` | measures direction not magnitude — right fit for text embeddings |
| Classification | TensorFlow/Keras (`TextVectorization` + MLP) | simple, fast to train, handles multi-label naturally with sigmoid |
| Persistence | Pickle (`embeddings.pkl`, `sentences.pkl`, `rec_model.pkl`) + `.h5` model | quick to load, no infra needed for a prototype |
| UI | Streamlit | fastest way to demo an ML app |

**"Does the project use a database?"**
> "Not in this prototype — it's a CSV dataset plus serialized model files (pickle/h5), loaded once at startup. That's a deliberate choice for a demo; it avoids infra overhead while I focus on the ML pipeline. In a production version I'd move paper metadata into **PostgreSQL** and the embeddings into a **vector index (FAISS / Pinecone / pgvector)** so similarity search doesn't stay a brute-force O(N×D) scan."

---

## 4. APIs & Modules

**"Which API did you use?"**
> "None externally right now — everything runs on the local pre-processed dataset. I designed it that way so the demo has zero external dependency and zero latency risk during a live interview or presentation."

**Internal modules (app.py):**
- `recommendation(input_paper)` — encodes title, does cosine similarity, returns top-5 titles.
- `predict_category(abstract, model, vectorizer, label_lookup)` — vectorizes abstract, runs MLP, rounds sigmoid output.
- `invert_multi_hot(encoded_labels)` — turns the multi-hot vector back into human-readable subject names.

**Future scope — why an API matters here:**
> "The natural next step is the **arXiv API** (or Semantic Scholar API) so the system pulls live, recent papers instead of a static 2021 snapshot. That keeps recommendations current without me re-scraping data by hand. I'd wrap it behind a **FastAPI** backend so the Streamlit frontend (or any client) just calls REST endpoints instead of loading pickle files directly — cleaner separation, easier to scale later."

---

## 5. Challenges Faced & How I Fixed Them

| Problem | Fix |
|---|---|
| Code expected column `abstracts`, dataset actually had `summaries` | Updated references + added column validation so it fails loudly, not silently |
| `sentence-transformers==2.2.2` broke on `cached_download` import from a newer `huggingface_hub` | Pinned compatible versions together (`huggingface-hub==0.14.1`) |
| `invert_multi_hot()` called before its cell had run in Jupyter | Moved function definition above first use, re-ran notebook top-to-bottom |
| Duplicate titles polluting recommendations | Dropped duplicates on `titles` before generating embeddings |
| 0.5 rounding threshold for multi-label output is a blunt instrument | Kept 0.5 as a working baseline, flagged threshold-tuning on validation data as a future improvement |

**Future scope — why these fixes point to improvements:**
- Add **schema validation** at data-load time → catches column mismatches before they become silent bugs.
- **Pin all dependencies** in `requirements.txt` (already done) to avoid the huggingface_hub-style breakage again.
- Move from a fixed **0.5 threshold** to **per-label thresholds tuned on validation data** → better precision/recall balance for rare subject areas.
- Replace brute-force **cosine similarity over all embeddings** with **FAISS** once the dataset grows past what fits comfortably in memory.

---

## 6. Database Design (if pushed further)

> "Right now it's flat files — a CSV plus pickled embeddings. If I redesigned it as a real system: a `papers` table (id, title, abstract, terms, embedding_id) in PostgreSQL for metadata and filtering, and a separate **vector store** (FAISS index or pgvector column) for the embeddings themselves, since similarity search doesn't belong in a relational engine at scale. The two are joined by `paper_id`."

---

## 7. Concurrency Scenario

**"Two users hit the app at the same time — how do you handle it?"**
> "Both operations here are **read-only** — they read the same loaded model and the same stored embeddings, and each request just creates its own input tensor. So there's no shared-state conflict in the current design. For a production version I'd: load models once and cache them, run inference through **stateless API workers** (multiple processes), and if I ever add writes (e.g. new papers), protect them with **database transactions or optimistic locking** so two updates to the same record don't clobber each other."

---

## 8. Deployment & Future Scope (full list)

**Current deployment flow:**
```
User → Streamlit → Loaded models (pickle/.h5) → Prediction/recommendation → UI
```
Required artifacts: `model.h5`, `embeddings.pkl`, `sentences.pkl`, `rec_model.pkl`, vectorizer config + weights, vocab.

**Future scope (and *why* each one):**
- **FAISS / vector DB** — brute-force cosine similarity is O(N×D); doesn't scale past a few hundred thousand papers.
- **arXiv / Semantic Scholar API** — keeps the paper pool current instead of a static snapshot.
- **FastAPI backend** — separates the ML serving layer from the UI, makes it reusable by other clients, easier to scale/deploy independently.
- **Use abstracts + titles together for recommendation** — title-only similarity can miss nuance that's only in the abstract.
- **Per-label threshold tuning** — improves multi-label precision/recall over the flat 0.5 cutoff.
- **User feedback loop** — thumbs up/down on recommendations to retrain/re-rank over time.
- **Better metrics** — move from binary accuracy to micro/macro F1, Hamming loss, Precision@K/Recall@K/NDCG for honest evaluation.
- **Transformer-based classifier** — swap the MLP for a fine-tuned transformer if accuracy needs to go beyond the current baseline.

---

## 9. Numbers I'm ready to defend

- **35,000+ records processed** — arXiv dataset, deduplicated on `titles` (raw → 38,972 rows after dedup, filtered further for rare labels before train/val/test split).
- **Top-5 recommendations** — `torch.topk(..., k=5)` on cosine similarity scores.
- If asked for a relevance number: *"I evaluated Precision@5 on a sample of representative queries — relevant results in the top-5 divided by 5 — rather than a formal large-scale benchmark. I'd call it a prototype-level measurement, not a production metric."* (Don't invent an exact percentage on the spot if not backed by a saved evaluation — say this is how I'd calculate it and offer to show the method.)

---

## 10. One-line closer if they ask "what would you do differently?"

> "I'd move similarity search to FAISS, add an API layer instead of loading pickles directly, and put more rigor into evaluation — proper Precision/Recall@K instead of manual spot-checks."
