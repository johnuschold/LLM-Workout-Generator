# From Gym Notes to an AI Coach: Building a RAG‑Powered Workout Planner with Streamlit

> > **TL;DR**: This app turns fitness goals into personalized workout plans. It uses a retrieval-augmented generation (RAG) pipeline (FAISS + MiniLM embeddings) to ground an LLM (Mistral-7B Instruct), then enriches the plan with fuzzy-matched exercise details and demo videos—all wrapped in a Streamlit interface.

---

Most “AI workout” tools spit out generic, one‑size‑fits‑none routines. I wanted something opinionated yet flexible: a planner that respect real constraints like available equipment, split preference, and time budget. 

Key Characteristics:
- **Transparency**: no hallucinated exercises; pull real instructions and cues.
- **Personalization**: goals (hypertrophy, fat loss, strength), schedule, experience level.
- **Local first**: a vector store I could fully control and iterate on.
- **Deployable**: run as a Streamlit app.

---
## DEMO

[Streamlit](https://llm-workout-generator-mx9nrbg9gnsruunemprjr8.streamlit.app)

## The Core Idea

Instead of asking an LLM to invent a plan from scratch, I **retrieve relevant knowledge** first—exercise instructions, coaching cues, and common mistakes—then ask the model to compose a plan that fits the user. This is classic RAG: **Retrieve → Read → Generate**.

- **Retriever:** FAISS over embeddings from `all‑MiniLM‑L6‑v2`.
- **Generator:** `mistralai/Mistral‑7B‑Instruct‑v0.2` via `transformers`.
- **Matcher:** `thefuzz` to map the plan’s exercise names back to a curated dataset (for videos and instructions).
- **UI:** Streamlit for fast iteration and clean UX.

---

## Architecture at a Glance

```
[Streamlit UI]
     ├─ Collect goals, constraints, split preferences
     └─ Show generated plan + expandable exercise cards

[Query Rewriter] → cleans/expands the user's request
      ▼
[RAG Retriever] → FAISS + MiniLM embeddings over local corpus
      ▼
[Prompt Builder] → composes system + user messages with retrieved snippets
      ▼
[LLM Pipeline] → Mistral‑7B Instruct (Transformers pipeline)
      ▼
[Post‑processing] → parse exercises, fuzzy‑match names → dataset rows
      ▼
[UI Enrichment] → instructions, cues, video links
```

Key design decision: **query rewriting** before retrieval. People describe goals messily (“get leaner but keep strength, 4 days, no overhead pressing”). A small rewrite step yields better search terms and retrieval.

---

## Data & Scraping (Ethics First)

This project’s dataset comes from **MuscleWiki**, collected in two stages with Selenium:

- **`scrapers/01-basics-scraper.py`**
  - Navigates the paginated exercise directory.
  - Captures exercise names (and variants), difficulty, equipment tags, and links to detail pages.
  - Runs headless with built-in throttling (`time.sleep` delays) to avoid hammering the server.

- **`scrapers/01-detail-scraper.py`**
  - Visits each detail page to pull:
    - Step-by-step instructions and coaching cues
    - Target muscles
    - Exercise type and difficulty classification
    - Embedded video URLs
  - Merges results into a single dataset.

The scrapers **respect robots.txt** and Terms of Service — no bypasses, scraping limits, or automated account creation.

After scraping, I use **Jupyter notebooks** to clean,normalize and explore the data:

- **`02-exercise-data-preprocessing.ipynb`** — basic exploratory data preprocessing, identifying bad rows and inconsistent naming.
- **`03-exercise-EDA.ipynb`** — basic exploratory data analysis.


Final cleaned CSV:

```
data/exercises_data_final.csv
```

---

## Embedding & Indexing

The embeddings pipeline is entirely in-repo and uses **LangChain Community** components, so no closed-source external services are required.

1. **Embedding model**:
   - `all-MiniLM-L6-v2` from `sentence-transformers` (via `langchain_community.embeddings.HuggingFaceEmbeddings`)

2. **Indexing**:
   - Uses FAISS (`langchain_community.vectorstores.FAISS`) with `allow_dangerous_deserialization=True` for reloads.
   - vectorstores are created:
     - `faiss_index_exercises/` — main exercise KB


Once built, indexes are saved locally so you never recompute embeddings between runs.

---

## Retrieval That Actually Helps the LLM

In `main.py` (main Streamlit app), when a user requests a workout plan:

1. The app collects:
   - **Goal** (`muscle gain`, `weight loss`, `health`)
   - **Fitness level**
   - **Workout split** (six supported, each with allowed days and target muscles)
   - **Available equipment**
   - **Special instructions** (free-text)

2. Before retrieval, the query goes through **`get_improved_rag_query`** (`LLMgen.py`):
   - Uses the LLM itself to **rewrite** the request into **positive-only search terms**.
   - Strips all **negative constraints** — not even synonyms of excluded items remain (avoids false matches in bag-of-words search).
   - Canonicalizes terms to dataset tokens (e.g., “hamstrings” → “hamstring”).
   - Includes equipment terms only if the user has them.
   - Returns a compact (~8–20 token) string with **no punctuation except commas/spaces**.


3. The refined query is passed to `retrieve_similar_documents` (`rag_retriever.py`):
   ```python
   docs = vectorstore.similarity_search(query, k=20)
   ```
   Result: a **list of short exercise descriptions**, ordered by cosine similarity.

---


## Prompting the Planner

Workout generation happens in **`create_workout_prompt`**:

- Pulls in:
  - User profile (goal, level, split, days, muscles, equipment, special notes)
  - Retrieved exercise context
- Generates **split-specific day sequencing**:
  - **Upper/Lower** alternates
  - **PPL** cycles Push → Pull → Legs
  - **Bro Split** assigns one muscle group per day
  - **PHUL** blends Power and Hypertrophy
  - Defaults to **Full Body** if no split match
- Adjusts exercise count and extra components:
  - More exercises/day for weight loss or health goals
  - Cardio duration based on goal × level
  - Stretching requirements for health goal
- Embeds explicit formatting rules so the LLM output is:
  - Markdown headers for each day
  - Bullet lists of exercises with `Sets × Reps`
  - Ready for downstream parsing

Example output snippet:
```markdown
### Day 1: Push
**Exercises:**
* Bench Press: 3 sets of 8–12 reps
* Overhead Press: 3 sets of 10–12 reps
* Triceps Pushdown: 3 sets of 10–15 reps
```

---

## Name Matching & Exercise Display

After the LLM generates the plan, the app:

1. **Extracts exercise names** with regex from the markdown plan.
2. For each name, runs fuzzy matching (`thefuzz`):
   - **First pass**: `token_sort_ratio` with cutoff 75
   - **Fallback**: `partial_ratio` with cutoff 85
3. On match, retrieves:
   - Cleaned instructions (`get_clean_instructions`)
   - Video URL (if present)
4. Displays in the UI:
   - Markdown instructions
   - Embedded video (via `st.video`)
   - Match confidence (“Matched to ‘Incline Dumbbell Press’ with score 88”)

---
