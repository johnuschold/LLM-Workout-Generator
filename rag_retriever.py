from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os

# === Config (adjust paths if yours differ) ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH_EXERCISE = "./faiss_index_exercises"
# If you also have a nutrition index, set its CSV path here too:
# SAVE_PATH_NUTRITION = "./faiss_index_nutrition"

CSV_EXERCISES = "data/exercises_data_final.csv"
# CSV_NUTRITION = "data/nutrition_data_final.csv"

# Single embeddings instance
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _build_vectorstore_from_csv(csv_path: str, save_path: str) -> FAISS:
    """(Re)build a FAISS store from the CSV under the current runtime versions."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Vectorstore rebuild requested but CSV not found at: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Choose columns to index; fall back sensibly if names differ
    name_col = "Exercise Name" if "Exercise Name" in df.columns else df.columns[0]
    text_col = "Combined_Text" if "Combined_Text" in df.columns else (
        df.columns[1] if len(df.columns) > 1 else df.columns[0]
    )

    names = df[name_col].fillna("").astype(str)
    bodies = df[text_col].fillna("").astype(str)
    texts = (names + " — " + bodies).tolist()

    metadatas = [{"name": n} for n in names]

    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)
    return vs


def _csv_for_path(save_path: str) -> str:
    """Map a FAISS folder path to its source CSV (so we don’t change your app code)."""
    norm = os.path.normpath
    if norm(save_path) == norm(SAVE_PATH_EXERCISE):
        return CSV_EXERCISES
    # If you also handle nutrition, uncomment:
    # if norm(save_path) == norm(SAVE_PATH_NUTRITION):
    #     return CSV_NUTRITION
    raise RuntimeError(
        f"No CSV mapping defined for vectorstore path: {save_path}. "
        "Add a mapping in _csv_for_path()."
    )


def load_vectorstore(path: str):
    """Drop-in replacement: load pickled FAISS, or rebuild from CSV if incompatible."""
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        # Typical Pydantic v1/v2 or pickle schema mismatch when loading old FAISS folders
        if "__fields_set__" in str(e) or "pickle" in str(e) or "Pydantic" in str(e):
            csv_path = _csv_for_path(path)
            return _build_vectorstore_from_csv(csv_path, path)
        raise


def retrieve_similar_documents(query: str, vectorstore, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]