

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Must match what you used when building the vector store
SAVE_PATH_EXERCISE = "./faiss_index_exercises"
SAVE_PATH_NUTRITION = "./faiss_index_nutrition"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)



def load_vectorstore(path: str):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)



def retrieve_similar_documents(query: str, vectorstore, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]



