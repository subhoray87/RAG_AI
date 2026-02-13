import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY_NEW")

MONGO_URI = "mongodb+srv://subhoray87_db_user:owUuRhttrBf3Po3x@rag.rtev49n.mongodb.net/?appName=RAG"
DB_NAME = "sample_mflix"
COLLECTION_NAME = "RHpdf"
INDEX_NAME = "vector_index"

# =========================
# CONNECT MONGO
# =========================
client=MongoClient(
    "mongodb+srv://subhoray87_db_user:owUuRhttrBf3Po3x@rag.rtev49n.mongodb.net/?appName=RAG",
    tls=True,
    tlsAllowInvalidCertificates=True)
collection=client["sample_mflix"]["RHpdf"]

# =========================
# LOAD EMBEDDINGS
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# =========================
# LOAD LLM (Groq)
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# =========================
# VECTOR SEARCH FUNCTION
# =========================
def vector_search(query, top_k=5):
    query_vector = embedding_model.embed_query(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 200,
                "limit": top_k
            }
        },
        {
            "$project": {
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    return results

# =========================
# BUILD CONTEXT
# =========================
def build_context(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"\n\n[Document {i} | Score: {doc['score']:.4f}]\n"
        context += doc["text"]
    return context

# =========================
# RAG PIPELINE
# =========================
def rag_query(user_query):

    # Step 1: Retrieve
    retrieved_docs = vector_search(user_query)

    if not retrieved_docs:
        return "No relevant documents found."

    # Step 2: Build context
    context = build_context(retrieved_docs)

    # Step 3: Create Prompt
    prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{user_query}
"""

    # Step 4: Call LLM
    response = llm([HumanMessage(content=prompt)])

    return response.content

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":

    while True:
        query = input("\nAsk your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag_query(query)

        print("\n=== ANSWER ===\n")
        print(answer)
