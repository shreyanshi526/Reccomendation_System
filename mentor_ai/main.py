from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from sentence_transformers import SentenceTransformer
from mentor_ai.core.embedding import generate_embeddings
from mentor_ai.core.faiss_index import build_index, load_index, search_similar, get_mentor_by_id

app = FastAPI()

# 🔁 Reuse models and index
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    index = load_index()
except FileNotFoundError:
    print("Index not found. Building a new one...")
    index = build_index()

# ✅ Request body model
class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 3  # optional, default to 3

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/search")
def search_mentors(request: QueryRequest):
    query = request.query_text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    query_embedding = model.encode([query])[0]
    results = search_similar(query_embedding, top_k=request.top_k)

    enriched_results = []
    for mentor_id, distance in results:
        mentor_doc = get_mentor_by_id(mentor_id)
        if mentor_doc:
            enriched_results.append({
                "mentor": jsonable_encoder(mentor_doc),
                "distance": float(distance)
            })
        else:
            enriched_results.append({
                "mentor_id": mentor_id,
                "distance": float(distance),
                "error": "Mentor not found in DB"
            })

    return enriched_results

