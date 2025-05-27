from sentence_transformers import SentenceTransformer
from mentor_ai.core.embedding import generate_embeddings
from mentor_ai.core.faiss_index import build_index,load_index,search_similar

def main():
    # Get both embeddings and mentor_ids
    embeddings, mentor_ids = generate_embeddings(return_ids=True)

    # Load or build FAISS index
    try:
        print("loading index")
        index = load_index()
    except FileNotFoundError:
        print("Index not found. Building a new one...")
        index = build_index()

    # # Create a query embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "reactjs"
    query_embedding = model.encode([query_text])[0]

    # # Search top 3 similar mentors
    results = search_similar(query_embedding, top_k=3)

    # print("Top similar mentors:")
    for mentor_id, distance in results:
        print(f"Mentor ID: {mentor_id}, Distance: {distance:.4f}")

if __name__ == "__main__":
    main()
