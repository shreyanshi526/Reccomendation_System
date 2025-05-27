from mentor_ai.db.connection import get_db
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(return_ids=False):
    db = get_db()
    collection = db.Mentors

    mentors = list(collection.find({'techStack': {"$ne": None}}))
    count = len(mentors)
    print(f"Mentors with techstack: {count}")

    texts = []
    mentor_ids = []

    for mentor in mentors:
        professional_title = mentor.get("professionalTitle", "")
        specialization = mentor.get("specialization", "")
        tech_stack = ", ".join(mentor.get("techStack", [])) if isinstance(mentor.get("techStack"), list) else ""
        combined = f"{professional_title}. {specialization}. Tech Stack: {tech_stack}"
        texts.append(combined.strip())
        mentor_ids.append(str(mentor["_id"]))

    embeddings = model.encode(texts, show_progress_bar=True)

    for i, emb in enumerate(embeddings):
        if not isinstance(emb, (list, np.ndarray)):
            print(f"❌ Invalid type at {i}: {type(emb)}")
        elif len(emb) != 384:
            print(f"❌ Invalid vector length at {i}: {len(emb)}")
        elif any(e is None or isinstance(e, str) for e in emb):
            print(f"❌ Non-numeric or None value at {i}: {emb[:5]}")
        else:
            pass  # Good embedding

    return (embeddings, mentor_ids) if return_ids else embeddings
