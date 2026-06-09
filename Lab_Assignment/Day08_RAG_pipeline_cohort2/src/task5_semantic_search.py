"""
Task 5 — Semantic Search Module với TF-IDF Fallback.
"""

from src.task4_chunking_indexing import get_vectorizer, get_collection

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    # Lấy vectorizer đã được train ở Task 4
    vectorizer = get_vectorizer()
    if vectorizer is None:
        return []
    
    query_embedding = vectorizer.transform([query]).toarray()[0].tolist()
    
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    output = []
    if results and results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: sim = 1 - dist
            similarity = 1.0 - dist
            output.append({
                "content": doc,
                "score": round(float(similarity), 4),
                "metadata": meta
            })
    
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]

if __name__ == "__main__":
    results = semantic_search("hình phạt tàng trữ ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
