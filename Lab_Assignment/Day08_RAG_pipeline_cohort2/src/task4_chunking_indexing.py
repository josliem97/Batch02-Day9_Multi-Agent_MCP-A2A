"""
Task 4 — Chunking & Indexing với Pure Python Strategy.
Tránh hoàn toàn các thư viện nặng (torch, transformers, langchain) để không bị lỗi DLL.
"""

import pickle
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
VECTORIZER_PATH = CHROMA_DB_DIR / "tfidf_vectorizer.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def simple_recursive_split(text, chunk_size, chunk_overlap):
    """Một implement đơn giản thay cho RecursiveCharacterTextSplitter."""
    if not text: return []
    
    # Ưu tiên tách theo: đoạn (\n\n) -> dòng (\n) -> câu (. ) -> từ ( ) -> từng ký tự
    separators = ["\n\n", "\n", ". ", " "]
    
    def split_text(text, seps):
        if len(text) <= chunk_size:
            return [text]
        
        if not seps:
            # Nếu không còn separator nào, bắt buộc cắt theo độ dài
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks
        
        sep = seps[0]
        if sep not in text:
            return split_text(text, seps[1:])
            
        parts = text.split(sep)
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            # Check if adding this part would overflow
            potential_chunk = (current_chunk + sep + part) if current_chunk else part
            if len(potential_chunk) > chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    # Bắt đầu chunk mới với overlap
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = (overlap_text + sep + part) if overlap_text else part
                    # Nếu chính phần (overlap + part) vẫn quá dài, phải đệ quy tiếp
                    if len(current_chunk) > chunk_size:
                        # Pop last chunk và giải quyết lại
                        final_chunks.pop()
                        # Đệ quy cho đoạn text này với các separator nhỏ hơn
                        # Hoặc đơn giản là cắt nhỏ part
                        pass # Sẽ xử lý bằng recursive check bên dưới
                else:
                    # Nếu một part duy nhất đã quá dài
                    current_chunk = part
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            final_chunks.append(current_chunk)
            
        # Recursive check: if any chunk is still too long, split it with smaller separators
        verified_chunks = []
        for c in final_chunks:
            if len(c) > chunk_size:
                verified_chunks.extend(split_text(c, seps[1:]))
            else:
                verified_chunks.append(c)
        return verified_chunks

    return split_text(text, separators)

def load_documents() -> list[dict]:
    documents = []
    if not STANDARDIZED_DIR.exists(): return []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip(): continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({"content": content, "metadata": {"source": md_file.name, "type": doc_type}})
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        splits = simple_recursive_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [c["content"] for c in chunks]
    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1024)
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks

def index_to_vectorstore(chunks: list[dict]):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try: client.delete_collection("DrugLawDocs")
    except: pass
    coll = client.get_or_create_collection(name="DrugLawDocs", metadata={"hnsw:space": "cosine"})
    
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        coll.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch]
        )
    print(f"✓ Indexed {len(chunks)} chunks to ChromaDB")

def get_vectorizer():
    if VECTORIZER_PATH.exists():
        with open(VECTORIZER_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_or_create_collection(name="DrugLawDocs")

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        chunks = embed_chunks(chunks)
        index_to_vectorstore(chunks)
    else:
        print("Không có documents để index.")
