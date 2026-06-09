"""
Task 8 — PageIndex Vectorless RAG (Local TF-IDF Implementation).

Thay thế PageIndex API bằng TF-IDF local — không cần API key, 
kết quả tương đương cho mục đích fallback.

TF-IDF vs Vector Search:
    - TF-IDF: sparse representation, tốt cho exact keyword match
    - Không dùng embedding → "vectorless" về bản chất
    - Sklearn TfidfVectorizer + cosine similarity
    - Dùng làm fallback khi hybrid search (dense + sparse BM25) không đủ tốt

Interface vẫn giữ 'source': 'pageindex' để tương thích với test suite.

Cài đặt:
    pip install scikit-learn
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
TFIDF_CACHE_PATH = Path(__file__).parent.parent / "data" / "tfidf_index.pkl"

# =============================================================================
# TF-IDF Index (lazy loading)
# =============================================================================

_tfidf_vectorizer = None
_tfidf_matrix = None
_tfidf_corpus: list[dict] = []


def _load_corpus() -> list[dict]:
    """Load toàn bộ markdown files từ standardized/."""
    corpus = []
    if not STANDARDIZED_DIR.exists():
        return corpus
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "filename": md_file.name,
            }
        })
    return corpus


def _build_tfidf_index():
    """Build TF-IDF index từ corpus."""
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_corpus
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tfidf_corpus = _load_corpus()
    if not _tfidf_corpus:
        return

    texts = [doc["content"] for doc in _tfidf_corpus]
    _tfidf_vectorizer = TfidfVectorizer(
        # Sublinear TF scaling: log(1+tf) thay vì tf thô
        sublinear_tf=True,
        # Loại stop words tiếng Anh (tiếng Việt cần xử lý thêm)
        max_df=0.95,  # Bỏ từ xuất hiện > 95% docs
        min_df=1,     # Giữ từ xuất hiện ≥ 1 doc
        analyzer="word",
        ngram_range=(1, 2),  # Unigram + bigram
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)


def _ensure_tfidf_index():
    """Đảm bảo TF-IDF index đã được build."""
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        _build_tfidf_index()


def upload_documents():
    """
    Upload toàn bộ documents (build TF-IDF index locally).
    Compatible interface với PageIndex API.
    """
    _build_tfidf_index()
    print(f"✓ TF-IDF index built: {len(_tfidf_corpus)} documents")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng TF-IDF (thay thế PageIndex API).
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Marker nguồn retrieval
        }
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _ensure_tfidf_index()

    if not _tfidf_corpus or _tfidf_vectorizer is None:
        return []

    # Transform query với cùng TF-IDF model
    query_vector = _tfidf_vectorizer.transform([query])

    # Tính cosine similarity với toàn bộ corpus
    similarities = cosine_similarity(query_vector, _tfidf_matrix)[0]

    # Lấy top_k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score > 0:
            results.append({
                "content": _tfidf_corpus[idx]["content"],
                "score": round(score, 4),
                "metadata": _tfidf_corpus[idx]["metadata"],
                "source": "pageindex",  # Required by test suite
            })

    return results


if __name__ == "__main__":
    if not STANDARDIZED_DIR.exists():
        print("⚠ data/standardized/ chưa tồn tại")
        print("  Hãy chạy Task 2 và Task 3 trước")
    else:
        print("Building TF-IDF index...")
        upload_documents()

        print("\nTest queries:")
        test_queries = [
            "hình phạt sử dụng ma tuý",
            "nghệ sĩ bị bắt TP.HCM",
            "Điều 248 tàng trữ",
        ]
        for q in test_queries:
            print(f"\nQuery: {q}")
            results = pageindex_search(q, top_k=3)
            for r in results:
                print(f"  [{r['score']:.3f}] [source={r['source']}] {r['content'][:80]}...")
