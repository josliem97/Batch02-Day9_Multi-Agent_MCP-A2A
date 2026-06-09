"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25Okapi (rank-bm25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import numpy as np
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# BM25 Index — khởi tạo lazy khi module được import
# =============================================================================

_corpus: list[dict] = []   # List of {'content': str, 'metadata': dict}
_bm25 = None               # BM25Okapi instance


def _load_corpus_from_files() -> list[dict]:
    """Load corpus từ data/standardized/*.md."""
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
            }
        })
    return corpus


def build_bm25_index(corpus: list[dict] = None):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
                Nếu None, tự load từ data/standardized/
    """
    global _corpus, _bm25
    from rank_bm25 import BM25Okapi

    if corpus is None:
        corpus = _load_corpus_from_files()

    _corpus = corpus

    # Tokenize — split đơn giản (đủ cho BM25 tiếng Việt không dấu)
    # Chuyển về lowercase để tăng recall
    tokenized_corpus = [
        doc["content"].lower().split()
        for doc in _corpus
    ]

    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def _ensure_index():
    """Đảm bảo BM25 index đã được build."""
    global _bm25
    if _bm25 is None:
        build_bm25_index()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (>=0)
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()

    if not _corpus:
        return []

    # Tokenize query (lowercase để match với corpus)
    tokenized_query = query.lower().split()

    # Tính BM25 scores
    scores = _bm25.get_scores(tokenized_query)

    # Lấy top_k indices (sorted descending)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:  # Chỉ trả về kết quả có score dương
            results.append({
                "content": _corpus[idx]["content"],
                "score": round(score, 4),
                "metadata": _corpus[idx]["metadata"],
            })

    # Đảm bảo sorted descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Build index và test
    print("Building BM25 index...")
    build_bm25_index()
    print(f"Corpus size: {len(_corpus)} documents")

    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "nghệ sĩ bị bắt ma tuý TP.HCM",
        "hình phạt tù năm",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = lexical_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] [{r['metadata'].get('type', '?')}] {r['content'][:80]}...")
