"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + TF-IDF fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank kết quả merged
    4. Nếu top result score < threshold → fallback sang pageindex (TF-IDF)
    5. Return top_k results
"""

import os
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.01  # Giảm ngưỡng cho phù hợp với TF-IDF (scores thường rất nhỏ)
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"   # "rrf" (default, no API) | "cross_encoder" | "mmr"


def _generate_hyde_query(query: str) -> str:
    """Tạo câu trả lời giả định (HyDE) để cải thiện retrieval."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return query
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Bạn là chuyên gia pháp luật. Hãy viết một đoạn trả lời ngắn (khoảng 2-3 câu) giả định cho câu hỏi sau: {query}. Lưu ý: Chỉ viết nội dung trả lời, không cần dẫn nhập."
        response = model.generate_content(prompt)
        print(f"  ✨ HyDE Query Generated")
        return response.text
    except:
        return query

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = True, # BẬT HyDE mặc định để lấy điểm bonus
) -> list[dict]:
    # Step 0: HyDE (Bonus)
    semantic_query = query
    if use_hyde:
        semantic_query = _generate_hyde_query(query)

    # Step 1: Chạy semantic + lexical search
    try:
        dense_results = semantic_search(semantic_query, top_k=top_k * 3)
    except Exception as e:
        print(f"  ⚠ Semantic search lỗi: {e}")
        dense_results = []

    try:
        # Lexical search vẫn dùng query gốc
        sparse_results = lexical_search(query, top_k=top_k * 3)
    except Exception as e:
        print(f"  ⚠ Lexical search lỗi: {e}")
        sparse_results = []

    # Step 2: Merge bằng RRF (từ nhiều ranked lists)
    all_lists = [l for l in [dense_results, sparse_results] if l]

    if not all_lists:
        # Cả hai đều không có kết quả → fallback ngay
        print(f"  ⚠ Không có kết quả hybrid, fallback → TF-IDF")
        fallback = pageindex_search(query, top_k=top_k)
        for item in fallback:
            item["source"] = "pageindex"
        return fallback

    merged = rerank_rrf(all_lists, top_k=top_k * 2, k=60)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback
    if not final_results or final_results[0]["score"] < score_threshold:
        best_score = final_results[0]["score"] if final_results else 0.0
        print(
            f"  ⚠ Hybrid best score ({best_score:.4f}) < threshold ({score_threshold}). "
            f"Fallback → TF-IDF pageindex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for item in fallback:
                item["source"] = "pageindex"
            # Giữ cả hybrid results nếu có, ưu tiên fallback
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2023",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    print("=" * 70)
    print("Task 9: Retrieval Pipeline Test")
    print("=" * 70)

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] [{r['metadata'].get('type', '?')}]")
            print(f"     {r['content'][:80]}...")
