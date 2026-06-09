"""
Task 7 — Reranking Module.

Implement Reciprocal Rank Fusion (RRF) làm phương pháp chính.

RRF hoạt động thế nào:
    - Nhận nhiều ranked lists từ các retriever khác nhau
    - Mỗi document được tính: RRF(d) = Σ 1 / (k + rank_r(d))
    - k=60 là hằng số làm mượt (Cormack et al. 2009)
    - Kết quả: merge hiệu quả không cần normalize scores từ các retriever khác nhau

Lợi thế của RRF:
    - Không cần API/model thêm
    - Robust: không bị ảnh hưởng bởi scale khác nhau của BM25 và cosine score
    - Đơn giản nhưng hiệu quả (thường tốt hơn simple weighted average)
"""

from typing import Optional


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}  # content key → RRF score
    content_map: dict[str, dict] = {}  # content key → full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            # Dùng content làm key (normalize: strip + lower 50 chars)
            key = item["content"].strip()[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content_key, score in sorted_items[:top_k]:
        item = content_map[content_key].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API).
    Fallback sang RRF nếu không có API key.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        # Fallback: dùng original scores (không rerank thực sự)
        print("  ⚠ JINA_API_KEY không có, dùng original scores")
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]

    try:
        import requests as req

        response = req.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": round(r["relevance_score"], 4)}
            for r in reranked
        ]
    except Exception as e:
        print(f"  ⚠ Jina API lỗi: {e}. Dùng original scores")
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    if not candidates:
        return []

    # Nếu candidates không có embedding, fallback sang RRF scores
    if "embedding" not in candidates[0]:
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"]
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # Default: RRF (không cần API key)
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking ("rrf" | "cross_encoder" | "mmr")

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "rrf":
        # Với RRF, wrap candidates vào 1 ranked list
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr([], candidates, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test với dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {"source": "luat.md"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý tại TP.HCM", "score": 0.7, "metadata": {"source": "news.md"}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ ma tuý", "score": 0.6, "metadata": {"source": "luat.md"}},
        {"content": "Python programming best practices", "score": 0.4, "metadata": {"source": "other.md"}},
    ]

    print("=== Test RRF Reranking ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="rrf")
    for r in results:
        print(f"  [{r['score']:.6f}] {r['content']}")

    print("\n=== Test RRF Fusion (2 lists) ===")
    list1 = dummy_candidates[:3]
    list2 = [dummy_candidates[2], dummy_candidates[0]]
    fused = rerank_rrf([list1, list2], top_k=3)
    for r in fused:
        print(f"  [{r['score']:.6f}] {r['content'][:60]}")
