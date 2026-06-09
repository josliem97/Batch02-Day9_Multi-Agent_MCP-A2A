"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve relevant chunks (Task 9)
    2. Reorder chunks để tránh "lost in the middle" effect
    3. Format context với source labels cho citation
    4. Build prompt (system + context + query)
    5. Call LLM (Gemini → OpenAI → template fallback)
    6. Return answer có citation + sources

Lựa chọn tham số:
    - TOP_K = 5: đủ evidence mà không gây "lost in the middle"
    - TOP_P = 0.9: diverse nhưng không quá random (factual RAG cần stable)
    - TEMPERATURE = 0.3: RAG cần factual, ít sáng tạo hơn creative writing
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5: đủ evidence nhiều chiều, không vượt quá context window LLM nhỏ
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# 0.9: giữ lại 90% probability mass → đủ đa dạng nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# 0.3: RAG cần factual accuracy → ít random, nhưng không 0 để tránh lặp
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Trả lời câu hỏi dưới đây một cách toàn diện bằng tiếng Việt.
Với mỗi phát biểu về sự thật hoặc khẳng định, hãy ngay lập tức chèn trích dẫn trong ngoặc vuông
 liên kết đến nguồn cụ thể (ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2023]).

Nếu thông tin không được nêu rõ trong ngữ cảnh được cung cấp, hãy nói rõ 
'Tôi không thể xác minh thông tin này từ nguồn hiện có' thay vì đoán mò.

Quy tắc:
- Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
- Mỗi khẳng định sự thật PHẢI có trích dẫn
- Nếu ngữ cảnh không đủ, hãy nói rõ
- Cấu trúc câu trả lời với đoạn văn rõ ràng"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]  (1 = most relevant)
    Output order:            [1, 3, 5, 4, 2]
    (best first, then odd-indexed, worst in middle, second-best at end)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Tách thành 2 nhóm: odd positions (ưu tiên đầu) và even positions (đặt cuối)
    first_half = chunks[0::2]   # Index 0, 2, 4, ... → đầu
    second_half = chunks[1::2]  # Index 1, 3, 5, ... → cuối (reversed)

    # Pattern: [best, 3rd, 5th, ..., 4th, 2nd]
    reordered = first_half + second_half[::-1]
    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        score = chunk.get("score", 0.0)

        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk['content']}\n"
        )

    return "\n---\n".join(context_parts)


# =============================================================================
# LLM GENERATION
# =============================================================================

def _call_gemini(user_message: str) -> str:
    """Gọi Google Gemini API."""
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY không có")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash"
    )
    # Di chuyển SYSTEM_PROMPT vào user_message để tương thích version cũ
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    )
    return response.text


def _call_openai(user_message: str) -> str:
    """Gọi OpenAI API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY không có")

    client_kwargs = {"api_key": api_key}
    
    # Hỗ trợ OpenRouter nếu key bắt đầu bằng sk-or-
    if api_key.startswith("sk-or-"):
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        # Sử dụng model ổn định nhất trên OpenRouter là gpt-4o-mini
        model_name = "openai/gpt-4o-mini"
    else:
        model_name = "gpt-4o-mini"

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content


def _generate_template_response(query: str, chunks: list[dict]) -> str:
    """
    Template response khi không có API key.
    Vẫn có citation dạng đúng format.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    sources_info = []
    content_summary = []
    for i, chunk in enumerate(chunks[:3], 1):
        source = chunk.get("metadata", {}).get("source", f"Tài liệu {i}")
        source_label = source.replace(".md", "").replace("-", " ").title()
        sources_info.append(f"[{source_label}]")
        content_preview = chunk["content"][:150].strip()
        content_summary.append(f"- {content_preview}... [{source_label}]")

    answer = (
        f"Dựa trên các tài liệu pháp luật và tin tức được cung cấp:\n\n"
        + "\n".join(content_summary)
        + f"\n\nThông tin trên được tổng hợp từ các nguồn: {', '.join(sources_info)}.\n\n"
        f"Để biết thêm chi tiết chính xác, vui lòng tham khảo trực tiếp các văn bản pháp luật liên quan."
    )
    return answer


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM (Gemini → OpenAI → template fallback)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3 & 4: Format context + build user message
    context = format_context(reordered)
    user_message = f"Ngữ cảnh:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Step 5: Call LLM (cascade: Gemini → OpenAI → template)
    answer = None
    for llm_fn, name in [(_call_gemini, "Gemini"), (_call_openai, "OpenAI")]:
        try:
            answer = llm_fn(user_message)
            print(f"  ✓ Generated via {name}")
            break
        except Exception as e:
            print(f"  ⚠ {name} không khả dụng: {e}")

    if answer is None:
        print("  ⚠ Không có API key, dùng template response")
        answer = _generate_template_response(query, reordered)

    # Step 6: Return
    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "none"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_src,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
