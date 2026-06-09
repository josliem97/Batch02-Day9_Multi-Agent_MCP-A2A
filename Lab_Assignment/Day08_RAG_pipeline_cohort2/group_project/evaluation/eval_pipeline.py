import os
import json
from pathlib import Path
from dotenv import load_dotenv
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Thêm src vào path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.task10_generation import generate_with_citation

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

def run_evaluation():
    """
    Chạy evaluation cho RAG pipeline sử dụng DeepEval.
    """
    print("=" * 50)
    print("Starting RAG Evaluation Pipeline")
    print("=" * 50)

    if not GOLDEN_DATASET_PATH.exists():
        print(f"Error: Golden dataset not found at {GOLDEN_DATASET_PATH}")
        return

    # 1. Load Golden Dataset
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 2. Tạo test cases bằng cách chạy pipeline thực tế
    test_cases = []
    print(f"Running pipeline for {len(golden_data)} test cases...")
    
    for i, item in enumerate(golden_data):
        print(f" [{i+1}/{len(golden_data)}] Query: {item['question']}")
        
        # Gọi RAG pipeline
        result = generate_with_citation(item["question"])
        
        # Chuẩn bị dữ liệu cho DeepEval
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item.get("expected_answer", ""),
            retrieval_context=[c["content"] for c in result["sources"]]
        )
        test_cases.append(test_case)

    # 3. Định nghĩa Metrics
    # Lưu ý: DeepEval dùng GPT-4 mặc định, ta cần config gemini nếu muốn tiết kiệm
    # Tuy nhiên ở bản này ta giả định USER đã có OPENAI_API_KEY hoặc dùng default
    
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7)
    ]

    # 4. Chạy Evaluate
    print("\nEvaluating metrics...")
    results = evaluate(test_cases, metrics)

    # 5. Xuất báo cáo kết quả ra Markdown
    save_results_to_markdown(results)
    print(f"\n✓ Evaluation complete! Results saved to {RESULTS_PATH}")

def save_results_to_markdown(results):
    """Lưu kết quả evaluation vào file kết quả."""
    content = "# RAG Evaluation Results\n\n"
    content += "| Metric | Score | Status |\n"
    content += "| --- | --- | --- |\n"
    
    # Tính điểm trung bình (ví dụ thô)
    # Trong thực tế kết quả của deepeval.evaluate trả về một list các kết quả chi tiết
    
    content += "\n## Detailed Results\n\n"
    for i, res in enumerate(results):
        content += f"### Test Case {i+1}\n"
        content += f"- **Input:** {res.input}\n"
        content += f"- **Output:** {res.actual_output[:200]}...\n"
        content += f"- **Success:** {res.success}\n\n"
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # Đăng ký Gemini với DeepEval nếu có key
    if os.getenv("GOOGLE_API_KEY"):
        # Import lazy để tránh lỗi nếu không cài langchain_google_genai
        try:
            from deepeval.models.gpt_model import GPTModel
            # DeepEval hỗ trợ custom models, nhưng config mặc định thường dùng OpenAI.
            # Ở đây ta giả định môi trường lab đã cài sẵn hoặc user sẽ cung cấp key.
            pass
        except:
            pass
            
    run_evaluation()
