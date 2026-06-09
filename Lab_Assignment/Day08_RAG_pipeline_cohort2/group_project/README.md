# Group Project — RAG Pipeline v2

Chào mừng đến với dự án nhóm của chúng tôi về **Hệ thống tư vấn pháp luật Ma tuý Việt Nam**.

## 🚀 Tính năng nổi bật
1.  **Giao diện Chatbot Streamlit**: Hiện đại, dễ sử dụng, hỗ trợ lịch sử chat.
2.  **Trích dẫn nguồn (Citations)**: Câu trả lời luôn đi kèm nguồn gốc tin cậy (Luật pháp/Tin tức).
3.  **Hybrid Search + RRF**: Kết hợp tìm kiếm ngữ nghĩa và từ khoá tối ưu.
4.  **Hệ thống đánh giá tự động (Evaluation)**: Sử dụng DeepEval để đo lường độ chính xác.

## 📁 Cấu trúc thư mục
- `app.py`: Giao diện chính của ứng dụng.
- `src/`: Mã nguồn core (Task 1-10).
- `group_project/evaluation/`:
    - `golden_dataset.json`: Bộ 15 câu hỏi mẫu.
    - `eval_pipeline.py`: Script chạy đánh giá tự động.
    - `results.md`: Báo cáo kết quả đánh giá.

## 🛠️ Hướng dẫn cài đặt & Chạy

1.  **Cài đặt dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install streamlit deepeval
    ```

2.  **Cấu hình API Key**:
    - Chỉnh sửa file `.env` và điền `GOOGLE_API_KEY` (hoặc `OPENAI_API_KEY`).

3.  **Chạy Chatbot**:
    ```bash
    streamlit run app.py
    ```

4.  **Chạy Đánh giá (Evaluation)**:
    ```bash
    python group_project/evaluation/eval_pipeline.py
    ```

## 📊 Kết quả Đánh giá (A/B Test)
Hệ thống đã được đánh giá qua 2 cấu hình:
- **Config A (Dense Search Only)**: Độ phủ thấp đối với câu hỏi từ khoá chính xác.
- **Config B (Hybrid + RRF)**: Tăng 25% điểm Context Recall cho các câu hỏi về định lượng ma tuý.

---
*Thực hiện bởi: Nhóm RAG Pipeline Cohort 2*
