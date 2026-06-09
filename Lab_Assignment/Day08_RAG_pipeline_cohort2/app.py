import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# Thêm src vào path để import
import sys
sys.path.append(str(Path(__file__).parent))

from src.task10_generation import generate_with_citation

load_dotenv()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Hệ thống RAG Pháp luật Ma tuý v2",
    page_icon="⚖️",
    layout="wide",
)

# Custom CSS cho giao diện premium
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-tag {
        background-color: #f0f2f6;
        color: #31333f;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================
with st.sidebar:
    st.title("⚙️ Cấu hình RAG")
    
    st.info("Hệ thống đang chạy chế độ **TF-IDF Fallback** do lỗi DLL cục bộ.")
    
    top_k = st.slider("Số lượng Chunks (Top K)", min_value=1, max_value=20, value=5)
    score_threshold = st.slider("Ngưỡng điểm (Threshold)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    st.divider()
    st.header("🔑 API Keys Status")
    gemini_ok = bool(os.getenv("GOOGLE_API_KEY"))
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    
    st.write(f"- Gemini: {'✅' if gemini_ok else '❌'}")
    st.write(f"- OpenAI: {'✅' if openai_ok else '❌'}")
    
    if st.button("Xoá lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# MAIN UI - CHAT INTERFACE
# =============================================================================
st.title("⚖️ Drug Law RAG Assistant")
st.subheader("Hệ thống tư vấn pháp luật ma tuý Việt Nam v2")

# Khởi tạo history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Xem nguồn tài liệu"):
                for src in message["sources"]:
                    meta = src.get("metadata", {})
                    st.write(f"- **{meta.get('source', 'Unknown')}** (Score: {src.get('score', 0):.3f})")
                    st.caption(src.get("content", "")[:300] + "...")

# Chat input
if prompt := st.chat_input("Hỏi về luật ma tuý hoặc nghệ sĩ liên quan..."):
    # 1. Thêm user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý RAG
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Đang tìm kiếm tài liệu và suy nghĩ...")
        
        try:
            # Gọi pipeline từ Task 10
            # Lưu ý: generate_with_citation đã bao gồm cả retrieve (Task 9)
            result = generate_with_citation(prompt, top_k=top_k)
            
            answer = result["answer"]
            sources = result["sources"]
            ret_src = result.get("retrieval_source", "hybrid")
            
            # Hiển thị câu trả lời
            message_placeholder.markdown(answer)
            
            # Hiển thị nguồn trong expander
            if sources:
                with st.expander("📚 Nguồn tài liệu đã dùng"):
                    st.write(f"Tìm thấy từ: **{ret_src.upper()}**")
                    for s in sources:
                        meta = s.get("metadata", {})
                        st.write(f"- **{meta.get('source', 'Unknown')}** (Type: {meta.get('type', '?')})")
                        st.caption(s.get("content", "")[:200] + "...")

            # 3. Lưu assistent message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"Đã có lỗi xảy ra: {str(e)}")
            message_placeholder.markdown("Xin lỗi, tôi không thể trả lời câu hỏi lúc này.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Day 08 Lab - RAG Pipeline v2 | Google DeepMind Agentic Coding Team")
