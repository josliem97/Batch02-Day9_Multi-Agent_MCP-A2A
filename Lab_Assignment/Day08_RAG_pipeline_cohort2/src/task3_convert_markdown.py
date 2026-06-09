"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Lưu ý: Đã đổi từ MarkItDown sang pypdf/python-docx do lỗi onnxruntime trên môi trường Windows.
Cài đặt:
    pip install pypdf python-docx
"""

import json
import re
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

def clean_text(text: str) -> str:
    """Làm sạch text: xoá khoảng trắng thừa, giữ lại ký tự in được."""
    if not text:
        return ""
    # Giữ lại các ký tự tiếng Việt và ký tự in được cơ bản
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(filepath: Path) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(filepath: Path) -> str:
    from docx import Document
    try:
        doc = Document(filepath)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_doc_fallback(filepath: Path) -> str:
    """
    Fallback cho file .doc cũ.
    Vì không có thư viện đọc .doc native dễ dùng, ta thử đọc binary và lọc text.
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        # Lọc các ký tự ASCII in được và khoảng trắng
        text = "".join(chr(b) if 32 <= b <= 126 or b in (10, 13) else " " for b in data)
        # Lọc các từ tiếng Việt thô (nếu có) - thực tế file .doc rất khó đọc kiểu này
        # Trả về thông báo lỗi nhẹ nhàng nếu kết quả quá tệ
        if len(re.sub(r'\s+', '', text)) < 50:
            return f"Note: File {filepath.name} là định dạng .doc cũ, hãy chuyển sang .docx để có kết quả tốt nhất."
        return text
    except Exception as e:
        return f"Error reading .doc: {str(e)}"

def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in legal_dir.iterdir():
        if filepath.name.startswith("."): continue
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        content = ""
        if filepath.suffix.lower() == ".pdf":
            content = extract_text_from_pdf(filepath)
        elif filepath.suffix.lower() == ".docx":
            content = extract_text_from_docx(filepath)
        elif filepath.suffix.lower() == ".doc":
            # Thử convert DOCX nếu có thể, hoặc dùng fallback
            content = extract_text_from_doc_fallback(filepath)
        
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(f"# {filepath.name}\n\n{clean_text(content)}", encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name}")
        converted += 1

    print(f"  → Converted {converted} legal documents")

def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ data/landing/news/ chưa tồn tại, bỏ qua")
        return

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header
            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            content = data.get("content_markdown", "")

            header = (
                f"# {title}\n\n"
                f"**Source:** {url}\n"
                f"**Crawled:** {date_crawled}\n\n"
                f"---\n\n"
            )

            full_content = header + content
            output_path.write_text(full_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(full_content)} chars)")
            converted += 1
        except Exception as e:
            print(f"  ⚠ Error converting {filepath.name}: {e}")

    print(f"  → Converted {converted} news articles")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    # Kiểm tra kết quả
    md_files = list(OUTPUT_DIR.rglob("*.md"))
    print(f"\n✓ Done! Tổng cộng {len(md_files)} markdown files tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
