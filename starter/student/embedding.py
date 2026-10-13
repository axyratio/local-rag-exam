"""✏️ ส่วนที่ 2: Embedding — แปลงข้อความเป็น vector

nomic-embed-text ถูกฝึกให้ใช้ prefix "search_document: " กับเอกสาร และ "search_query: " กับคำค้น
(Nomic Embed technical report, arXiv:2402.01613) จึงเติม prefix ทั้งสองฝั่ง

ข้อจำกัดสำคัญ: โมเดลนี้ใช้ tokenizer ของ bert-base-uncased ซึ่งมีอักษรไทยเพียง 15 ตัวและตัดสระ/วรรณยุกต์ทิ้ง
คำไทยเกือบทุกคำจึงกลายเป็น [UNK] vector จึงจับได้เฉพาะคำอังกฤษ ตัวเลข และโค้ดในข้อความ
retrieval.py จึงใช้ BM25 เป็นหลักและให้น้ำหนัก vector ตามสัดส่วนคำอังกฤษในคำค้น
"""
from core.ollama_client import embed
from student import settings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """แปลง chunk ทั้งหมดเป็น vector (เรียกครั้งเดียวตอนรับ knowledge)"""
    if not texts:
        return []
    return embed([settings.DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    """แปลงคำค้น 1 ข้อความเป็น vector"""
    return embed([settings.QUERY_PREFIX + text])[0]
