"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

วิธีที่ใช้: hybrid search แบบ convex combination (Bruch et al., 2023: ดีกว่า RRF เมื่อปรับน้ำหนักได้)
    score = (1 - w) * BM25_norm + w * cosine_norm
  - BM25 ตัดคำไทยด้วย newmm + bigram (lexical.py) เป็นช่องทางหลัก
    วัดกับชุดทดสอบ 62 ข้อ: hit@5 0.55 (ตัดตามช่องว่าง) -> 0.98 (ตัดคำไทย) เมื่อค้นด้วยคำถามอย่างเดียว
  - w ปรับตามสัดส่วนคำอังกฤษ/ตัวเลขในคำค้น เพราะ nomic-embed-text อ่านภาษาไทยไม่ออก (เป็น [UNK])
    คำค้นภาษาไทยล้วนจึงแทบไม่ใช้ vector ส่วนคำค้นที่มีชื่อเฉพาะ/โค้ดภาษาอังกฤษจะใช้ vector มากขึ้น
"""
import re

from core.types import Chunk
from core.utils import cosine_many
from student import lexical, settings

_LATIN_WORD = re.compile(r"[A-Za-z0-9_]+")
_THAI_RUN = re.compile(r"[\u0E00-\u0E7F]+")
_index_cache: dict = {}


def make_query(question: str, options: list[str]) -> str:
    """คำค้น = คำถาม + ข้อความของทุกตัวเลือก (hit@1 0.87 -> 0.94 บนชุดทดสอบ เทียบกับคำถามอย่างเดียว)"""
    parts = [question.strip()]
    if settings.QUERY_WITH_OPTIONS:
        parts += [o.strip() for o in options if o and o.strip()]
    return "\n".join(p for p in parts if p) or question or "?"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """คืน chunk ที่เกี่ยวข้องที่สุด (สูงสุด settings.TOP_K ชิ้น) เรียงจากคะแนนมากไปน้อย"""
    if not chunks:
        return []
    bm25 = _minmax(_bm25(chunks).scores(lexical.tokenize(query_text)))

    weight = 0.0
    vectors = [c.vector for c in chunks]
    if query_vector is not None and all(v is not None for v in vectors) and settings.VECTOR_WEIGHT > 0:
        weight = settings.VECTOR_WEIGHT * _latin_share(query_text)
    if weight > 0:
        dense = _minmax(cosine_many(query_vector, vectors))
        scores = [(1 - weight) * b + weight * d for b, d in zip(bm25, dense)]
    else:
        scores = bm25

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top_k = max(1, int(settings.TOP_K))
    if len(chunks) <= top_k:  # เอกสารน้อยกว่า top-k: ส่งทุก chunk (เรียงตามคะแนน) ไม่ต้องกรอง
        return [(chunks[i], float(scores[i])) for i in order]
    best = scores[order[0]] or 1.0
    floor = settings.MIN_SCORE if settings.MIN_SCORE is not None else 0.0
    hits = [(chunks[i], float(scores[i])) for i in order
            if scores[i] >= floor and scores[i] >= settings.MIN_RELATIVE_SCORE * best]
    return hits[:top_k]


def _bm25(chunks: list[Chunk]) -> lexical.BM25:
    """สร้าง index ครั้งเดียวต่อชุด chunk แล้วเก็บไว้ (token ของแต่ละ chunk เก็บใน chunk.meta)"""
    key = (settings.THAI_TOKENIZER, tuple(hash(c.text) for c in chunks))
    if key not in _index_cache:
        docs = []
        for c in chunks:
            if c.meta.get("tok_mode") != settings.THAI_TOKENIZER:
                c.meta["tokens"] = lexical.tokenize(c.text)
                c.meta["tok_mode"] = settings.THAI_TOKENIZER
            docs.append(c.meta["tokens"])
        if len(_index_cache) > 8:
            _index_cache.clear()
        _index_cache[key] = lexical.BM25(docs)
    return _index_cache[key]


def _latin_share(text: str) -> float:
    """สัดส่วนของข้อความที่ nomic-embed-text อ่านได้ (อักษรอังกฤษ/ตัวเลข) เทียบกับอักษรไทย"""
    latin = sum(len(w) for w in _LATIN_WORD.findall(text))
    thai = sum(len(w) for w in _THAI_RUN.findall(text)) / 2.5  # คำไทยยาวกว่าคำอังกฤษโดยเฉลี่ย
    return latin / (latin + thai) if latin + thai else 0.0


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi > lo:
        return [(v - lo) / (hi - lo) for v in values]
    return [1.0 if hi > 0 else 0.0 for _ in values]  # คะแนนเท่ากันหมด: ถ้ามีคำตรงกันให้ถือว่าเกี่ยวข้องเท่ากัน
