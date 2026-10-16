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
    """คืน chunk ที่เกี่ยวข้องที่สุด

    คลังที่มีทั้งเอกสารภาษาไทยและเอกสารภาษาอื่น (อังกฤษ/จีน/อาหรับ) ใช้ _search_groups
    คลังภาษาเดียวใช้ _search_hybrid (วิธีเดิม)
    """
    if not chunks:
        return []
    thai, other = _split_by_language(chunks)
    if settings.GROUP_SEARCH and thai and other:
        return _search_groups(chunks, thai, other, query_text, query_vector)
    return _search_hybrid(chunks, query_text, query_vector, k)


def _search_groups(chunks: list[Chunk], thai: list[int], other: list[int], query_text: str,
                   query_vector: list[float] | None) -> list[tuple[Chunk, float]]:
    """ค้นแยกเอกสารไทยกับเอกสารภาษาอื่น แล้วเอาชิ้นที่ดีที่สุดของทั้งสองกลุ่มมารวมกัน

    คำถามเป็นภาษาไทยแต่เอกสารอาจเป็นอังกฤษ/จีน/อาหรับ BM25 ที่ตัดคำไทยจะดึงคำถามไปหาเอกสารไทยเสมอ
    (วัดบนชุด 100 ข้อ: ค้นเจอเอกสารถูก 5/50 เมื่อเอกสารไม่ใช่ไทย) จึงให้
      - เอกสารไทย: จัดอันดับด้วย BM25 ตัดคำไทย (แม่นกับภาษาไทย)
      - เอกสารอื่น: จัดอันดับด้วย BM25 เฉพาะคำอังกฤษ/ตัวเลขในคำค้น + cosine ของ vector
    แล้วเรียงกลุ่มที่ vector คล้ายคำค้นมากกว่าไว้ก่อน (prompt วางชิ้นที่เกี่ยวข้องที่สุดไว้ต้น)
    """
    tokens = lexical.tokenize(query_text)
    index = _bm25(chunks)
    bm25_all = _minmax(index.scores(tokens))
    latin_tokens = [t for t in tokens if not t.startswith("#") and not _THAI_RUN.search(t)]
    bm25_latin = _minmax(index.scores(latin_tokens)) if latin_tokens else [0.0] * len(chunks)

    vectors = [c.vector for c in chunks]
    if query_vector is not None and all(v is not None for v in vectors):
        cosine = [float(x) for x in cosine_many(query_vector, vectors)]
        dense = _minmax(cosine)
    else:
        cosine, dense = None, [0.0] * len(chunks)

    w = settings.GROUP_VECTOR_WEIGHT
    thai_scores = {i: bm25_all[i] for i in thai}
    other_scores = {i: (1 - w) * bm25_latin[i] + w * dense[i] for i in other}
    groups = [(_top_of_group(thai_scores, chunks), thai), (_top_of_group(other_scores, chunks), other)]
    if cosine is not None:  # กลุ่มที่มีชิ้นคล้ายคำค้นที่สุด (cosine ดิบ เทียบข้ามภาษาได้) มาก่อน
        groups.sort(key=lambda g: max(cosine[i] for i in g[1]), reverse=True)
    return [(chunks[i], float(score)) for top, _ in groups for i, score in top]


def _top_of_group(scores: dict[int, float], chunks: list[Chunk]) -> list[tuple[int, float]]:
    """เลือกชิ้นที่ดีที่สุดของกลุ่ม ตัดชิ้นที่ไม่มีคำตรงเลยและชิ้นที่คะแนนต่ำกว่า MIN_RELATIVE_SCORE ของอันดับ 1

    ถ้า MAX_PER_SECTION > 0 จะไม่เอาชิ้นจากหัวข้อเดียวกันเกินจำนวนนั้นในรอบแรก (ให้หลักฐานกระจายหลายหัวข้อ)
    แล้วเติมที่ว่างที่เหลือด้วยชิ้นที่ถูกข้ามตามลำดับคะแนน จำนวนรวมจึงไม่ลดลง
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return []
    best = ranked[0][1]
    kept = [(i, s) for i, s in ranked if s > 0 and s >= settings.MIN_RELATIVE_SCORE * best]
    limit = max(1, int(settings.GROUP_TOP_K))
    cap = int(settings.MAX_PER_SECTION)
    if cap <= 0:
        return kept[:limit]
    picked: list[tuple[int, float]] = []
    skipped: list[tuple[int, float]] = []
    per_section: dict[tuple[str, str], int] = {}
    for i, s in kept:
        key = (chunks[i].doc_id, chunks[i].text.split("\n", 1)[0])  # บรรทัดแรกของ chunk = เส้นทางหัวข้อ
        if per_section.get(key, 0) < cap and len(picked) < limit:
            per_section[key] = per_section.get(key, 0) + 1
            picked.append((i, s))
        else:
            skipped.append((i, s))
    picked += skipped[:limit - len(picked)]
    return picked


def _split_by_language(chunks: list[Chunk]) -> tuple[list[int], list[int]]:
    """แยกตำแหน่ง chunk เป็น (เอกสารไทย, เอกสารอื่น) ตามสัดส่วนอักษรไทยของทั้งเอกสาร (ไม่ใช่ทีละ chunk)"""
    key = ("lang", settings.THAI_DOC_SHARE, tuple(hash(c.text) for c in chunks))
    if key not in _index_cache:
        thai_chars: dict[str, int] = {}
        total_chars: dict[str, int] = {}
        for c in chunks:
            thai_chars[c.doc_id] = thai_chars.get(c.doc_id, 0) + sum(len(m) for m in _THAI_RUN.findall(c.text))
            total_chars[c.doc_id] = total_chars.get(c.doc_id, 0) + len(c.text)
        is_thai = {d: thai_chars[d] / max(1, total_chars[d]) >= settings.THAI_DOC_SHARE for d in total_chars}
        if len(_index_cache) > 8:
            _index_cache.clear()
        _index_cache[key] = ([i for i, c in enumerate(chunks) if is_thai[c.doc_id]],
                             [i for i, c in enumerate(chunks) if not is_thai[c.doc_id]])
    return _index_cache[key]


def _search_hybrid(chunks: list[Chunk], query_text: str, query_vector: list[float] | None,
                   k: int) -> list[tuple[Chunk, float]]:
    """วิธีเดิม: hybrid BM25 + cosine รวมทุกเอกสาร คืนสูงสุด settings.TOP_K ชิ้น เรียงจากคะแนนมากไปน้อย"""
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
