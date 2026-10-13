"""BM25 สำหรับภาษาไทยปนอังกฤษ (ใช้ใน retrieval.py)

nomic-embed-text ใช้ tokenizer ของ bert-base-uncased ซึ่งแทบไม่มีอักษรไทย คำไทยเกือบทุกคำจึงกลายเป็น [UNK]
การค้นด้วยคำ (BM25) จึงเป็นช่องทางหลักของข้อความภาษาไทย

tokenize แบบผสม:
  - ภาษาไทย: ตัดคำด้วย PyThaiNLP (newmm) ถ้ามี และเพิ่ม bigram ของตัวอักษร (ทนต่อการตัดคำผิด)
  - อังกฤษ/ตัวเลข/สัญลักษณ์โค้ด: ตัวพิมพ์เล็ก แยกตามตัวอักษรและตัวเลข เช่น "BB-SAVE-26" -> bb, save, 26
"""
import math
import re
from collections import Counter

from student import settings

_THAI_RUN = re.compile(r"[\u0E00-\u0E7F]+")
_LATIN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
_THAI_MARKS = re.compile(r"[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]")  # สระบน/ล่างและวรรณยุกต์

try:  # PyThaiNLP อยู่ใน requirements.txt; ถ้า import ไม่ได้ให้ใช้ n-gram อย่างเดียว
    from pythainlp.tokenize import word_tokenize as _thai_words
except Exception:  # noqa: BLE001
    _thai_words = None

_STOP = set("ที่ และ ใน ของ เป็น ได้ ให้ มี การ ว่า ไม่ จะ ด้วย จาก ซึ่ง หรือ แต่ คือ ไว้ กับ แล้ว ก็ นี้ ตาม ใด ข้อ "
            "อะไร อย่างไร เท่าใด กี่ the a an of to is in and or for".split())


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    mode = settings.THAI_TOKENIZER
    for match in _THAI_RUN.finditer(text):
        run = match.group()
        if mode in ("newmm", "newmm+bigram") and _thai_words is not None:
            tokens += [w for w in _thai_words(run, engine="newmm", keep_whitespace=False) if w not in _STOP]
        if mode in ("bigram", "newmm+bigram") or (_thai_words is None and mode == "newmm"):
            bare = _THAI_MARKS.sub("", run) if settings.NGRAM_STRIP_MARKS else run
            tokens += ["#" + bare[i:i + 2] for i in range(len(bare) - 1)] or ["#" + bare]
        if mode == "trigram":
            tokens += ["#" + run[i:i + 3] for i in range(len(run) - 2)] or ["#" + run]
    tokens += [t.replace(",", "") for t in _LATIN.findall(_THAI_RUN.sub(" ", text)) if t not in _STOP]  # 1,000 = 1000
    return tokens


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) บน chunk ทั้งหมดของ namespace"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 1.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = []
        for tf, length in zip(self.tf, self.len):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
            s = 0.0
            for term, qf in q.items():
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm) * (1 + math.log(qf))
            out.append(s)
        return out
