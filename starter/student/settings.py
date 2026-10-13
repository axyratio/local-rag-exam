"""✏️ ค่าที่ปรับจูนได้ — รวมไว้ที่เดียว

ค่าที่เลือกมาจากการวัดผล (ดู docs/RESEARCH.md) ตัวเลขในวงเล็บคือผลบนชุดทดสอบภาษาไทย 62 ข้อ (tools/thai_pack)
"""

# ---------- Chunking (student/chunking.py) ----------
CHUNK_SIZE = 1000           # ความยาวสูงสุดของ chunk (ตัวอักษร) — หัวข้อส่วนใหญ่ในเอกสารเรียนสั้นกว่านี้ จึงได้ 1 หัวข้อต่อ chunk
CHUNK_OVERLAP = 150         # ซ้อนบล็อกท้าย (ย่อหน้า/ตาราง) ไม่เกินค่านี้ไปยัง chunk ถัดไป (~15%)
SPLIT_BY_HEADING = True     # แบ่งตามหัวข้อ markdown ก่อน แล้วค่อยรวมบล็อกตามความยาว
INCLUDE_HEADING = True      # ใส่เส้นทางหัวข้อ "ชื่อเอกสาร > หัวข้อ" ไว้บรรทัดแรก (hit@1 0.919 -> 0.935)

# ---------- Embedding (student/embedding.py) ----------
DOC_PREFIX = "search_document: "    # prefix ที่ nomic-embed-text ใช้กับเอกสาร
QUERY_PREFIX = "search_query: "     # prefix ที่ nomic-embed-text ใช้กับคำค้น

# ---------- Retrieval (student/retrieval.py) ----------
TOP_K = 6                   # จำนวน chunk สูงสุดที่ส่งต่อให้ prompt (prompt ตัดเพิ่มตามงบ context)
QUERY_WITH_OPTIONS = True   # รวมข้อความตัวเลือก ก–จ ในคำค้น (hit@1 0.871 -> 0.935)
VECTOR_WEIGHT = 0.5         # น้ำหนัก vector สูงสุด (ใช้เต็มค่าเมื่อคำค้นเป็นอังกฤษล้วน, ใกล้ 0 เมื่อเป็นไทยล้วน)
MIN_SCORE = None            # ตัด chunk ที่คะแนนรวมต่ำกว่านี้ (None = ไม่ตัด)
MIN_RELATIVE_SCORE = 0.15   # ตัด chunk ที่คะแนนต่ำกว่า 15% ของอันดับ 1

# ---------- Lexical search (student/lexical.py) ----------
THAI_TOKENIZER = "newmm+bigram"   # "newmm" | "bigram" | "trigram" | "newmm+bigram"
NGRAM_STRIP_MARKS = False         # ตัดสระบน/ล่างและวรรณยุกต์ออกก่อนทำ bigram

# ---------- Prompt (student/prompts.py) ----------
MAX_CONTEXT_CHARS = 9000    # เพดานความยาวเอกสารใน prompt (ตัวอักษร)
NUM_CTX = 8192              # context ของโมเดลที่ core ตั้งไว้
RESERVED_TOKENS = 1500      # เผื่อไว้ให้ system prompt, คำถาม และคำตอบ (นอกเหนือจากที่ประเมินจากความยาวคำถาม)
