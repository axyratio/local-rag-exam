# Local RAG for Programming — Final Exam

RAG service ที่รันใน Docker ใช้ LLM บนเครื่องผ่าน Ollama ทำ 2 งาน
- **Knowledge** (`POST /knowledge`, `POST /ask`): รับเอกสาร lecture note / case study มาทำดัชนี แล้วตอบข้อสอบปรนัย ก–จ
- **Coding** (`POST /code`): รับโจทย์ภาษาไทย แล้วเขียนฟังก์ชัน Python ที่ผ่าน unit test

สร้างจาก starter kit ของรายวิชา `core/` และ `Dockerfile` ไม่ได้แก้ (ระบบตรวจ checksum) ส่วนที่เขียนเองอยู่ใน `student/` ทั้งหมด
คู่มือเดิมของ starter kit (สัญญา API และกติกา) ย้ายไปไว้ที่ [`docs/STARTER_README.md`](docs/STARTER_README.md)

## โครงสร้าง
```
core/              🔒 ระบบของผู้สอน (API, Ollama client, fallback)
student/
  settings.py      ค่าที่ปรับจูนได้ทั้งหมด พร้อมผลวัดที่ใช้เลือก
  chunking.py      แบ่งเอกสารตามหัวข้อ markdown แล้วรวมบล็อกจนใกล้ CHUNK_SIZE
  lexical.py       ตัดคำไทย (newmm + bigram) และ BM25
  embedding.py     nomic-embed-text พร้อม prefix search_document / search_query
  retrieval.py     hybrid search: BM25 + cosine แบบ convex combination
  prompts.py       prompt ข้อสอบปรนัย และ prompt เขียนโค้ด
docs/RESEARCH.md   ตารางเปรียบเทียบทุกทางเลือก ตัวเลขที่วัดเอง และแหล่งอ้างอิง
docs/RAG_BENCHMARKS.md  ตัวเลขจากงานวิจัย: เทคนิค RAG แบบไหนเร็ว/แม่น และอะไรใช้ได้ในงบ 90 วินาที
docs/CODE_REVIEW.md     ผลรีวิวโค้ด bug ที่พบและวิธีแก้
tools/             selfcheck, doctor, pack และชุดทดสอบ
```

## วิธีทำงาน
```
/knowledge  เอกสาร ─▶ แบ่งตามหัวข้อ (ไม่ตัดกลางตาราง/โค้ด/list) ─▶ BM25 index + nomic vector
/ask        คำถาม+ตัวเลือก ─▶ hybrid search top-k ─▶ prompt ยึดเอกสาร ─▶ LLM (JSON reason→answer) ─▶ ก–จ
/code       โจทย์ไทย ─▶ prompt (signature, ฟังก์ชันที่ห้ามใช้, วางแผนก่อน) ─▶ LLM ─▶ code block เดียว
```

## ทำไมเลือกแบบนี้ (สรุปจากการวัด)
รายละเอียดและแหล่งอ้างอิงอยู่ใน [`docs/RESEARCH.md`](docs/RESEARCH.md)

| เรื่อง | ทางเลือกที่เทียบ | ผล | เลือก |
|---|---|---|---|
| Embedding | nomic-embed-text กับภาษาไทย | ข้อความไทยทุกประโยคเป็น `[UNK]` ได้ cosine 1.000 กันหมด | ใช้ vector เฉพาะคำค้นที่มีอังกฤษ/โค้ด |
| วิธีค้นหา (hit@5, 62 ข้อ) | vector 0.774 / RRF 0.935 / BM25 ตัดคำไทย 1.000 | BM25 ดีที่สุด | BM25 + vector ถ่วงตามสัดส่วนอักษรอังกฤษ |
| ตัดคำ BM25 (hit@5) | ช่องว่าง 0.548 / bigram 0.984 / newmm 0.984 | ต้องตัดคำไทย | newmm + bigram |
| คำค้น (hit@1) | คำถาม 0.871 / คำถาม+ตัวเลือก 0.935 | | คำถาม+ตัวเลือก |
| Chunk | 300 / 500 / 700–1,800 ตัวอักษร | ≥700 ได้ 1 หัวข้อ = 1 chunk | 1,000 + ใส่เส้นทางหัวข้อ |
| ให้คิดก่อนตอบ | บังคับ JSON แล้วตอบทันที vs ให้เหตุผลก่อน | Tam et al. 2024: บังคับ JSON โดยไม่ให้คิด GSM8K 74.7 → 48.9 | `reason` ก่อน `answer` โดยคัดประโยคหลักฐานจากเอกสารก่อน |
| ลำดับใน prompt | ข้อมูลอยู่ต้น 75.8% / กลาง 53.8% / ท้าย 63.2% | Liu et al. 2023 | chunk ที่เกี่ยวข้องที่สุดไว้ต้น คำถามไว้ท้าย |
| LLM | qwen2.5-coder 7b vs 14b (MMLU 68.0 vs 75.2) | 14b ช้ากว่า ~2 เท่า เสี่ยงเกิน 90 วินาที/ข้อ | 7b |

ผล selfcheck ชุดตัวอย่างบนเครื่องพัฒนา (ไม่ใช้ Docker, qwen2.5-coder:7b, GTX 1650 4 GB ซึ่งโมเดลล้นไปรันบน CPU 58% ได้ ~6 token/s)
- ปรนัย 4/6 ข้อ ตอบผิด 2 ข้อทั้งที่ค้นเจอหัวข้อที่มีคำตอบ และตอบ "จ" ทั้งคู่ จึงปรับ prompt ให้คัดประโยคหลักฐานก่อนเลือก และไม่ปิดท้าย prompt ด้วยรายการตัวเลือก (ลดการเอนไปตัวเลือกท้าย) ยังไม่ได้วัดซ้ำหลังปรับ
- โค้ด: รอบแรก 3/3 ข้อ (unit test 25/25) รอบหลัง 2/3 เพราะข้อสุดท้าย LLM ช้าจนหมดเวลา จึงจำกัดแผนก่อนเขียนโค้ดให้สั้นลง

## รันบนเครื่อง
ต้องมี Docker Desktop, Ollama, Python 3.10+
```
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
docker compose up --build          # http://localhost:8000/health
python tools/selfcheck.py                          # ชุดตัวอย่าง 6 ข้อ + โจทย์โค้ด 3 ข้อ
```
รันแบบไม่ใช้ Docker (สำหรับพัฒนา, Windows PowerShell) ต้องตั้ง `PYTHONUTF8=1` ก่อน ไม่งั้น pip อ่านคอมเมนต์ภาษาไทยใน requirements.txt ไม่ได้
```powershell
$env:PYTHONUTF8 = "1"
$env:OLLAMA_HOST = "http://localhost:11434"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uvicorn core.server:app --port 8000
```
ถ้าขึ้น `[Errno 10048]` แปลว่ามี server ค้างอยู่ที่พอร์ต 8000 ปิดด้วย
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## ก่อนส่ง
1. แก้ `manifest.json` ให้ `student_id` และ `name` เป็นของตัวเอง
2. `python tools/doctor.py` ถ้าผ่านทั้งหมดจะได้ `dist/submission_<รหัส>.zip`
3. ส่งที่ https://stsi69.zephyr-dssi.dev (zip หรือ GitHub URL)

## ข้อจำกัด
- ชุดทดสอบภาษาไทย 62 ข้อแต่งขึ้นเอง ความต่างน้อยกว่า ~5 จุดยังสรุปไม่ได้
- ความเร็วของเครื่องตรวจไม่รู้ล่วงหน้า `/ask` ต้องจบใน 90 วินาที จึงเรียก LLM ครั้งเดียวต่อข้อ
