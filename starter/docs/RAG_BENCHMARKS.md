# ตัวชี้วัดวิธีทำ RAG: อะไรเร็ว อะไรแม่น (ตัวเลขจากงานวิจัย)

เป้าหมาย: เลือกเทคนิคที่ทำให้ RAG **แม่นที่สุดภายใต้งบเวลา 90 วินาที/ข้อ** โดยใช้ได้แค่ `qwen2.5-coder:7b` (หรือ 14b) กับ `nomic-embed-text` ผ่าน Ollama เรียก LLM ครั้งละข้อ ไม่มีอินเทอร์เน็ตตอนตรวจ
ทุกตัวเลขมาจากแหล่งในตาราง ตัวเลขของเราเองระบุว่า "วัดเอง"

วิธีอ่าน: ความแม่นที่เพิ่ม (Δ) เทียบกับต้นทุนเวลา เทคนิคที่ต้องเรียก LLM เพิ่มต่อข้อ มีต้นทุน ~1 เท่าของเวลาตอบทั้งข้อ ซึ่งบนเครื่องพัฒนาของเรา (6 token/วินาที) ใช้ไม่ได้เลย

## 1. Retrieval: ค้นอย่างไรให้เจอ

| วิธี | ตัวเลข | แหล่ง | ต้นทุนเวลา | ใช้ในงานนี้ |
|---|---|---|---|---|
| BM25 อย่างเดียว | BEIR 18 ชุดข้อมูล zero-shot: DPR แย่กว่า BM25 47.7%, ANCE −7.4%, TAS-B −2.8%; DPR แพ้ BM25 ใน 17/18 ชุด | Thakur et al. 2021 (BEIR) | ~0 | ✅ ช่องทางหลัก |
| BM25 กับภาษาไทย | MIRACL Thai nDCG@10: BM25 0.484 > mDPR 0.358; hybrid 0.599 | Zhang et al. 2023 (MIRACL) | ~0 | ✅ |
| Hybrid (BM25 + dense) | MS MARCO DL19 nDCG@10: dense 70.20 → hybrid 72.50, R@1k 84.48 → 89.08; latency 2.61 → 3.20 วินาที | Wang et al. 2024 (Best Practices in RAG) | ต่ำ | ✅ แต่ถ่วง dense ตามสัดส่วนอังกฤษ |
| Convex combination vs RRF | convex ดีกว่า RRF ทั้งใน/นอกโดเมน และ RRF ไวต่อพารามิเตอร์ | Bruch et al. 2023 | ~0 | ✅ |
| (วัดเอง) วิธีรวมคะแนน | hit@1: RRF 0.694, convex 0.5/0.5 0.855, BM25 ถ่วงตามภาษา **0.935** | ชุดทดสอบไทย 62 ข้อ | ~0 | ✅ |
| (วัดเอง) ตัดคำไทยก่อน BM25 | hit@5: ตัดตามช่องว่าง 0.548 → newmm 0.984 | ชุดทดสอบไทย 62 ข้อ | ~0 | ✅ สำคัญที่สุด |
| HyDE (ให้ LLM เขียนคำตอบสมมติแล้วค้น) | DL19 nDCG@10: Contriever 44.5 → 61.3; latency 2.61 → 7.21 วินาที (+HyDE บน LLM-Embedder) | Gao et al. 2022; Wang et al. 2024 | **เรียก LLM เพิ่ม 1 ครั้ง** | ❌ เกินงบเวลา และ nomic อ่านไทยไม่ได้ |
| Contextual retrieval (เติมบริบทเอกสารให้ทุก chunk ด้วย LLM) | top-20 ค้นพลาด 5.7% → 3.7% (−35%) → +contextual BM25 2.9% (−49%) → +rerank 1.9% (−67%) | Anthropic 2024 | เรียก LLM 1 ครั้งต่อ chunk ตอนทำดัชนี | ⚠️ แนวคิดใช้ได้ฟรี: ใส่เส้นทางหัวข้อนำหน้า chunk (วัดเอง hit@1 0.919 → 0.935) |
| Query rewriting / decomposition | ไม่ช่วยเมื่อเทียบ baseline | Wang et al. 2024 | เรียก LLM เพิ่ม | ❌ |

## 2. Re-ranking: เรียงใหม่ให้ชิ้นที่ถูกขึ้นก่อน

| วิธี | MRR@10 (MS MARCO dev) | latency ต่อคำค้น | แหล่ง |
|---|---|---|---|
| BM25 ไม่ re-rank | 11.65 | – | Wang et al. 2024 |
| TILDEv2 | 27.83 | 0.02 วินาที | 〃 |
| monoT5 | 31.78 | 4.5 วินาที | 〃 |
| monoBERT | 31.69 | 15.8 วินาที | 〃 |
| RankLLaMA (LLM 7B) | 32.35 | 82.4 วินาที | 〃 |
| BM25 + cross-encoder (BEIR) | +11% เทียบ BM25; 450 ms (GPU) / 6,100 ms (CPU) ต่อคำค้น | | Thakur et al. 2021 |

ข้อสรุปสำหรับงานนี้: re-ranker แบบ cross-encoder ใช้ไม่ได้ (เรียกได้เฉพาะโมเดลใน manifest) ส่วนการให้ LLM 7b re-rank ใช้เวลาระดับ RankLLaMA (~80 วินาที) จึงเกินงบ
ทางที่เหลือ: ทำให้การค้นชั้นแรกแม่นพอ (hit@3 = 1.000 บนชุดของเรา) แล้วไม่ต้อง re-rank

## 3. Chunking

| วิธี | ตัวเลข | แหล่ง |
|---|---|---|
| ขนาด chunk (token) | faithfulness: 2048 = 80.4, 1024 = 94.3, **512 = 97.6**, 256 = 97.2, 128 = 95.7 | Wang et al. 2024 |
| sliding window vs ตัดตรง | faithfulness 95.74 → 97.41 | 〃 |
| ขนาด/overlap (text-embedding-3-large, 5 ชิ้น) | recall: 800 tok overlap 400 = 85.4–87.9; **400 tok ไม่ overlap = 89.2–89.5**; LLM chunker 91.9 | Chroma 2024 |
| overlap | ตัด overlap ทำให้ IoU ดีขึ้น (token ซ้ำน้อยลง) | Chroma 2024 |
| ตามโครงสร้างเอกสาร vs ตามความยาว (ภาษาไทย) | MRR 0.786 → 0.834, F1 0.722 → 0.739 | NitiBench (Akarajaradwong et al. 2025) |
| (วัดเอง) ขนาด chunk ตัวอักษรไทย | hit@1: 300 = 0.887, 500 = 0.919, 700–1,800 = 0.935 | ชุดทดสอบไทย 62 ข้อ |

ภาษาไทย ~1.5 ตัวอักษร/token ของ Qwen ดังนั้น 1,000 ตัวอักษรไทย ≈ 650 token ใกล้ช่วงที่ดีที่สุดของงานวิจัย (400–512 token)

## 4. Generation: ทำให้ LLM ตอบถูก

| เทคนิค | ตัวเลข | แหล่ง | ต้นทุนเวลา | ใช้ |
|---|---|---|---|---|
| ตำแหน่งเอกสารใน context | GPT-3.5 20 เอกสาร: อยู่ต้น 75.8%, กลาง 53.8%, ท้าย 63.2%, ไม่มีเอกสาร 56.1% | Liu et al. 2024 (Lost in the Middle) | 0 | ✅ ชิ้นที่ดีที่สุดไว้ต้น |
| ลำดับ repack | Reverse 0.483 / Forward 0.474 / Sides 0.478 (average score) | Wang et al. 2024 | 0 | ต่างกันน้อย |
| บังคับ JSON โดยไม่ให้คิด | GSM8K: LLaMA-3-8B 74.7 → 48.9; GPT-3.5 76.6 → 49.3 | Tam et al. 2024 | 0 | ✅ ให้ `reason` ก่อน `answer` |
| ย่อ context (LongLLMLingua) | NQ +21.4% ด้วย token น้อยลง ~4 เท่า; latency เร็วขึ้น 1.4–2.6 เท่า | Jiang et al. 2024 | ต้องมีโมเดลย่อเพิ่ม | ❌ ใช้ได้แค่โมเดลใน manifest |
| ย่อด้วย Recomp | average score 0.480 → 0.483, latency 10.97 → 11.70 วินาที | Wang et al. 2024 | ต้องมีโมเดลเพิ่ม | ❌ |
| ความลำเอียงต่อตำแหน่งตัวเลือก | GPT-3.5 MMLU แม่นยำ 60.9–74.2 ขึ้นกับตำแหน่งคำตอบ; PriDe แก้ได้ 67.2 → 68.0 ด้วยต้นทุน ×1.15 (สลับตัวเลือกครบ ×4) | Zheng et al. 2024 | ×1.15–×4 | ❌ เกินงบ ใช้ prompt ลดอคติแทน |
| Self-consistency (ถามหลายรอบแล้วโหวต) | เพิ่มความแม่นได้มากในโจทย์เหตุผล | Wang et al. 2023 | ×N | ❌ เกินงบ |

## 5. เขียนโค้ด

| เทคนิค | ตัวเลข | แหล่ง | ต้นทุน | ใช้ |
|---|---|---|---|---|
| 7b vs 14b | HumanEval 88.4 vs 89.6, MBPP 83.5 vs 86.2, LiveCodeBench 18.2 vs 23.4 (Instruct) | Hui et al. 2024 (Qwen2.5-Coder) | 14b ใช้หน่วยความจำ ~9 GB ช้ากว่า ~2 เท่า | 7b |
| วางแผนก่อนเขียน (self-planning) | Pass@1 สูงขึ้นสูงสุด 25.4% (relative) เทียบเขียนตรง, 11.9% เทียบ CoT | Jiang et al. 2023 | +token แผน | ✅ แผนสั้น ≤4 บรรทัด |
| แก้จาก error (self-debug) | MBPP/TransCoder สูงขึ้นสูงสุด 12% | Chen et al. 2024 | +1 รอบ | ✅ core ทำให้ 1 รอบเมื่อ compile ไม่ผ่าน |
| Flow engineering (AlphaCodium) | GPT-4 CodeContests pass@5 19% → 44% | Ridnik et al. 2024 | หลายรอบ | ❌ เกินงบ 180 วินาที |
| ภาษาของโจทย์ | โจทย์ที่ไม่ใช่อังกฤษได้ pass@1 ต่ำกว่า (23 ภาษา ไม่มีไทย ใช้เป็นแนวโน้มเท่านั้น) | Peng et al. 2024 (HumanEval-XL) | 0 | ✅ คำสั่งเป็นอังกฤษ |

## 6. ความเร็ว: งบ 90 วินาทีหมดไปกับอะไร

เวลาตอบ 1 ข้อ ≈ (token ของ prompt ÷ ความเร็วอ่าน) + (token ของคำตอบ ÷ ความเร็วเขียน)
วัดเองบนเครื่องพัฒนา (GTX 1650 4 GB, 7b รันบน CPU 58%): อ่าน 49.7 token/วินาที เขียน 6.1 token/วินาที

| ตั้งค่า | prompt | คำตอบ | เวลาโดยประมาณ (เครื่องนี้) |
|---|---|---|---|
| TOP_K 6, reason ไทย | ~4,400 | ~100 | ~105 วินาที ⚠️ |
| TOP_K 4, reason ไทย | ~3,100 | ~100 | ~78 วินาที |
| TOP_K 4, reason อังกฤษ | ~3,100 | ~60 | ~72 วินาที |

ตัวที่กินเวลามากที่สุดคือ**การอ่านเอกสาร** ไม่ใช่การเขียนคำตอบ วิธีลดเวลาที่คุ้มที่สุดจึงเป็นการส่ง chunk น้อยลงแต่แม่นขึ้น
Ollama เก็บ prefix cache ไว้ใช้ซ้ำ ถ้า system prompt เหมือนเดิมทุกข้อและอยู่ต้น prompt จะไม่ต้องอ่านใหม่ทุกข้อ (โค้ดตอนนี้เป็นแบบนี้แล้ว)

## 7. สูตรที่เร็วและแม่นที่สุดสำหรับงานนี้

| ขั้น | ทางเลือกที่ดีที่สุดภายใต้กติกา | เหตุผล (ตัวเลข) |
|---|---|---|
| Chunk | ตามหัวข้อ markdown ~1,000 ตัวอักษรไทย (~650 token) ใส่เส้นทางหัวข้อ ไม่ตัดกลางตาราง | NitiBench MRR +0.048; วัดเอง hit@1 0.935 |
| ค้น | BM25 ตัดคำ newmm+bigram เป็นหลัก ถ่วง vector ตามสัดส่วนอักษรอังกฤษ คำค้น = คำถาม + ตัวเลือก | BEIR, MIRACL Thai; วัดเอง hit@5 1.000 |
| Re-rank | ไม่ทำ | ทุกแบบที่แม่นขึ้นใช้ 4.5–82 วินาทีต่อคำค้น |
| Context | 4–6 ชิ้น เรียงชิ้นที่ดีที่สุดไว้ต้น คำถามไว้ท้าย | Lost in the Middle 75.8% ต้น vs 53.8% กลาง |
| ตอบ | เรียก LLM ครั้งเดียว JSON `reason` ก่อน `answer` | Tam et al. −25.8 จุดเมื่อไม่ให้คิด |
| โค้ด | วางแผนสั้นแล้วเขียน code block เดียว ให้ core แก้ 1 รอบ | Self-planning +25.4%, Self-debug +12% |

## แหล่งอ้างอิง
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models* (NeurIPS 2021) — https://arxiv.org/abs/2104.08663
- Zhang et al., *MIRACL* (TACL 2023) — https://arxiv.org/abs/2210.09984
- Bruch, Gai, Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval* (TOIS 2023) — https://arxiv.org/abs/2210.11934
- Wang et al., *Searching for Best Practices in Retrieval-Augmented Generation* (EMNLP 2024) — https://arxiv.org/abs/2407.01219
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)* (2022) — https://arxiv.org/abs/2212.10496
- Anthropic, *Introducing Contextual Retrieval* (2024) — https://www.anthropic.com/news/contextual-retrieval
- Chroma, *Evaluating Chunking Strategies for Retrieval* (2024) — https://www.trychroma.com/research/evaluating-chunking
- Akarajaradwong et al., *NitiBench* (2025) — https://arxiv.org/abs/2502.10868
- Liu et al., *Lost in the Middle* (TACL 2024) — https://arxiv.org/abs/2307.03172
- Tam et al., *Let Me Speak Freely?* (2024) — https://arxiv.org/abs/2408.02442
- Jiang et al., *LongLLMLingua* (ACL 2024) — https://arxiv.org/abs/2310.06839
- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors* (ICLR 2024) — https://arxiv.org/abs/2309.03882
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning* (ICLR 2023) — https://arxiv.org/abs/2203.11171
- Hui et al., *Qwen2.5-Coder Technical Report* (2024) — https://arxiv.org/abs/2409.12186
- Jiang et al., *Self-planning Code Generation with LLMs* (TOSEM 2024) — https://arxiv.org/abs/2303.06689
- Chen et al., *Teaching Large Language Models to Self-Debug* (ICLR 2024) — https://arxiv.org/abs/2304.05128
- Ridnik et al., *Code Generation with AlphaCodium* (2024) — https://arxiv.org/abs/2401.08500
- Peng et al., *HumanEval-XL* (LREC-COLING 2024) — https://arxiv.org/abs/2402.16694
