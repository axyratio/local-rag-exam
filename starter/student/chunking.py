"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

วิธีที่ใช้: structure-aware chunking
  1. แยกเอกสารตามหัวข้อ markdown (#, ##, ###) และจำเส้นทางหัวข้อไว้ เช่น "ชื่อเอกสาร > 2. การอัปเดต"
  2. ในแต่ละหัวข้อ แบ่งเป็น "บล็อก" ที่ห้ามตัดกลาง: ย่อหน้า, ตารางทั้งตาราง, code block ทั้งก้อน, list ทั้งชุด
  3. รวมบล็อกต่อกันจนใกล้ CHUNK_SIZE แล้วซ้อนบล็อกท้ายไปยัง chunk ถัดไป (overlap)
  4. บล็อกที่ยาวเกิน CHUNK_SIZE เอง (เช่น ตารางยาว) ตัดตามบรรทัด โดยตารางจะใส่หัวตารางซ้ำทุกชิ้น
  5. ใส่เส้นทางหัวข้อไว้บรรทัดแรกของทุก chunk
"""
import re

from student import settings

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ (แต่ละชิ้นขึ้นต้นด้วยเส้นทางหัวข้อ)"""
    size = max(200, int(settings.CHUNK_SIZE))
    chunks: list[str] = []
    for path, body in _sections(text, doc_id):
        header = (" > ".join(path)[:size // 4] + "\n") if settings.INCLUDE_HEADING and path else ""  # หัวข้อยาวผิดปกติไม่กินที่ทั้ง chunk
        blocks = _blocks(body)
        if not blocks:
            continue
        for piece in _pack(blocks, size - len(header), settings.CHUNK_OVERLAP):
            chunks.append(header + piece)
    return chunks or [text.strip() or doc_id]


def _sections(text: str, doc_id: str) -> list[tuple[list[str], str]]:
    """คืน (เส้นทางหัวข้อ, เนื้อหา) ของแต่ละหัวข้อ ไม่นับ # ที่อยู่ใน code block"""
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        titles = [t for _, t in stack] or [doc_id]
        sections.append((titles, lines.copy()))
        lines.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match and settings.SPLIT_BY_HEADING:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip("#* ")))
        else:
            lines.append(line)
    flush()
    return [(path, "\n".join(body).strip()) for path, body in sections if "\n".join(body).strip()]


def _blocks(body: str) -> list[str]:
    """แบ่งเนื้อหาเป็นบล็อกที่ไม่ควรตัดกลาง: code block, ตาราง, list และย่อหน้า"""
    blocks: list[str] = []
    current: list[str] = []
    kind = None  # "fence" | "table" | "list" | "para"

    def flush() -> None:
        nonlocal kind
        if current and "\n".join(current).strip():
            blocks.append("\n".join(current).strip("\n"))
        current.clear()
        kind = None

    for line in body.splitlines():
        if kind == "fence":
            current.append(line)
            if _FENCE.match(line):
                flush()
            continue
        if _FENCE.match(line):
            flush()
            kind = "fence"
            current.append(line)
            continue
        if not line.strip():
            if kind != "list":  # list items may be separated by blank lines
                flush()
            continue
        line_kind = "table" if _TABLE_ROW.match(line) else "list" if _LIST_ITEM.match(line) else None
        if line_kind and line_kind != kind:
            # a lead-in sentence that ends with ":" stays with its table/list
            lead = current[:] if kind == "para" and current and current[-1].rstrip().endswith(":") else []
            if lead:
                current.clear()
            flush()
            current.extend(lead)
            kind = line_kind
        elif not line_kind and kind in ("table",):
            flush()
            kind = "para"
        elif not line_kind and kind == "list" and not line.startswith((" ", "\t")):
            flush()
            kind = "para"
        elif kind is None:
            kind = "para"
        current.append(line)
    flush()
    return blocks


def _pack(blocks: list[str], size: int, overlap: int) -> list[str]:
    """รวมบล็อกให้ยาวใกล้ size; ซ้อนบล็อกท้ายของชิ้นก่อน (ไม่เกิน overlap ตัวอักษร) ไปยังชิ้นถัดไป"""
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_long(block, size))
    out: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = "\n\n".join(current + [piece])
        if current and len(candidate) > size:
            out.append("\n\n".join(current))
            tail = current[-1]
            current = [tail] if overlap and len(tail) <= overlap else []
        current.append(piece)
    if current:
        out.append("\n\n".join(current))
    return out


def _split_long(block: str, size: int) -> list[str]:
    """ตัดบล็อกที่ยาวเกิน size ตามบรรทัด (ตาราง: ใส่หัวตารางซ้ำ) หรือตามประโยค/ช่องว่าง"""
    if len(block) <= size:
        return [block]
    lines = block.split("\n")
    repeat = ""
    if len(lines) > 2 and _TABLE_ROW.match(lines[0]) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[1]):
        repeat = "\n".join(lines[:2]) + "\n"
        lines = lines[2:]
    if len(lines) == 1:
        return _split_line(block, size)
    out: list[str] = []
    current = ""
    for line in lines:
        if current and len(repeat) + len(current) + len(line) + 1 > size:
            out.append(repeat + current)
            current = ""
        if len(repeat) + len(line) > size:
            out.extend(repeat + part for part in _split_line(line, size - len(repeat)))
            continue
        current = f"{current}\n{line}" if current else line
    if current:
        out.append(repeat + current)
    return out


def _split_line(line: str, size: int) -> list[str]:
    """ตัดบรรทัดยาวที่ช่องว่าง (ภาษาไทยใช้ช่องว่างคั่นประโยค) หรือที่ขอบ size ถ้าไม่มีช่องว่าง"""
    size = max(size, 50)  # กันวนไม่รู้จบเมื่อ size เหลือน้อยหรือติดลบ
    out: list[str] = []
    while len(line) > size:
        cut = line.rfind(" ", size // 2, size)
        cut = cut if cut > 0 else size
        out.append(line[:cut].strip())
        line = line[cut:].strip()
    if line:
        out.append(line)
    return out
