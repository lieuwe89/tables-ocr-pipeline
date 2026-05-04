from dataclasses import dataclass, field

@dataclass
class OcrWord:
    """A single word with its bounding box."""
    id: str          # Unique word ID (e.g., "w_0001")
    text: str        # Recognized text
    bbox: list[int]  # [x1, y1, x2, y2] in pixels
    confidence: int  # 0–100
    line_id: str     # Parent line ID


@dataclass
class OcrLine:
    """A line of text."""
    id: str
    bbox: list[int]
    words: list[OcrWord] = field(default_factory=list)


@dataclass
class OcrBlock:
    """A text block (region)."""
    id: str
    bbox: list[int]
    lines: list[OcrLine] = field(default_factory=list)


@dataclass
class OcrPage:
    """Full OCR result for a single page."""
    scan_file: str
    width: int
    height: int
    blocks: list[OcrBlock] = field(default_factory=list)
    hocr_raw: str = ""

    @property
    def all_words(self) -> list[OcrWord]:
        words = []
        for block in self.blocks:
            for line in block.lines:
                words.extend(line.words)
        return words

    @property
    def word_index(self) -> dict[str, OcrWord]:
        return {w.id: w for w in self.all_words}

    @property
    def line_index(self) -> dict[str, OcrLine]:
        """Index of all lines by their ID."""
        lines = {}
        for block in self.blocks:
            for line in block.lines:
                lines[line.id] = line
        return lines

    def to_numbered_word_list(self) -> str:
        """Format words as a numbered list for the LLM prompt."""
        return "\n".join(f"{w.id}: {w.text}" for w in self.all_words)
