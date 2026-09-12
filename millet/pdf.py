"""PDF transcript generation using ReportLab.

Produces a clean, professional PDF with:
  - Page 1+: AI meeting summary (if provided)
  - Remaining pages: Full diarized transcript

Layout modeled after a professional conversation transcript document:
  - Letter-size pages (8.5 x 11 in)
  - Header with title, metadata (date, duration, participants)
  - Speaker labels in bold, timestamps in grey
  - Flowing paragraph text grouped by speaker turns
  - Footer with page numbers
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
)

if TYPE_CHECKING:
    from millet.summarize import MeetingSummary
    from millet.transcribe import Transcript


# ─── Font registration ──────────────────────────────────────────────────────

# DejaVu Sans covers Latin, Cyrillic, Greek, Turkish, and most European scripts.
_DEJAVU_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Noto Naskh Arabic for Farsi/Persian RTL text.
_NOTO_ARABIC_REGULAR = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
_NOTO_ARABIC_BOLD = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"

_fonts_registered = False

from millet.languages import PDF_SECTIONS as _PDF_SECTIONS
from millet.languages import RTL_LANGUAGES as _RTL_LANGUAGES


def _register_fonts():
    """Register Unicode TrueType fonts (called once)."""
    global _fonts_registered
    if _fonts_registered:
        return
    if Path(_DEJAVU_REGULAR).exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", _DEJAVU_REGULAR))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", _DEJAVU_BOLD))
        pdfmetrics.registerFontFamily(
            "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
        )
    if Path(_NOTO_ARABIC_REGULAR).exists():
        pdfmetrics.registerFont(TTFont("NotoNaskhArabic", _NOTO_ARABIC_REGULAR))
        pdfmetrics.registerFont(TTFont("NotoNaskhArabic-Bold", _NOTO_ARABIC_BOLD))
        pdfmetrics.registerFontFamily(
            "NotoNaskhArabic", normal="NotoNaskhArabic", bold="NotoNaskhArabic-Bold",
        )
    _fonts_registered = True


def _get_font_names(language: str) -> tuple[str, str]:
    """Return (regular, bold) font names appropriate for the language.

    DejaVu Sans is used for ALL languages (including RTL) as it has full
    Arabic/Persian glyph coverage and consistent rendering.
    """
    if Path(_DEJAVU_REGULAR).exists():
        return ("DejaVuSans", "DejaVuSans-Bold")
    # Fallback to built-in Helvetica (Latin-1 only).
    return ("Helvetica", "Helvetica-Bold")


def _is_rtl(language: str) -> bool:
    """Check if a language uses right-to-left script."""
    return language in _RTL_LANGUAGES


def _reshape_rtl(text: str) -> str:
    """Reshape and reorder RTL text for PDF rendering.

    ReportLab does not handle RTL natively, so we use arabic-reshaper
    to join glyphs and python-bidi to reorder for visual display.
    Returns the original text unchanged if the libraries are not installed.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        return text


# ─── Constants ──────────────────────────────────────────────────────────────

_PAGE_W, _PAGE_H = letter  # 612 x 792 pt
_MARGIN_LEFT = 0.75 * inch
_MARGIN_RIGHT = 0.75 * inch
_MARGIN_TOP = 0.75 * inch
_MARGIN_BOTTOM = 0.75 * inch

_COLOR_PRIMARY = HexColor("#1a1a2e")    # Dark navy for headings
_COLOR_SECONDARY = HexColor("#16213e")  # Slightly lighter
_COLOR_SPEAKER = HexColor("#0f3460")    # Speaker names
_COLOR_TIMESTAMP = HexColor("#888888")  # Grey timestamps
_COLOR_TEXT = HexColor("#2c2c2c")       # Body text
_COLOR_ACCENT = HexColor("#e94560")     # Accent / highlights
_COLOR_LIGHT_BG = HexColor("#f5f5f5")   # Light background for summary box


# ─── Styles ─────────────────────────────────────────────────────────────────

def _build_styles(language: str = "en"):
    """Build the paragraph styles used in the PDF."""
    _register_fonts()
    font_regular, font_bold = _get_font_names(language)
    rtl = _is_rtl(language)
    text_align = TA_RIGHT if rtl else TA_JUSTIFY

    styles = getSampleStyleSheet()

    s = {}

    s["title"] = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=20,
        leading=26,
        textColor=_COLOR_PRIMARY,
        alignment=TA_RIGHT if rtl else TA_LEFT,
        spaceAfter=4,
    )

    s["subtitle"] = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=10,
        leading=14,
        textColor=_COLOR_TIMESTAMP,
        alignment=TA_RIGHT if rtl else TA_LEFT,
        spaceAfter=2,
    )

    s["section_heading"] = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=_COLOR_PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=0,
        borderPadding=0,
        alignment=TA_RIGHT if rtl else TA_LEFT,
    )

    s["summary_heading"] = ParagraphStyle(
        "SummaryHeading",
        parent=styles["Heading3"],
        fontName=font_bold,
        fontSize=12,
        leading=16,
        textColor=_COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        alignment=TA_RIGHT if rtl else TA_LEFT,
    )

    s["summary_body"] = ParagraphStyle(
        "SummaryBody",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=10,
        leading=14,
        textColor=_COLOR_TEXT,
        alignment=text_align,
        spaceAfter=4,
    )

    if rtl:
        s["summary_bullet"] = ParagraphStyle(
            "SummaryBullet",
            parent=styles["Normal"],
            fontName=font_regular,
            fontSize=10,
            leading=14,
            textColor=_COLOR_TEXT,
            rightIndent=18,
            alignment=TA_RIGHT,
            spaceAfter=3,
        )
    else:
        s["summary_bullet"] = ParagraphStyle(
            "SummaryBullet",
            parent=styles["Normal"],
            fontName=font_regular,
            fontSize=10,
            leading=14,
            textColor=_COLOR_TEXT,
            leftIndent=18,
            firstLineIndent=-12,
            spaceAfter=3,
        )

    s["speaker"] = ParagraphStyle(
        "Speaker",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_SPEAKER,
        fontName=font_bold,
        spaceBefore=10,
        spaceAfter=2,
        alignment=TA_RIGHT if rtl else TA_LEFT,
    )

    s["timestamp"] = ParagraphStyle(
        "Timestamp",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8,
        leading=10,
        textColor=_COLOR_TIMESTAMP,
        spaceAfter=1,
        alignment=TA_RIGHT if rtl else TA_LEFT,
    )

    s["transcript_text"] = ParagraphStyle(
        "TranscriptText",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=10,
        leading=14,
        textColor=_COLOR_TEXT,
        alignment=text_align,
        spaceAfter=2,
    )

    s["footer"] = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=8,
        leading=10,
        textColor=_COLOR_TIMESTAMP,
        alignment=TA_CENTER,
    )

    return s


# ─── Helpers ────────────────────────────────────────────────────────────────

from millet.utils import fmt_time_short as _fmt_time


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    if s > 0 or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def _escape_xml(text: str) -> str:
    """Escape text for ReportLab's XML-based Paragraph markup."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _extract_date_from_filename(audio_file: str) -> str | None:
    """Try to extract a date from the audio filename (meeting-YYYYMMDD-HHMMSS)."""
    match = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", audio_file)
    if match:
        y, mo, d, h, mi, s = match.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
            return dt.strftime("%B %d, %Y at %H:%M")
        except ValueError:
            pass
    return None


def _group_speaker_turns(transcript: Transcript) -> list[dict]:
    """Group consecutive segments from the same speaker into turns.

    Returns a list of dicts:
        {"speaker": str, "start": float, "end": float, "text": str}
    """
    turns: list[dict] = []
    for seg in transcript.segments:
        speaker = seg.speaker or "UNKNOWN"
        text = seg.text.strip()
        if not text:
            continue

        # Merge with previous turn if same speaker
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.end
        else:
            turns.append({
                "speaker": speaker,
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })

    return turns


# ─── Summary Markdown → Flowables ──────────────────────────────────────────

def _md_to_markup(text: str, rtl_wrap=None) -> str:
    """Convert inline Markdown (bold, italic) to ReportLab XML markup.

    Handles **bold**, *italic*, and mixed patterns.  XML-escapes all
    plain-text segments so the result is safe for Paragraph().
    """
    if rtl_wrap is None:
        rtl_wrap = lambda t: t  # noqa: E731

    # RTL reshaping must run on the RAW text BEFORE XML-escaping: get_display
    # bidi-reorders characters, which would scramble multi-char entities like
    # ``&amp;`` if escaping happened first.  So escape the reshaped output.
    def _prep(raw: str) -> str:
        return _escape_xml(rtl_wrap(raw))

    # Split on bold markers first (**...**), then italic (*...*)
    parts = re.split(r"(\*\*.*?\*\*)", text)
    built = ""
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            inner = part[2:-2]
            built += f"<b>{_prep(inner)}</b>"
        else:
            # Handle italic (*...*) within non-bold segments
            sub_parts = re.split(r"(\*[^*]+?\*)", part)
            for sp in sub_parts:
                if sp.startswith("*") and sp.endswith("*") and len(sp) > 2:
                    inner = sp[1:-1]
                    built += f"<i>{_prep(inner)}</i>"
                else:
                    built += _prep(sp)
    return built


def _strip_json_block(md: str) -> str:
    """Remove trailing fenced JSON block from summary markdown.

    The summary prompt asks models to append a ```json block after the
    5 Markdown sections.  This block is consumed by frontmatter tooling
    but should NOT appear in the PDF.
    """
    return re.sub(r"\n*```json\s*\n[\s\S]*?```\s*$", "", md, flags=re.DOTALL).rstrip()


def _summary_to_flowables(
    summary_md: str, styles: dict, *, language: str = "en",
) -> list:
    """Convert the Markdown summary into ReportLab flowables."""
    # Strip the structured JSON block before rendering
    summary_md = _strip_json_block(summary_md)
    flowables: list = []
    lines = summary_md.split("\n")
    rtl = _is_rtl(language)

    def _rtl_wrap(text: str) -> str:
        """Apply RTL reshaping to RAW text if needed (callers escape after)."""
        return _reshape_rtl(text) if rtl else text

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ### Heading or ## Heading (strip leading #s)
        heading_match = re.match(r"^(#{2,4})\s+(.+)$", stripped)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            # Remove wrapping bold from headings like ### **Title**
            if heading_text.startswith("**") and heading_text.endswith("**"):
                heading_text = heading_text[2:-2]
            flowables.append(
                Paragraph(
                    _escape_xml(_rtl_wrap(heading_text)),
                    styles["summary_heading"],
                )
            )
            continue

        # Checkbox items: - [ ] or - [x]
        if stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            bullet_text = _md_to_markup(stripped[6:].strip(), _rtl_wrap)
            flowables.append(
                Paragraph(
                    f"\u2610 {bullet_text}", styles["summary_bullet"],
                )
            )
            continue

        # Bullet: - text, * text (with optional leading whitespace for sub-bullets)
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet_match:
            raw = bullet_match.group(1).strip()
            built = _md_to_markup(raw, _rtl_wrap)
            # Indent sub-bullets (original line had leading whitespace)
            if line.startswith("    ") or line.startswith("\t"):
                flowables.append(
                    Paragraph(f"\u2013 {built}", styles["summary_bullet"])
                )
            else:
                flowables.append(
                    Paragraph(f"\u2022 {built}", styles["summary_bullet"])
                )
            continue

        # Numbered list: 1. text, 2. text, etc.
        numbered_match = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if numbered_match:
            num = numbered_match.group(1)
            raw = numbered_match.group(2).strip()
            built = _md_to_markup(raw, _rtl_wrap)
            flowables.append(
                Paragraph(f"{num}. {built}", styles["summary_bullet"])
            )
            continue

        # Regular paragraph text — still convert inline bold/italic
        flowables.append(
            Paragraph(
                _md_to_markup(stripped, _rtl_wrap), styles["summary_body"],
            )
        )

    return flowables


# ─── Page template with header/footer ──────────────────────────────────────

class _PDFDocTemplate(BaseDocTemplate):
    """Custom doc template with header line and page-number footer."""

    def __init__(self, filename, title: str = "", *,
                 confidential: bool = False, attested: bool = False, **kwargs):
        super().__init__(filename, **kwargs)
        self._pdf_title = title
        self._confidential = confidential
        self._attested = attested

        frame = Frame(
            _MARGIN_LEFT,
            _MARGIN_BOTTOM + 0.3 * inch,  # room for footer
            _PAGE_W - _MARGIN_LEFT - _MARGIN_RIGHT,
            _PAGE_H - _MARGIN_TOP - _MARGIN_BOTTOM - 0.3 * inch,
            id="main",
        )
        self.addPageTemplates([
            PageTemplate(id="main", frames=[frame], onPage=self._draw_page),
        ])

    def _draw_page(self, canvas, doc):
        """Draw header line and footer on every page."""
        canvas.saveState()

        # Footer: page number (+ CONFIDENTIAL if applicable)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_COLOR_TIMESTAMP)
        page_text = f"Page {doc.page}"
        canvas.drawCentredString(_PAGE_W / 2, _MARGIN_BOTTOM * 0.5, page_text)

        if self._confidential:
            canvas.setFont("Helvetica-Bold", 7)
            canvas.setFillColor(HexColor("#cc0000"))
            canvas.drawCentredString(
                _PAGE_W / 2, _MARGIN_BOTTOM * 0.5 + 12, "CONFIDENTIAL"
            )
        elif self._attested:
            # Since 0.19.0 every summary is produced in a TEE, so stamping
            # each one CONFIDENTIAL in red would make the warning wallpaper.
            # State the provenance quietly instead, and keep the red banner
            # for sessions explicitly flagged sensitive by the caller.
            canvas.setFont("Helvetica", 6.5)
            canvas.setFillColor(_COLOR_TIMESTAMP)
            canvas.drawCentredString(
                _PAGE_W / 2, _MARGIN_BOTTOM * 0.5 + 11,
                "Summarized in a hardware-attested TEE",
            )

        # Header: CONFIDENTIAL banner (if applicable)
        if self._confidential:
            canvas.setFont("Helvetica-Bold", 7)
            canvas.setFillColor(HexColor("#cc0000"))
            canvas.drawCentredString(
                _PAGE_W / 2, _PAGE_H - _MARGIN_TOP + 14, "CONFIDENTIAL"
            )

        # Thin line at top of content area
        y_line = _PAGE_H - _MARGIN_TOP + 4
        canvas.setStrokeColor(HexColor("#dddddd"))
        canvas.setLineWidth(0.5)
        canvas.line(_MARGIN_LEFT, y_line, _PAGE_W - _MARGIN_RIGHT, y_line)

        canvas.restoreState()


# ─── Public API ─────────────────────────────────────────────────────────────

def generate_pdf(
    transcript: Transcript,
    output_path: str | Path,
    summary: MeetingSummary | None = None,
    title: str = "Meeting Transcript",
    language: str = "en",
    confidential: bool = False,
) -> Path:
    """Generate a PDF transcript document.

    Args:
        transcript: The Transcript object with segments and speaker info.
        output_path: Where to write the PDF file.
        summary: Optional AI-generated meeting summary to include as
            the first section.
        title: Document title shown on the first page.
        language: Language code (e.g. "en", "de", "fa") for font and
            RTL selection.
        confidential: If True, print "CONFIDENTIAL" in red on every page
            header and footer.  Caller opt-in only since 0.19.0 — a
            TEE-backed summary alone no longer triggers it (every summary
            is TEE-backed now), it gets a quiet attestation footer instead.

    Returns:
        Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rtl = _is_rtl(language)
    styles = _build_styles(language)
    sections = _PDF_SECTIONS.get(language, _PDF_SECTIONS["en"])
    story: list = []

    # ── Title block ──
    story.append(Paragraph(title, styles["title"]))

    # Metadata line(s)
    meta_parts: list[str] = []

    date_str = _extract_date_from_filename(transcript.audio_file)
    if date_str:
        meta_parts.append(f"Date: {date_str}")

    if transcript.duration and transcript.duration > 0:
        meta_parts.append(f"Duration: {_fmt_duration(transcript.duration)}")

    if transcript.speakers:
        speaker_names = ", ".join(s.label or s.id for s in transcript.speakers)
        meta_parts.append(f"Participants: {speaker_names}")

    meta_parts.append("Recording source: AI transcription (millet)")

    for part in meta_parts:
        story.append(Paragraph(_escape_xml(part), styles["subtitle"]))

    story.append(Spacer(1, 12))

    # ── Summary section (if provided) ──
    if summary:
        summary_title = sections["summary"]
        if rtl:
            summary_title = _reshape_rtl(summary_title)
        story.append(Paragraph(
            _escape_xml(summary_title), styles["section_heading"],
        ))
        story.append(
            Paragraph(
                f"<i>Generated by {_escape_xml(summary.model)} "
                f"in {summary.elapsed_seconds:.0f}s</i>",
                styles["subtitle"],
            )
        )
        story.append(Spacer(1, 4))

        summary_flowables = _summary_to_flowables(
            summary.markdown, styles, language=language,
        )
        story.extend(summary_flowables)

        story.append(Spacer(1, 16))

    # ── Transcript section ──
    transcript_title = sections["transcript"]
    if rtl:
        transcript_title = _reshape_rtl(transcript_title)
    story.append(Paragraph(
        _escape_xml(transcript_title), styles["section_heading"],
    ))
    story.append(Spacer(1, 4))

    turns = _group_speaker_turns(transcript)

    for turn in turns:
        speaker = turn["speaker"]
        start_ts = _fmt_time(turn["start"])

        # Speaker + timestamp header
        header = (
            f'<font color="{_COLOR_SPEAKER}">'
            f'<b>{_escape_xml(speaker)}</b></font>'
            f'  <font color="{_COLOR_TIMESTAMP}" size="8">{start_ts}</font>'
        )
        story.append(Paragraph(header, styles["speaker"]))

        # Transcript text — reshape RAW text before escaping so bidi
        # reordering can't scramble multi-char XML entities (&amp; etc.).
        raw_text = turn["text"]
        if rtl:
            raw_text = _reshape_rtl(raw_text)
        story.append(Paragraph(_escape_xml(raw_text), styles["transcript_text"]))

    # ── Build PDF ──
    # A TEE-produced summary is *attested*, which is now the norm rather than
    # an exception, so it earns a quiet footer.  The red CONFIDENTIAL banner
    # is reserved for callers that explicitly ask for it.
    is_confidential = confidential
    is_attested = bool(
        summary and getattr(summary, "backend", "") in ("tinfoil", "tinfoil-tee")
    )

    doc = _PDFDocTemplate(
        str(output_path),
        title=title,
        confidential=is_confidential,
        attested=is_attested,
        pagesize=letter,
        leftMargin=_MARGIN_LEFT,
        rightMargin=_MARGIN_RIGHT,
        topMargin=_MARGIN_TOP,
        bottomMargin=_MARGIN_BOTTOM,
    )

    doc.build(story)
    return output_path
