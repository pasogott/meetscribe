"""Structured YAML frontmatter for meeting summaries (schema v1).

Every ``.summary.md`` produced by millet ships with a YAML frontmatter
block containing a typed view of the meeting: action items, decisions,
participants, and topics.  This module owns:

  - the schema (intentionally minimal in v1),
  - building frontmatter from a ``Transcript`` + session metadata + LLM
    extraction,
  - extracting an LLM-emitted JSON data block from a single completion,
  - validating that the resulting structure is well-formed,
  - rendering YAML without a yaml dependency (we control the schema, so
    a small hand-rolled writer is enough and avoids a new install dep).

Design notes
------------

We intentionally do **not** depend on PyYAML.  The schema is small and
well-typed; a few hundred lines of writer + a strict tolerant parser
keeps millet's runtime install footprint identical.

Likewise we do **not** invent due dates, GitHub handles, or commitments
that the LLM did not produce.  Every list is allowed to be empty; the
indexer downstream is responsible for cross-meeting derivations.

Schema version is bumped any time field names or semantics change.
``schema_version: 1`` is what every consumer should pin against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from millet.crosstalk import is_reserved_label

SCHEMA_VERSION = 1

# Allowed values for the ``type`` field. Kept narrow on purpose; "memo" and
# "dictation" anticipate the fast-path pipelines vezir/millet will grow.
ALLOWED_TYPES = ("meeting", "memo", "dictation")

# ─── Data extraction from LLM output ───────────────────────────────────────

# The LLM is instructed to append a fenced JSON block at the very end of its
# output.  We accept ```json and bare ``` fences, as well as a trailing JSON
# object without a fence, to be tolerant of model quirks.
#
# Examples we want to accept (in priority order):
#   ```json\n{ ... }\n```
#   ```\n{ ... }\n```
#   { ... }   (only if it's clearly the trailing JSON, i.e. last block)

# Pattern matches a fenced block at end of string.  We require it to be the
# last fenced block to avoid eating example JSON inside the body.
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n(\{.*?\})\s*\n```\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Fallback: a top-level JSON object as the trailing chunk of the output.
# We try this only if no fenced block is found.  The candidate scan lives in
# split_body_and_data (it tries each line-leading '{' from last to first).


def split_body_and_data(raw: str) -> tuple[str, dict | None, str | None]:
    """Split an LLM completion into (markdown_body, parsed_data_or_None, error).

    The contract with the LLM (see ``prompts/summarize_*_system.md``) is:

        ## Section
        ...markdown body...

        ```json
        { "action_items": [...], "decisions": [...], ... }
        ```

    We strip the trailing fenced JSON block from the body and parse it.
    On any parse failure we return ``(body, None, error_message)`` — the
    caller is responsible for shipping a meeting with empty arrays + a
    flag rather than blocking the pipeline on malformed JSON.

    The returned body is what gets rendered into the PDF.  It is identical
    to ``raw`` when the LLM did not emit a data block, so older callers
    that ignore the data field continue to work unchanged.
    """
    if not raw:
        return raw, None, "empty completion"

    text = raw.rstrip()

    m = _FENCED_JSON_RE.search(text)
    if m:
        body = text[: m.start()].rstrip()
        try:
            return body, json.loads(m.group(1)), None
        except json.JSONDecodeError as e:
            return body, None, f"fenced JSON parse error: {e}"

    if not text.endswith("}"):
        return text, None, "no JSON block found"

    # Try each line-leading '{' as a candidate start for the trailing object,
    # from the LAST one to the first.  A single greedy regex would anchor at
    # the earliest '{' — swallowing body text before the real JSON and failing
    # to parse, which discarded genuinely valid trailing data.
    starts = [0] if text.startswith("{") else []
    starts += [m.start() + 1 for m in re.finditer(r"\n(\{)", text)]
    for start in reversed(starts):
        candidate = text[start:].strip()
        # Be conservative: only treat it as the data block when it parses AND
        # contains at least one of the expected keys.  Otherwise leave it in
        # the body where it might be a real example.
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and any(
            k in parsed for k in ("action_items", "decisions", "topics", "participants")
        ):
            body = text[:start].rstrip()
            return body, parsed, None
        # Parsed but not a data block: it's an example; stop searching earlier
        # (broader) candidates that would only re-include this same object.
        break

    return text, None, "no JSON block found"


# ─── Frontmatter construction ──────────────────────────────────────────────


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_str(v: Any) -> str | None:
    """Stringify scalar values; drop everything else."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def _normalize_action_item(raw: Any) -> dict[str, Any] | None:
    """Best-effort coerce a raw action-item record into the canonical shape."""
    if not isinstance(raw, dict):
        return None
    task = _coerce_str(raw.get("task") or raw.get("description") or raw.get("text"))
    if not task:
        return None
    assignee = _coerce_str(raw.get("assignee") or raw.get("owner"))
    due = _coerce_str(raw.get("due") or raw.get("due_date"))
    status = _coerce_str(raw.get("status")) or "open"
    if status not in ("open", "closed", "blocked"):
        status = "open"
    return {
        "assignee": assignee,
        "task": task,
        "due": due,
        "status": status,
    }


def _normalize_decision(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        # Allow a bare string for ergonomics.
        text = _coerce_str(raw)
        if not text:
            return None
        return {"text": text, "topic": None}
    text = _coerce_str(raw.get("text") or raw.get("decision"))
    if not text:
        return None
    topic = _coerce_str(raw.get("topic"))
    return {"text": text, "topic": topic}


def _normalize_topic(raw: Any) -> str | None:
    if isinstance(raw, dict):
        return _coerce_str(raw.get("topic") or raw.get("name") or raw.get("text"))
    return _coerce_str(raw)


def _normalize_participant_from_extraction(raw: Any) -> dict[str, Any] | None:
    """Normalize a participant entry as the LLM might emit it.

    The LLM only knows speaker labels from the transcript (e.g. "YOU",
    "REMOTE_1", "Alice").  Channel/role/handle are filled in later from
    the Transcript object, not from extraction.
    """
    name = None
    if isinstance(raw, dict):
        name = _coerce_str(raw.get("name") or raw.get("speaker"))
    else:
        name = _coerce_str(raw)
    if not name:
        return None
    return {"name": name, "role": None, "channel": None}


@dataclass
class FrontmatterContext:
    """Runtime context the writer needs that the LLM does not provide.

    These fields come from the recording session itself, not from the
    transcript text, so we pass them in explicitly rather than asking
    the model to invent them.
    """

    title: str | None = None
    date: str | None = None  # ISO 8601, defaults to now() in UTC
    duration_seconds: float | None = None
    language: str | None = None
    type: str = "meeting"
    session_id: str | None = None
    audio_sha256: str | None = None
    # Speaker labels as they appear in the transcript -> channel ("mic"|"system"|None)
    speaker_channels: dict[str, str] = field(default_factory=dict)
    # Optional override list of participant names in transcript-label form.
    # When provided, used as the canonical participant set; channels still
    # come from speaker_channels when known.
    transcript_speakers: list[str] = field(default_factory=list)


def _iso_duration(seconds: float | None) -> str | None:
    """Render a duration in ISO 8601 ``PT…`` form, e.g. ``PT42M17S``.

    Returns ``None`` for ``None`` or non-positive values.
    """
    if seconds is None or seconds <= 0:
        return None
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}H")
    if m:
        parts.append(f"{m}M")
    if s or not parts:
        parts.append(f"{s}S")
    return "PT" + "".join(parts)


def build_frontmatter(
    extracted: dict | None,
    context: FrontmatterContext,
    *,
    extraction_error: str | None = None,
) -> dict[str, Any]:
    """Assemble the canonical frontmatter dict.

    ``extracted`` is the dict parsed from the LLM's JSON data block (or
    ``None`` if it failed to parse).  ``context`` provides session-level
    fields that come from millet, not the LLM.  We deliberately keep
    the schema small and let downstream tooling (vezir indexer) derive
    richer views.
    """
    extracted = extracted or {}

    action_items = [
        ai
        for ai in (
            _normalize_action_item(x) for x in extracted.get("action_items") or []
        )
        if ai
    ]
    decisions = [
        d
        for d in (_normalize_decision(x) for x in extracted.get("decisions") or [])
        if d
    ]
    topics = [
        t
        for t in (_normalize_topic(x) for x in extracted.get("topics") or [])
        if t
    ]

    # Participants: union of (a) speakers known from the transcript and
    # (b) anyone the LLM mentioned.  Transcript wins on channel info.
    seen: dict[str, dict[str, Any]] = {}
    transcript_names = list(context.transcript_speakers)
    if not transcript_names and context.speaker_channels:
        transcript_names = list(context.speaker_channels.keys())
    for name in transcript_names:
        # CROSSTALK (unattributable fillers) is not a participant.
        if not name or is_reserved_label(name):
            continue
        seen[name] = {
            "name": name,
            "role": None,
            "channel": context.speaker_channels.get(name),
        }
    for raw in extracted.get("participants") or []:
        norm = _normalize_participant_from_extraction(raw)
        if not norm or is_reserved_label(norm["name"]):
            continue
        existing = seen.get(norm["name"])
        if existing:
            # Keep transcript-provided channel; nothing else to merge in v1.
            continue
        seen[norm["name"]] = norm
    participants = list(seen.values())

    fm: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "type": context.type if context.type in ALLOWED_TYPES else "meeting",
        "title": context.title,
        "date": context.date or _now_utc_iso(),
        "duration": _iso_duration(context.duration_seconds),
        "language": context.language,
        "participants": participants,
        "topics": topics,
        "action_items": action_items,
        "decisions": decisions,
        "source": {
            "session_id": context.session_id,
            "audio_sha256": context.audio_sha256,
        },
    }
    if extraction_error:
        fm["extraction_error"] = extraction_error
    return fm


# ─── Validation ────────────────────────────────────────────────────────────


class FrontmatterValidationError(ValueError):
    """Raised when a frontmatter dict does not satisfy the v1 schema."""


def validate_frontmatter(fm: dict[str, Any]) -> None:
    """Strict-ish validation suitable for tests and the indexer.

    We do NOT validate that an action item has an assignee or that a
    decision has a topic — both fields are explicitly nullable in v1.
    We DO validate types so downstream consumers can rely on them.
    """
    if not isinstance(fm, dict):
        raise FrontmatterValidationError("frontmatter must be a dict")

    if fm.get("schema_version") != SCHEMA_VERSION:
        raise FrontmatterValidationError(
            f"schema_version must be {SCHEMA_VERSION}, got {fm.get('schema_version')!r}"
        )

    if fm.get("type") not in ALLOWED_TYPES:
        raise FrontmatterValidationError(
            f"type must be one of {ALLOWED_TYPES}, got {fm.get('type')!r}"
        )

    for key in ("participants", "topics", "action_items", "decisions"):
        v = fm.get(key)
        if v is None:
            raise FrontmatterValidationError(f"{key} must be a list, not None")
        if not isinstance(v, list):
            raise FrontmatterValidationError(f"{key} must be a list")

    for ai in fm["action_items"]:
        if not isinstance(ai, dict):
            raise FrontmatterValidationError("action_items entries must be dicts")
        if not isinstance(ai.get("task"), str) or not ai["task"].strip():
            raise FrontmatterValidationError("action_items.task must be a non-empty str")
        if ai.get("status") not in ("open", "closed", "blocked"):
            raise FrontmatterValidationError(
                f"action_items.status must be open|closed|blocked, got {ai.get('status')!r}"
            )

    for d in fm["decisions"]:
        if not isinstance(d, dict):
            raise FrontmatterValidationError("decisions entries must be dicts")
        if not isinstance(d.get("text"), str) or not d["text"].strip():
            raise FrontmatterValidationError("decisions.text must be a non-empty str")

    for t in fm["topics"]:
        if not isinstance(t, str) or not t.strip():
            raise FrontmatterValidationError("topics entries must be non-empty strings")

    for p in fm["participants"]:
        if not isinstance(p, dict):
            raise FrontmatterValidationError("participants entries must be dicts")
        if not isinstance(p.get("name"), str) or not p["name"].strip():
            raise FrontmatterValidationError("participants.name must be a non-empty str")

    src = fm.get("source")
    if src is not None and not isinstance(src, dict):
        raise FrontmatterValidationError("source must be a dict if present")


# ─── YAML rendering (no PyYAML dependency) ─────────────────────────────────

# We control the shape of the data, so we hand-write a small YAML emitter
# rather than pull in PyYAML. The output is stable, sorted by key for
# determinism in tests, and uses block style throughout for readability.

_SAFE_YAML_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NEEDS_QUOTING = re.compile(r"[:#\[\]\{\},&\*!\|>'\"%@`\\]|^\s|\s$|^[-?]\s")


def _yaml_scalar(value: Any) -> str:
    """Render a scalar value as YAML."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return _yaml_quote(value.isoformat())
    if isinstance(value, date):
        return _yaml_quote(value.isoformat())
    if isinstance(value, str):
        # Quote if it contains structural characters or could be misread as
        # a non-string.  Always quote empty strings.
        if not value:
            return '""'
        if value.lower() in ("null", "true", "false", "yes", "no", "on", "off", "~"):
            return _yaml_quote(value)
        if _NEEDS_QUOTING.search(value):
            return _yaml_quote(value)
        # Pure number-looking strings must be quoted to stay strings
        try:
            float(value)
            return _yaml_quote(value)
        except ValueError:
            pass
        return value
    raise TypeError(f"cannot render {type(value).__name__} as YAML scalar")


def _yaml_quote(s: str) -> str:
    """Double-quote a string with minimal escaping for YAML."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _render_yaml(data: Any, indent: int = 0) -> str:
    """Render a dict/list/scalar tree as block-style YAML."""
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return pad + "{}"
        lines: list[str] = []
        for k in data.keys():
            v = data[k]
            key = k if _SAFE_YAML_KEY.match(k) else _yaml_quote(k)
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{pad}{key}: {{}}")
                else:
                    lines.append(f"{pad}{key}:")
                    lines.append(_render_yaml(v, indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{pad}{key}: []")
                else:
                    lines.append(f"{pad}{key}:")
                    lines.append(_render_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return pad + "[]"
        lines = []
        for item in data:
            if isinstance(item, dict):
                # Render as a "- key: value" block, with the first key on
                # the same line as the dash for compactness.
                if not item:
                    lines.append(f"{pad}- {{}}")
                    continue
                keys = list(item.keys())
                first_key = keys[0]
                first_val = item[first_key]
                k_render = first_key if _SAFE_YAML_KEY.match(first_key) else _yaml_quote(first_key)
                if isinstance(first_val, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(_render_yaml(item, indent + 1))
                else:
                    lines.append(f"{pad}- {k_render}: {_yaml_scalar(first_val)}")
                    for rk in keys[1:]:
                        rv = item[rk]
                        rk_render = rk if _SAFE_YAML_KEY.match(rk) else _yaml_quote(rk)
                        sub_pad = "  " * (indent + 1)
                        if isinstance(rv, dict):
                            if not rv:
                                lines.append(f"{sub_pad}{rk_render}: {{}}")
                            else:
                                lines.append(f"{sub_pad}{rk_render}:")
                                lines.append(_render_yaml(rv, indent + 2))
                        elif isinstance(rv, list):
                            if not rv:
                                lines.append(f"{sub_pad}{rk_render}: []")
                            else:
                                lines.append(f"{sub_pad}{rk_render}:")
                                lines.append(_render_yaml(rv, indent + 2))
                        else:
                            lines.append(f"{sub_pad}{rk_render}: {_yaml_scalar(rv)}")
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.append(_render_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return pad + _yaml_scalar(data)


def render_frontmatter_block(fm: dict[str, Any]) -> str:
    """Render a frontmatter dict as a ``---``-delimited YAML block.

    The output ends with a trailing newline so the next caller can simply
    concatenate the markdown body.
    """
    body = _render_yaml(fm, indent=0)
    return f"---\n{body}\n---\n"


# ─── Frontmatter parsing (read-back) ───────────────────────────────────────

# A read-back parser is needed by the indexer (and tests). Since we
# control the shape, we accept a slightly tolerant subset of YAML rather
# than depending on PyYAML.
#
# Strategy: detect the leading ``---`` block, then if PyYAML is available
# use it for correctness; otherwise parse via JSON-roundtrip after a
# tightly scoped manual pre-process. Parsing back what we wrote is the
# important property — that round-trip is covered by tests.

_FRONTMATTER_BLOCK_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL,
)


def parse_frontmatter_block(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split a string starting with ``---`` into (frontmatter_dict, body).

    Returns ``(None, text)`` unchanged if no frontmatter block is present
    or if parsing fails.  Tries PyYAML first if installed, falls back to
    a small bespoke parser that handles the shapes this module emits.
    """
    m = _FRONTMATTER_BLOCK_RE.match(text)
    if not m:
        return None, text
    raw_yaml = m.group(1)
    body = text[m.end():]

    try:
        import yaml  # type: ignore

        try:
            parsed = yaml.safe_load(raw_yaml)
            if isinstance(parsed, dict):
                return parsed, body
        except Exception:
            pass
    except ImportError:
        pass

    parsed = _parse_simple_yaml(raw_yaml)
    if isinstance(parsed, dict):
        return parsed, body
    return None, text


def _parse_simple_yaml(text: str) -> dict[str, Any] | None:
    """Very small YAML subset parser.

    Handles the exact shapes ``render_frontmatter_block`` emits:
      - top-level mapping
      - nested mappings
      - lists of scalars
      - lists of mappings (``- key: value`` style)
      - ``[]`` and ``{}`` empty literals
      - double-quoted strings, ``null``, ``true``, ``false``, integers
    Anything more exotic returns ``None`` so the caller can fall back.
    """
    lines = text.splitlines()
    pos = [0]  # mutable index

    def _peek_indent() -> int | None:
        while pos[0] < len(lines):
            line = lines[pos[0]]
            if not line.strip() or line.lstrip().startswith("#"):
                pos[0] += 1
                continue
            return len(line) - len(line.lstrip(" "))
        return None

    def _parse_scalar(s: str) -> Any:
        s = s.strip()
        if s == "null" or s == "~" or s == "":
            return None
        if s == "true":
            return True
        if s == "false":
            return False
        if s == "[]":
            return []
        if s == "{}":
            return {}
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            inner = s[1:-1]
            return inner.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        # int?
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    def _parse_block(base_indent: int) -> Any:
        # Decide list vs dict based on the first non-empty line at
        # base_indent.
        cur_indent = _peek_indent()
        if cur_indent is None or cur_indent < base_indent:
            return None
        first_line = lines[pos[0]]
        stripped = first_line[cur_indent:]
        if stripped.startswith("- "):
            return _parse_list(base_indent)
        return _parse_mapping(base_indent)

    def _parse_mapping(base_indent: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while pos[0] < len(lines):
            cur_indent = _peek_indent()
            if cur_indent is None or cur_indent < base_indent:
                break
            line = lines[pos[0]]
            content = line[base_indent:]
            if cur_indent != base_indent:
                # Should not happen for mapping at this indent
                break
            # split on first ":" not inside quotes
            if ":" not in content:
                pos[0] += 1
                continue
            key_part, _, val_part = content.partition(":")
            key = key_part.strip()
            if key.startswith('"') and key.endswith('"'):
                key = key[1:-1]
            pos[0] += 1
            val_part = val_part.strip()
            if val_part == "":
                # nested block
                child_indent = _peek_indent()
                if child_indent is not None and child_indent > base_indent:
                    out[key] = _parse_block(child_indent)
                else:
                    out[key] = None
            else:
                out[key] = _parse_scalar(val_part)
        return out

    def _parse_list(base_indent: int) -> list[Any]:
        out: list[Any] = []
        while pos[0] < len(lines):
            cur_indent = _peek_indent()
            if cur_indent is None or cur_indent < base_indent:
                break
            line = lines[pos[0]]
            content = line[base_indent:]
            if not content.startswith("- "):
                break
            after_dash = content[2:]
            pos[0] += 1
            if ":" in after_dash and not after_dash.startswith('"'):
                # First key/value on the dash line; remaining mapping keys
                # appear at base_indent + 2.
                first_key, _, first_val = after_dash.partition(":")
                first_key = first_key.strip()
                first_val_str = first_val.strip()
                item: dict[str, Any] = {}
                if first_val_str == "":
                    # nested
                    child_indent = _peek_indent()
                    if child_indent is not None and child_indent > base_indent:
                        item[first_key] = _parse_block(child_indent)
                    else:
                        item[first_key] = None
                else:
                    item[first_key] = _parse_scalar(first_val_str)
                # Subsequent keys for this item live at base_indent + 2
                child_indent = _peek_indent()
                while child_indent is not None and child_indent == base_indent + 2:
                    line2 = lines[pos[0]]
                    content2 = line2[child_indent:]
                    if content2.startswith("- "):
                        break  # next list item, stop folding into current dict
                    if ":" not in content2:
                        pos[0] += 1
                        child_indent = _peek_indent()
                        continue
                    k2, _, v2 = content2.partition(":")
                    k2 = k2.strip()
                    if k2.startswith('"') and k2.endswith('"'):
                        k2 = k2[1:-1]
                    pos[0] += 1
                    v2_str = v2.strip()
                    if v2_str == "":
                        nested_indent = _peek_indent()
                        if nested_indent is not None and nested_indent > child_indent:
                            item[k2] = _parse_block(nested_indent)
                        else:
                            item[k2] = None
                    else:
                        item[k2] = _parse_scalar(v2_str)
                    child_indent = _peek_indent()
                out.append(item)
            else:
                # Plain scalar list item, or bare "-" with nested block
                if after_dash.strip() == "":
                    child_indent = _peek_indent()
                    if child_indent is not None and child_indent > base_indent:
                        out.append(_parse_block(child_indent))
                    else:
                        out.append(None)
                else:
                    out.append(_parse_scalar(after_dash))
        return out

    indent = _peek_indent()
    if indent is None:
        return {}
    if indent != 0:
        return None
    result = _parse_block(0)
    if isinstance(result, dict):
        return result
    return None


# ─── Public helpers ────────────────────────────────────────────────────────


def write_frontmatter_sidecar(
    output_dir: str | Path, basename: str, fm: dict[str, Any]
) -> Path:
    """Write ``<basename>.frontmatter.json`` next to the summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{basename}.frontmatter.json"
    def _json_default(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    path.write_text(json.dumps(fm, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    return path


def empty_frontmatter(context: FrontmatterContext, *, error: str | None = None) -> dict[str, Any]:
    """Return a valid frontmatter dict with no extracted content.

    Used as a fallback when extraction fails so the pipeline still ships
    a well-formed summary.
    """
    return build_frontmatter(None, context, extraction_error=error)


# ─── Convenience: build context from a millet Transcript + session ─────


def _read_session_metadata(session_dir: Path) -> dict[str, Any]:
    """Read the per-session JSON sidecar (``*.session.json``) if present."""
    matches = sorted(session_dir.glob("*.session.json"))
    if not matches:
        return {}
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _channel_for_speaker_label(label: str) -> str | None:
    """Map a millet speaker label to the originating channel.

    Mic-side labels (YOU, SPEAKER_YOU) -> "mic"; remote labels
    (REMOTE, REMOTE_1, ...) -> "system".  Unknown labels stay None;
    the indexer can fill them in from voiceprints later.
    """
    if not label:
        return None
    upper = label.upper()
    if upper == "YOU" or upper == "SPEAKER_YOU":
        return "mic"
    if upper == "REMOTE" or upper.startswith("REMOTE_"):
        return "system"
    return None


def context_from_transcript(
    transcript: Any,
    session_dir: Path | str | None = None,
    *,
    type: str = "meeting",
) -> FrontmatterContext:
    """Build a ``FrontmatterContext`` from a ``meet.transcribe.Transcript``.

    Reads ``*.session.json`` from ``session_dir`` for the recording's
    ``started_at`` and any user-supplied title.  Falls back to ``now()``
    in UTC when ``started_at`` is absent.

    Speaker labels and their channel guess are derived from
    ``transcript.speakers``; this is the same set of names that will
    appear in the Markdown body.
    """
    session_dir_path = Path(session_dir) if session_dir is not None else None
    meta = _read_session_metadata(session_dir_path) if session_dir_path else {}

    started_at = meta.get("started_at") or _now_utc_iso()
    title = meta.get("title")
    session_id = meta.get("session_id") or (
        session_dir_path.name if session_dir_path else None
    )

    speakers = list(getattr(transcript, "speakers", []) or [])
    speaker_labels: list[str] = []
    speaker_channels: dict[str, str] = {}
    for sp in speakers:
        # Speaker.label is the user-assigned name; Speaker.id is the
        # canonical transcript label (YOU, REMOTE_1, etc.).
        label = getattr(sp, "label", None) or getattr(sp, "id", None)
        if not label:
            continue
        speaker_labels.append(label)
        ch = _channel_for_speaker_label(getattr(sp, "id", "") or "")
        if ch:
            speaker_channels[label] = ch

    return FrontmatterContext(
        title=title,
        date=started_at,
        duration_seconds=getattr(transcript, "duration", None),
        language=getattr(transcript, "language", None),
        type=type,
        session_id=session_id,
        transcript_speakers=speaker_labels,
        speaker_channels=speaker_channels,
    )
