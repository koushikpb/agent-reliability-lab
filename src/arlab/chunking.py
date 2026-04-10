import re
from dataclasses import dataclass

import yaml

# all-MiniLM-L6-v2 truncates input at 256 word pieces, so anything past that never reaches the
# embedding. ~180 English words fits; code and flag names tokenize worse and can still be cut off.
MAX_WORDS = 180
# Bump when chunking logic changes chunk ids or text, so index_version changes with it.
CHUNKER_VERSION = 2

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo renders these as localized section titles; without them the heading is empty and the
# section silently merges into the one before it.
HEADING_SHORTCODE = re.compile(r'\{\{%\s*heading\s+"(\w+)"\s*%\}\}')
HEADING_TITLES = {
    "prerequisites": "Before you begin",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
    "whatsnext": "What's next",
    "seealso": "See also",
}
GLOSSARY = re.compile(r"\{\{<\s*glossary_tooltip\b(.*?)>\}\}", re.DOTALL)
# Removes only the tags of paired shortcodes like note/warning, so their inner text stays.
# include and code_sample point at other files, and their content is lost.
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*#*\s*$")
# "## Rolling back {#rollback}" sets the page anchor; using it keeps chunk ids equal to doc anchors.
ANCHOR = re.compile(r"\s*\{#([\w-]+)\}$")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_path: str
    heading: str
    text: str


def slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _glossary_text(match: re.Match[str]) -> str:
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
    return attrs.get("text") or attrs.get("term_id", "")


def clean_markdown(raw: str) -> tuple[str, str]:
    title = ""
    front = FRONT_MATTER.match(raw)
    if front:
        meta = yaml.safe_load(front.group(1)) or {}
        title = str(meta.get("title", ""))
        raw = raw[front.end() :]
    raw = HTML_COMMENT.sub("", raw)
    raw = HEADING_SHORTCODE.sub(lambda m: HEADING_TITLES.get(m.group(1), m.group(1)), raw)
    raw = GLOSSARY.sub(_glossary_text, raw)
    raw = SHORTCODE.sub("", raw)
    return title, raw


def _sections(body: str) -> list[tuple[list[str], str | None, str]]:
    """Split on ## and ### headings outside code fences; returns (heading path, anchor, text)."""
    sections: list[tuple[list[str], str | None, str]] = []
    path: list[str] = []
    anchor: str | None = None
    lines: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        sections.append((path, anchor, "\n".join(lines)))
        title = match.group(2)
        explicit = ANCHOR.search(title)
        anchor = explicit.group(1) if explicit else None
        title = ANCHOR.sub("", title)
        level = len(match.group(1))
        path = (path[:1] if level == 3 else []) + [title]
        lines = []
    sections.append((path, anchor, "\n".join(lines)))
    return sections


def _blocks(text: str) -> list[str]:
    """Paragraphs split at blank lines, keeping each code fence whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_to_cap(text: str, max_words: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for block in _blocks(text):
        words = len(block.split())
        if words > max_words:
            # A single block over the cap (usually a long code sample) loses its line breaks here.
            if current:
                parts.append("\n\n".join(current))
                current, size = [], 0
            tokens = block.split()
            parts.extend(
                " ".join(tokens[i : i + max_words]) for i in range(0, len(tokens), max_words)
            )
            continue
        if size + words > max_words and current:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += words
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_document(doc_path: str, raw: str, max_words: int = MAX_WORDS) -> list[Chunk]:
    title, body = clean_markdown(raw)
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for path, anchor, text in _sections(body):
        parts = _split_to_cap(text, max_words)
        if not parts:
            continue
        # Repeated headings get -1, -2 like Hugo anchors, so ids stay unique within a doc.
        slug = anchor or (slugify(path[-1]) if path else "intro")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        unique = slug if count == 0 else f"{slug}-{count}"
        heading = " > ".join([title, *path]) if title else " > ".join(path)
        for i, part in enumerate(parts):
            chunks.append(Chunk(f"{doc_path}#{unique}-{i}", doc_path, heading, part))
    return chunks
