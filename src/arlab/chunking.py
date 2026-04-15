import re
from collections.abc import Callable
from dataclasses import dataclass

import yaml

# all-MiniLM-L6-v2 truncates at 256 word pieces including [CLS] and [SEP], so anything past 254
# never reaches the embedding. A word cap didn't hold: code and flag names tokenize to many pieces,
# and 31% of 180-word chunks were cut off. The budget covers heading + text, the embedding input.
MAX_TOKENS = 254
# Sections smaller than this are mostly heading once embedded, and crowd the top-k.
MIN_TOKENS = 16
# Link lists with no answers in them.
SKIPPED_SECTIONS = {"What's next", "See also"}
# Bump when chunking logic changes chunk ids or text, so index_version changes with it.
CHUNKER_VERSION = 4

TokenCounter = Callable[[str], int]
ReadExample = Callable[[str], str | None]

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
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
# Examples are often where field names and error-prone settings appear, so they are inlined.
CODE_SAMPLE = re.compile(r"\{\{[<%]\s*code_sample\b(.*?)[>%]\}\}", re.DOTALL)
# Removes only the tags of paired shortcodes like note/warning, so their inner text stays.
# include points at shared boilerplate (cluster prerequisites) and its content is dropped.
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
    attrs = _attrs(match.group(1))
    return attrs.get("text") or attrs.get("term_id", "")


def _attrs(text: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', text))


def _code_sample(match: re.Match[str], read_example: ReadExample) -> str:
    attrs = _attrs(match.group(1))
    content = read_example(attrs.get("file", ""))
    if content is None:
        return ""
    language = attrs.get("language") or attrs.get("file", "").rsplit(".", 1)[-1]
    return f"```{language}\n{content.rstrip()}\n```"


def clean_markdown(raw: str, read_example: ReadExample = lambda file: None) -> tuple[str, str]:
    title = ""
    front = FRONT_MATTER.match(raw)
    if front:
        meta = yaml.safe_load(front.group(1)) or {}
        title = str(meta.get("title", ""))
        raw = raw[front.end() :]
    raw = HTML_COMMENT.sub("", raw)
    raw = HEADING_SHORTCODE.sub(lambda m: HEADING_TITLES.get(m.group(1), m.group(1)), raw)
    raw = GLOSSARY.sub(_glossary_text, raw)
    raw = CODE_SAMPLE.sub(lambda m: _code_sample(m, read_example), raw)
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


def _pack(pieces: list[str], sep: str, budget: int, count: TokenCounter) -> list[str]:
    """Greedily join pieces into parts of at most budget tokens; oversized pieces split further."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        n = count(piece)
        if current and size + n > budget:
            parts.append(sep.join(current))
            current, size = [], 0
        if n > budget:
            parts.extend(_split_oversized(piece, budget, count))
            continue
        current.append(piece)
        size += n
    if current:
        parts.append(sep.join(current))
    return parts


def _split_oversized(piece: str, budget: int, count: TokenCounter) -> list[str]:
    # Line breaks first so lists and code keep their shape; words only for a single huge line.
    lines = [line for line in piece.splitlines() if line.strip()]
    if len(lines) > 2 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        # Re-fence each part so a split code sample stays valid markdown when cited.
        fence = lines[0]
        inner_budget = budget - count(fence) - count("```")
        return [f"{fence}\n{part}\n```" for part in _pack(lines[1:-1], "\n", inner_budget, count)]
    if len(lines) > 1:
        return _pack(lines, "\n", budget, count)
    tokens = piece.split()
    if len(tokens) > 1:
        return _pack(tokens, " ", budget, count)
    return [piece]


def chunk_document(
    doc_path: str,
    raw: str,
    count: TokenCounter,
    max_tokens: int = MAX_TOKENS,
    read_example: ReadExample = lambda file: None,
) -> list[Chunk]:
    title, body = clean_markdown(raw, read_example)
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for path, anchor, text in _sections(body):
        if not text.strip() or (path and path[-1] in SKIPPED_SECTIONS):
            continue
        heading = " > ".join([title, *path]) if title else " > ".join(path)
        if chunks and path and count(text) < MIN_TOKENS:
            merged = f"{chunks[-1].text}\n\n{path[-1]}\n{text.strip()}"
            if count(chunks[-1].heading) + count(merged) <= max_tokens:
                chunks[-1] = Chunk(chunks[-1].id, doc_path, chunks[-1].heading, merged)
                continue
        parts = _pack(_blocks(text), "\n\n", max_tokens - count(heading), count)
        # Repeated headings get -1, -2 like Hugo anchors, so ids stay unique within a doc.
        slug = anchor or (slugify(path[-1]) if path else "intro")
        n_seen = seen.get(slug, 0)
        seen[slug] = n_seen + 1
        unique = slug if n_seen == 0 else f"{slug}-{n_seen}"
        for i, part in enumerate(parts):
            chunks.append(Chunk(f"{doc_path}#{unique}-{i}", doc_path, heading, part))
    return chunks
