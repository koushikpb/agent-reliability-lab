from arlab.chunking import MAX_WORDS, chunk_document, clean_markdown, slugify

DOC = """---
title: Debug Pods
content_type: task
---

<!-- overview -->
This guide helps you debug {{< glossary_tooltip text="Pods" term_id="pod" >}}.

## Debugging Pods

{{< note >}}
Check events first.
{{< /note >}}

```shell
# not a heading
kubectl describe pods ${POD_NAME}
```

### My pod stays pending

The scheduler cannot place it.

## Examples

First.

## Examples

Second.
"""


def test_clean_markdown_strips_front_matter_comments_and_shortcodes():
    title, body = clean_markdown(DOC)
    assert title == "Debug Pods"
    assert "content_type" not in body
    assert "<!--" not in body
    assert "{{" not in body
    assert "debug Pods." in body
    assert "Check events first." in body


def test_glossary_tooltip_without_text_falls_back_to_term_id():
    _, body = clean_markdown('Use a {{< glossary_tooltip term_id="configmap" >}} here.')
    assert body.strip() == "Use a configmap here."


def test_chunks_split_on_headings_not_code_comments():
    chunks = chunk_document("tasks/debug/debug-pods", DOC)
    ids = [c.id for c in chunks]
    assert ids == [
        "tasks/debug/debug-pods#intro-0",
        "tasks/debug/debug-pods#debugging-pods-0",
        "tasks/debug/debug-pods#my-pod-stays-pending-0",
        "tasks/debug/debug-pods#examples-0",
        "tasks/debug/debug-pods#examples-1-0",
    ]
    assert "# not a heading" in chunks[1].text
    assert chunks[2].heading == "Debug Pods > Debugging Pods > My pod stays pending"
    assert all(c.doc_path == "tasks/debug/debug-pods" for c in chunks)


def test_long_section_splits_under_cap_with_numbered_parts():
    para = " ".join(["word"] * 100)
    doc = "---\ntitle: T\n---\n## Big\n\n" + "\n\n".join([para] * 5)
    chunks = chunk_document("d", doc)
    assert [c.id for c in chunks] == ["d#big-0", "d#big-1", "d#big-2", "d#big-3", "d#big-4"]
    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)


def test_oversized_single_block_is_split_by_words():
    doc = "---\ntitle: T\n---\n## Big\n\n" + " ".join(["w"] * (MAX_WORDS * 2 + 1))
    chunks = chunk_document("d", doc)
    assert len(chunks) == 3
    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)


def test_code_fence_is_not_split_at_blank_lines():
    doc = "---\ntitle: T\n---\n## S\n\n```yaml\na: 1\n\nb: 2\n```\n"
    chunks = chunk_document("d", doc)
    assert len(chunks) == 1
    assert "a: 1\n\nb: 2" in chunks[0].text


def test_empty_sections_are_dropped():
    doc = "---\ntitle: T\n---\n## Empty\n\n## Full\n\ntext\n"
    assert [c.id for c in chunk_document("d", doc)] == ["d#full-0"]


def test_slugify():
    assert slugify("My Pod: stays `Pending`!") == "my-pod-stays-pending"


def test_explicit_anchor_sets_slug_and_is_dropped_from_heading():
    doc = "---\ntitle: T\n---\n## Rolling back to a previous revision {#rollback}\n\ntext\n"
    [chunk] = chunk_document("d", doc)
    assert chunk.id == "d#rollback-0"
    assert chunk.heading == "T > Rolling back to a previous revision"


def test_heading_shortcode_becomes_a_real_heading():
    doc = '---\ntitle: T\n---\n## A\n\none\n\n## {{% heading "whatsnext" %}}\n\nlinks\n'
    chunks = chunk_document("d", doc)
    assert [c.id for c in chunks] == ["d#a-0", "d#what-s-next-0"]
    assert chunks[1].heading == "T > What's next"
