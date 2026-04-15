from arlab.chunking import chunk_document, clean_markdown, slugify


def words(text: str) -> int:
    return len(text.split())


def chunk(doc: str, max_tokens: int = 254, read_example=None):
    return chunk_document("d", doc, words, max_tokens=max_tokens, read_example=read_example)


DOC = """---
title: Debug Pods
content_type: task
---

<!-- overview -->
This guide helps you debug {{< glossary_tooltip text="Pods" term_id="pod" >}} that run in a
cluster and do not behave the way you expect them to.

## Debugging Pods

{{< note >}}
Check events first, because they often name the failing container and the reason.
{{< /note >}}

```shell
# not a heading
kubectl describe pods ${POD_NAME}
```

### My pod stays pending

The scheduler cannot place it on any node, usually because of resource requests or taints. Check its events to confirm.

## Examples

First example of a pod that fails to start because its image name has a typo in it. Check its events to confirm.

## Examples

Second example of a pod that fails because a ConfigMap it mounts does not exist yet. Check its events to confirm.
"""


def test_clean_markdown_strips_front_matter_comments_and_shortcodes():
    title, body = clean_markdown(DOC)
    assert title == "Debug Pods"
    assert "content_type" not in body
    assert "<!--" not in body
    assert "{{" not in body
    assert "debug Pods that run" in body
    assert "Check events first" in body


def test_front_matter_without_trailing_newline():
    title, body = clean_markdown("---\ntitle: T\n---")
    assert (title, body) == ("T", "")


def test_glossary_tooltip_without_text_falls_back_to_term_id():
    _, body = clean_markdown('Use a {{< glossary_tooltip term_id="configmap" >}} here.')
    assert body.strip() == "Use a configmap here."


def test_code_sample_is_inlined_as_a_fence():
    files = {"pods/simple-pod.yaml": "apiVersion: v1\nkind: Pod"}
    _, body = clean_markdown(
        'See:\n{{% code_sample file="pods/simple-pod.yaml" %}}\n', read_example=files.get
    )
    assert "```yaml\napiVersion: v1\nkind: Pod\n```" in body


def test_missing_code_sample_is_dropped():
    _, body = clean_markdown('{{% code_sample file="nope.yaml" %}}', read_example=lambda f: None)
    assert body == ""


def test_chunks_split_on_headings_not_code_comments():
    chunks = chunk(DOC)
    assert [c.id for c in chunks] == [
        "d#intro-0",
        "d#debugging-pods-0",
        "d#my-pod-stays-pending-0",
        "d#examples-0",
        "d#examples-1-0",
    ]
    assert "# not a heading" in chunks[1].text
    assert chunks[2].heading == "Debug Pods > Debugging Pods > My pod stays pending"


def test_budget_includes_the_heading():
    # Embedding input is heading + text, so the heading's tokens come out of the budget.
    doc = "---\ntitle: T\n---\n## Big\n\n" + "\n\n".join(["w " * 8] * 3)
    chunks = chunk(doc, max_tokens=20)
    assert all(words(c.heading) + words(c.text) <= 20 for c in chunks)
    assert len(chunks) == 2


def test_long_list_splits_at_line_breaks_not_mid_line():
    items = "\n".join(f"- item {i} has four words" for i in range(10))
    chunks = chunk("---\ntitle: T\n---\n## L\n\n" + items, max_tokens=20)
    for c in chunks:
        assert all(line.startswith("- item") for line in c.text.splitlines())
    assert sum(len(c.text.splitlines()) for c in chunks) == 10


def test_single_huge_line_falls_back_to_words():
    chunks = chunk("---\ntitle: T\n---\n## L\n\n" + "w " * 50, max_tokens=20)
    assert all(words(c.heading) + words(c.text) <= 20 for c in chunks)
    assert sum(words(c.text) for c in chunks) == 50


def test_code_fence_is_not_split_at_blank_lines():
    chunks = chunk("---\ntitle: T\n---\n## S\n\n```yaml\na: 1\n\nb: 2\n```\n")
    assert len(chunks) == 1
    assert "a: 1\n\nb: 2" in chunks[0].text


def test_tiny_section_merges_into_previous_chunk():
    doc = (
        "---\ntitle: T\n---\n## Main\n\n"
        + "word " * 40
        + "\n\n## Clean up\n\nDelete the namespace.\n"
    )
    [c] = chunk(doc)
    assert c.id == "d#main-0"
    assert c.text.endswith("Clean up\nDelete the namespace.")


def test_link_list_sections_are_dropped():
    doc = "---\ntitle: T\n---\n## A\n\n" + "word " * 40 + "\n\n## What's next\n\n* [x](/y)\n"
    assert [c.id for c in chunk(doc)] == ["d#a-0"]


def test_empty_sections_are_dropped():
    doc = "---\ntitle: T\n---\n## Empty\n\n## Full\n\n" + "word " * 40
    assert [c.id for c in chunk(doc)] == ["d#full-0"]


def test_explicit_anchor_sets_slug_and_is_dropped_from_heading():
    doc = "---\ntitle: T\n---\n## Rolling back to a previous revision {#rollback}\n\n" + "w " * 40
    [c] = chunk(doc)
    assert c.id == "d#rollback-0"
    assert c.heading == "T > Rolling back to a previous revision"


def test_heading_shortcode_becomes_a_real_heading():
    doc = "---\ntitle: T\n---\n## A\n\n" + "w " * 40 + '\n\n## {{% heading "prerequisites" %}}\n\n'
    doc += "v " * 40
    chunks = chunk(doc)
    assert [c.id for c in chunks] == ["d#a-0", "d#before-you-begin-0"]


def test_slugify():
    assert slugify("My Pod: stays `Pending`!") == "my-pod-stays-pending"


def test_split_code_block_is_refenced_in_every_part():
    code = "```yaml\n" + "\n".join(f"key{i}: value" for i in range(30)) + "\n```"
    chunks = chunk("---\ntitle: T\n---\n## S\n\n" + code, max_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[0] == "```yaml" and lines[-1] == "```"
        assert words(c.heading) + words(c.text) <= 20
