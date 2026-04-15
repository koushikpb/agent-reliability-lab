from pathlib import Path

import pytest

from arlab import corpus
from arlab.chunking import MAX_TOKENS, chunk_document

CORPUS = Path("data/corpus")

pytestmark = pytest.mark.skipif(
    not (CORPUS / corpus.DOCS_ROOT).exists(), reason="corpus not fetched, run `arlab ingest`"
)


@pytest.fixture(scope="module")
def chunks():
    from arlab.ingest import count_tokens

    return [
        c
        for path, raw in corpus.iter_docs(CORPUS)
        for c in chunk_document(
            path, raw, count_tokens, read_example=lambda f: corpus.read_example(CORPUS, f)
        )
    ]


def test_real_corpus_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_real_corpus_embedding_input_fits_the_model(chunks):
    from arlab.ingest import count_tokens, embedding_input

    over = [c.id for c in chunks if count_tokens(embedding_input(c)) > MAX_TOKENS]
    # A single unsplittable token run can exceed the budget; there should be almost none.
    assert len(over) <= 5, over[:5]


def test_real_corpus_has_no_hugo_shortcodes_left(chunks):
    assert not [c.id for c in chunks if "{{<" in c.text or "{{%" in c.text]
