import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

REPO = "https://github.com/kubernetes/website.git"
# Pinned so chunk ids, and the relevance labels that point at them, never shift under us.
COMMIT = "6c42e266fcc1f0e0edd4805baea904e1b3cbaf9c"
DOCS_ROOT = "content/en/docs"
EXAMPLES_ROOT = "content/en/examples"
SECTIONS = [
    "concepts/cluster-administration",
    "concepts/configuration",
    "concepts/containers",
    "concepts/policy",
    "concepts/scheduling-eviction",
    "concepts/security",
    "concepts/services-networking",
    "concepts/storage",
    "concepts/workloads",
    "tasks/access-application-cluster",
    "tasks/administer-cluster",
    "tasks/configure-pod-container",
    "tasks/debug",
    "tasks/inject-data-application",
    "tasks/manage-kubernetes-objects",
    "tasks/network",
    "tasks/run-application",
]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_corpus(dest: Path) -> None:
    """Sparse, blobless fetch of the doc sections and examples at the pinned commit."""
    # Fetched into a temp dir and renamed when complete: blobs arrive during checkout, so a network
    # drop can leave a half-written tree that a later run would otherwise accept.
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _git(tmp, "init", "-q")
    _git(tmp, "remote", "add", "origin", REPO)
    paths = [f"{DOCS_ROOT}/{s}" for s in SECTIONS] + [EXAMPLES_ROOT]
    _git(tmp, "sparse-checkout", "set", *paths)
    _git(tmp, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    _git(tmp, "checkout", "-q", "FETCH_HEAD")
    tmp.rename(dest)


def ensure_corpus(corpus_dir: Path) -> None:
    if not corpus_dir.exists():
        fetch_corpus(corpus_dir)
    head = _git(corpus_dir, "rev-parse", "HEAD")
    if head != COMMIT:
        raise RuntimeError(f"{corpus_dir} is at {head}, expected {COMMIT}; delete it to refetch")
    if not (corpus_dir / EXAMPLES_ROOT).exists():
        # Dirs fetched before examples were added; widening the sparse set fetches the new blobs.
        _git(corpus_dir, "sparse-checkout", "add", EXAMPLES_ROOT)


def read_example(corpus_dir: Path, file: str) -> str | None:
    path = corpus_dir / EXAMPLES_ROOT / file
    return path.read_text(encoding="utf-8") if path.is_file() else None


def iter_docs(corpus_dir: Path) -> Iterator[tuple[str, str]]:
    """Yields (doc path relative to the docs root without .md, raw markdown), in a stable order."""
    root = corpus_dir / DOCS_ROOT
    for section in SECTIONS:
        for path in sorted((root / section).rglob("*.md")):
            yield (
                path.relative_to(root).with_suffix("").as_posix(),
                path.read_text(encoding="utf-8"),
            )
