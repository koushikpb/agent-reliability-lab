import subprocess
from collections.abc import Iterator
from pathlib import Path

REPO = "https://github.com/kubernetes/website.git"
# Pinned so chunk ids, and the relevance labels that point at them, never shift under us.
COMMIT = "6c42e266fcc1f0e0edd4805baea904e1b3cbaf9c"
DOCS_ROOT = "content/en/docs"
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


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def fetch_corpus(dest: Path) -> None:
    """Sparse, blobless fetch of just the doc sections at the pinned commit."""
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "-q")
    _git(dest, "remote", "add", "origin", REPO)
    _git(dest, "sparse-checkout", "set", *(f"{DOCS_ROOT}/{s}" for s in SECTIONS))
    _git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    _git(dest, "checkout", "-q", "FETCH_HEAD")


def iter_docs(corpus_dir: Path) -> Iterator[tuple[str, str]]:
    """Yields (doc path relative to the docs root without .md, raw markdown), in a stable order."""
    root = corpus_dir / DOCS_ROOT
    for section in SECTIONS:
        for path in sorted((root / section).rglob("*.md")):
            yield path.relative_to(root).with_suffix("").as_posix(), path.read_text()
