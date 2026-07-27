"""Shared fixtures.

The two API keys are injected before ``src.config`` is first imported so the
whole suite runs offline: every LLM call is mocked, and a real key is never
needed to exercise anything but a live smoke test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must precede any `from src...` import: Settings has no defaults for these.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
# Keep torch single-threaded in CI so N pytest workers don't thrash the box.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="session")
def sample_markdown_bytes() -> bytes:
    return (
        b"# Vector Databases\n\n"
        b"A vector database stores high-dimensional embeddings and supports "
        b"approximate nearest neighbour search over them.\n\n"
        b"## Cost model\n\n"
        b"Managed vector databases bill for provisioned capacity, so a large "
        b"but lightly queried index pays full rent for occasional reads.\n"
    )


@pytest.fixture(scope="session")
def sample_html_bytes() -> bytes:
    return (
        b"<html><head><title>T</title><style>body{color:red}</style></head>"
        b"<body><h1>Embedded Stores</h1>"
        b"<p>LanceDB is an embedded columnar vector database built on Arrow.</p>"
        b"<script>console.log('should not be ingested')</script>"
        b"<p>It requires no server process and no managed account.</p>"
        b"</body></html>"
    )


@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    return tmp_path / "lancedb_store"
