"""공유할 노트북은 코드와 설명만 담는다."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.rglob("*.ipynb")))
def test_notebook_has_no_saved_media(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert "widgets" not in notebook.get("metadata", {})
    for cell in notebook["cells"]:
        assert not cell.get("attachments")
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
        assert "data:image/" not in "".join(cell.get("source", []))
