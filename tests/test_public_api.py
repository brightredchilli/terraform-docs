"""The surface other projects depend on.

These guard the library contract rather than behaviour: what is importable,
what importing costs, and that the packaged data travels with the wheel.
"""

from __future__ import annotations

import subprocess
import sys


import terraform_docs_mcp
from terraform_docs_mcp._config import data_dir


class TestPublicSurface:
    def test_exports_only_the_consumer_facing_api(self):
        """The package root is a facade, not a dumping ground.

        Corpus globs, provider->submodule mappings and the packaged-data path
        are build-time details. Exporting them would invite consumers to depend
        on internals we want to stay free to change.
        """
        assert set(terraform_docs_mcp.__all__) == {
            "Index",
            "IndexUnavailable",
            "__version__",
        }

    def test_internals_are_not_re_exported(self):
        for name in ("PROVIDERS", "DOC_GLOB", "DOCS_SUBPATH", "DOC_SUFFIXES", "data_dir"):
            assert not hasattr(terraform_docs_mcp, name), f"{name} leaked into the API"

    def test_submodules_do_not_import_from_the_package_root(self):
        """Guards the import cycle that forced this layout.

        If a submodule imports from ``terraform_docs_mcp`` directly, the root
        can no longer import ``index`` to re-export ``Index``.
        """
        import pathlib

        src = pathlib.Path(terraform_docs_mcp.__file__).parent
        offenders = [
            p.name
            for p in src.glob("*.py")
            if p.name != "__init__.py" and "\nfrom . import " in p.read_text()
        ]
        assert offenders == [], f"these import from the package root: {offenders}"

    def test_index_importable_from_package_root(self):
        from terraform_docs_mcp import Index, IndexUnavailable

        assert Index.__module__ == "terraform_docs_mcp.index"
        assert issubclass(IndexUnavailable, RuntimeError)

    def test_py_typed_marker_present(self):
        """Without this, type checkers ignore our annotations in consumers."""
        assert (data_dir().parent / "py.typed").exists()


class TestImportCost:
    """Importing the package must not drag in the deep-learning stack.

    numpy is unavoidable -- it is what the vector search runs on. torch is not:
    it arrives only when an embedding model is actually constructed, which
    happens on the first search. Keeping that boundary means a consumer that
    imports this package at module scope pays ~30 ms, not several seconds.
    """

    @staticmethod
    def _modules_after(statement: str) -> set[str]:
        code = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        return set(out.stdout.split())

    def test_import_does_not_load_torch(self):
        loaded = self._modules_after("from terraform_docs_mcp import Index")
        assert "torch" not in loaded, "torch must wait until an index is searched"

    def test_constructing_the_index_still_does_not_load_torch(self):
        """Even opening the index is torch-free; only searching needs the model."""
        loaded = self._modules_after(
            "from terraform_docs_mcp import Index; Index()"
        )
        assert "torch" not in loaded
