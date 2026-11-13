from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_readme_links  # noqa: E402


class DocumentationTests(unittest.TestCase):
    def test_repository_markdown_links_resolve(self) -> None:
        self.assertEqual(check_readme_links.check(ROOT), [])

    def test_anchor_checker_catches_missing_fragment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-doc-test-") as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# Present\n\n[ok](#present)\n[bad](#missing)\n")
            errors = check_readme_links.check(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("missing anchor", errors[0])


if __name__ == "__main__":
    unittest.main()
