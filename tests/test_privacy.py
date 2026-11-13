from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PrivacyTests(unittest.TestCase):
    def test_sanitizer_removes_identity_and_raw_frames(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llvnr-privacy-test-") as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            user_home = "/" + "Users" + "/example"
            summary = {
                "host": "private-host.example",
                "runDirectory": user_home + "/Library/Application Support/LLVNRDemo/Runs/1",
                "source_model_sha256": "a" * 64,
                "error": "failed at " + user_home + "/private/model",
                "nested": {"path": user_home + "/private", "value": 3},
            }
            (run / "summary.json").write_text(json.dumps(summary))
            (run / "frames.jsonl").write_text(
                json.dumps({"event": "received", "id": 1}) + "\n" +
                json.dumps({"event": "recoverable_error", "error": "bad " + user_home + "/file"}) + "\n"
            )
            output = Path(tmp) / "shareable"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/sanitize_run_report.py"), str(run), "--output", str(output)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (output / "summary.json").read_text()
            self.assertNotIn("private-host", text)
            self.assertNotIn("/" + "Users" + "/", text)
            self.assertNotIn("runDirectory", text)
            self.assertNotIn("frames.redacted.jsonl", {p.name for p in output.iterdir()})
            manifest = json.loads((output / "SANITIZATION.json").read_text())
            self.assertFalse(manifest["frames_included"])
            self.assertFalse(manifest["raw_frames_omitted_by_default"] is False)


if __name__ == "__main__":
    unittest.main()
