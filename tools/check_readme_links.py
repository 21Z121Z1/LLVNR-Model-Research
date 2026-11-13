#!/usr/bin/env python3
"""Check relative Markdown links and fragments without network access."""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
IGNORED = {".git", ".build", "__pycache__", "dist", "Validation"}


def slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\u3400-\u9fff-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def anchors(path: Path) -> set[str]:
    result: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slug(match.group(1))
        number = counts.get(base, 0)
        result.add(base if number == 0 else f"{base}-{number}")
        counts[base] = number + 1
    return result


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.md")
        if path.is_file() and not any(part in IGNORED for part in path.relative_to(root).parts)
    )


def check(root: Path) -> list[str]:
    errors: list[str] = []
    cache: dict[Path, set[str]] = {}
    for source in markdown_files(root):
        text = source.read_text(errors="replace")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "codex://", "app://")):
                continue
            if target.startswith("//"):
                continue
            path_text, _, fragment = target.partition("#")
            path_text = unquote(path_text)
            destination = source if not path_text else (source.parent / path_text).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                errors.append(f"{source.relative_to(root)}: link escapes repository: {target}")
                continue
            if not destination.exists():
                errors.append(f"{source.relative_to(root)}: missing link target: {target}")
                continue
            if fragment and destination.suffix.lower() == ".md":
                if destination not in cache:
                    cache[destination] = anchors(destination)
                if unquote(fragment) not in cache[destination]:
                    errors.append(f"{source.relative_to(root)}: missing anchor #{fragment} in {destination.relative_to(root)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    errors = check(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"checked {len(markdown_files(root))} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
