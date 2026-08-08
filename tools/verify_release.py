"""Run repository hygiene checks before publishing the PhysX-CoT code."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    for path in root.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(ord(char) > 127 for char in path.name):
            errors.append(f"non-ASCII filename: {path.relative_to(root)}")
        if not path.is_file() or path.suffix in {".png", ".jpg", ".jpeg", ".webp", ".glb", ".mp4"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(ord(char) > 127 for char in text):
            errors.append(f"non-ASCII text: {path.relative_to(root)}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(f"syntax error: {path.relative_to(root)}:{exc.lineno}: {exc.msg}")
        private_tokens = ("/workspace/", "PhysX-Anything-main", "C:" + "\\Users\\")
        if any(token in text for token in private_tokens):
            errors.append(f"private path: {path.relative_to(root)}")

    if errors:
        print("Release verification failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Release verification passed: ASCII text, public paths, and Python syntax are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
