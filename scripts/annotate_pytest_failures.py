"""Turn a pytest run's failures into GitHub Actions annotations.

    pytest ... | tee pytest-output.txt
    python scripts/annotate_pytest_failures.py pytest-output.txt

**Why this exists.** A failing CI job's annotation says only "Process completed
with exit code 1". The assertion and its traceback live in the step log, and
reading that log requires write access to the repository - so a contributor
without it, or any tool reading the public API, is reduced to guessing what
broke from the job name alone.

Annotations, unlike logs, are public. Emitting the failure summary and the
traceback as `::error::` workflow commands makes a red build explain itself to
whoever can see it, which is the whole point of running it in public.

A workflow command is a single line, so newlines are escaped as `%0A`; GitHub
renders them as a multi-line annotation. `%` must be escaped first or the
escaping eats itself.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: GitHub truncates very long annotations, and a wall of text is unreadable
#: anyway. The first failure's traceback is almost always the informative one.
MAX_TRACEBACK_CHARS = 6000


def escape(text: str) -> str:
    """Encode a multi-line message for a workflow command.

    Order matters: `%` must be escaped before the `%0A` sequences are
    introduced, otherwise their own percent signs get double-encoded and the
    annotation renders as literal `%250A`.
    """
    return (
        text.replace("%", "%25")
        .replace("\r", "")
        .replace("\n", "%0A")
    )


def summary_lines(output: str) -> list[str]:
    """The one-line-per-failure summary pytest prints at the end."""
    return [
        line.strip()
        for line in output.splitlines()
        if line.startswith(("FAILED", "ERROR"))
    ]


def first_traceback(output: str) -> str | None:
    """The FAILURES section, which holds the actual assertion and traceback."""
    match = re.search(
        r"=+ FAILURES =+\n(.*?)(?:\n=+ (?:warnings summary|short test summary|ERRORS))",
        output,
        re.S,
    )
    if not match:
        return None
    body = match.group(1).strip()
    return body[:MAX_TRACEBACK_CHARS] if body else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="captured pytest output")
    args = parser.parse_args()

    if not args.output.exists():
        print(f"::error::{args.output} not found - pytest produced no output")
        return 0

    text = args.output.read_text(encoding="utf-8", errors="replace")

    for line in summary_lines(text):
        print(f"::error::{escape(line)}")

    traceback = first_traceback(text)
    if traceback:
        print(f"::error title=pytest traceback::{escape(traceback)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
