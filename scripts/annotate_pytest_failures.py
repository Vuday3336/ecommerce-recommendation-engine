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

#: GitHub keeps at most 10 annotations per step and silently drops the rest.
#: That is why the traceback is emitted *first* and the per-test lines are
#: capped below it: on the first run of this script the budget was spent on
#: summary lines and the traceback - the only part that actually explains the
#: failure - was the one thing discarded.
MAX_SUMMARY_ANNOTATIONS = 8


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
    """The one-line-per-failure summary pytest prints at the end.

    Read from the "short test summary info" section specifically, not by
    scanning the whole output for lines starting with FAILED or ERROR. The
    naive version matched pytest's *captured log records* too - a test that
    logs `ERROR  app.main:main.py:194 unhandled error` produces a line starting
    with ERROR that is output, not a result. Seven of those consumed the
    annotation budget and pushed out the traceback.
    """
    section = re.search(
        r"=+ short test summary info =+\n(.*?)(?:\n=+ .* =+|\Z)", output, re.S
    )
    if not section:
        return []
    return [
        line.strip()
        for line in section.group(1).splitlines()
        # A result line names a test node; a log record does not.
        if line.startswith(("FAILED", "ERROR")) and "::" in line
    ]


def strip_framework_frames(body: str) -> str:
    """Drop stack frames from installed packages.

    A FastAPI request traceback is mostly starlette and fastapi frames
    describing how ASGI middleware calls itself. None of it is actionable, and
    it is what pushes the application frames - and the exception - past the
    length limit.
    """
    kept: list[str] = []
    skipping = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("File "):
            skipping = "site-packages" in stripped or "/lib/python" in stripped
            if skipping:
                continue
        elif skipping:
            # Drop only the frame's own continuation - its source line and
            # caret, which are indented under the `File` line. Indentation is
            # the discriminator that matters: the exception that ends a
            # traceback ("KeyError: 'price_band'") sits at column 0, and an
            # earlier version that skipped every line until the next `File`
            # swallowed exactly that line - producing an annotation with a
            # tidy stack and no error in it.
            if line.startswith((" ", "\t")):
                continue
            skipping = False
        kept.append(line)
    return "\n".join(kept)


def first_traceback(output: str) -> str | None:
    """The FAILURES section, which holds the actual assertion and traceback.

    Truncated from the **front**, not the back. An exception's type and message
    are the last lines of a traceback, so keeping the first N characters keeps
    the part that says how the call got there and discards the part that says
    what went wrong - which is how the first version of this produced an
    annotation ending mid-frame, with no exception in it at all.
    """
    match = re.search(
        r"=+ FAILURES =+\n(.*?)(?:\n=+ (?:warnings summary|short test summary|ERRORS))",
        output,
        re.S,
    )
    if not match:
        return None
    body = strip_framework_frames(match.group(1).strip())
    if not body:
        return None
    if len(body) <= MAX_TRACEBACK_CHARS:
        return body
    return "[earlier frames omitted]\n" + body[-MAX_TRACEBACK_CHARS:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="captured pytest output")
    args = parser.parse_args()

    if not args.output.exists():
        print(f"::error::{args.output} not found - pytest produced no output")
        return 0

    text = args.output.read_text(encoding="utf-8", errors="replace")

    # Traceback first: it is the annotation that explains the failure, and the
    # 10-per-step cap means whatever is emitted last is what gets dropped.
    traceback = first_traceback(text)
    if traceback:
        print(f"::error title=pytest traceback::{escape(traceback)}")

    failures = summary_lines(text)
    for line in failures[:MAX_SUMMARY_ANNOTATIONS]:
        print(f"::error::{escape(line)}")
    if len(failures) > MAX_SUMMARY_ANNOTATIONS:
        remaining = len(failures) - MAX_SUMMARY_ANNOTATIONS
        print(f"::error::and {remaining} more failure(s) - see the pytest-output artifact")

    if not traceback and not failures:
        print("::error::pytest failed but produced no recognisable summary")

    return 0


if __name__ == "__main__":
    sys.exit(main())
