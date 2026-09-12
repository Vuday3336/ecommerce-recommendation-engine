"""The CI failure annotator must not lose the exception.

This exists because both of its early versions silently discarded the one line
that explains a failure - once by truncating the traceback from the front
(an exception is the *last* line), and once by skipping every line after a
framework frame (the exception sits at column 0, after the last frame).

Both produced a plausible-looking annotation with no error in it, which is
worse than no annotation at all: a red build that appears to explain itself but
does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotate_pytest_failures import (  # noqa: E402
    escape,
    first_traceback,
    strip_framework_frames,
    summary_lines,
)

PYTEST_OUTPUT = """\
=================================== FAILURES ===================================
_ TestSurfaces.test_returns_200[/home/42] _

    def test_returns_200(self, client, path):
>       assert client.get(path).status_code == 200
E       AssertionError: assert 500 == 200

backend/tests/test_api.py:61: AssertionError
----------------------------- Captured stdout call -----------------------------
ERROR    app.main:main.py:194 unhandled error on GET /home/42
Traceback (most recent call last):
  File "/repo/backend/app/main.py", line 190, in observability
    response = await call_next(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/python3.12/site-packages/starlette/middleware/base.py", line 168, in call_next
    raise app_exc from app_exc.__cause__
  File "/repo/ml/recsys/inference/engine.py", line 412, in homepage
    rails = self._build_rails(user_id)
KeyError: 'price_band'
=============================== warnings summary ===============================
some warning
=========================== short test summary info ============================
FAILED backend/tests/test_api.py::TestSurfaces::test_returns_200[/home/42] - AssertionError: assert 500 == 200
ERROR backend/tests/test_other.py::test_broken
1 failed, 2 passed
"""


class TestTracebackExtraction:
    def test_the_exception_is_kept(self):
        """The whole point. An annotation without the error explains nothing."""
        body = first_traceback(PYTEST_OUTPUT)
        assert body is not None
        assert "KeyError: 'price_band'" in body

    def test_application_frames_are_kept(self):
        body = first_traceback(PYTEST_OUTPUT)
        assert "recsys/inference/engine.py" in body
        assert "backend/app/main.py" in body

    def test_framework_frames_are_dropped(self):
        """Middleware frames are what push the exception past the size limit."""
        body = first_traceback(PYTEST_OUTPUT)
        assert "site-packages" not in body

    def test_truncation_keeps_the_end_not_the_beginning(self):
        """An exception is the last line, so the front is what may be dropped."""
        import annotate_pytest_failures as module

        original = module.MAX_TRACEBACK_CHARS
        module.MAX_TRACEBACK_CHARS = 120
        try:
            body = first_traceback(PYTEST_OUTPUT)
        finally:
            module.MAX_TRACEBACK_CHARS = original

        assert body.startswith("[earlier frames omitted]")
        assert "KeyError: 'price_band'" in body

    def test_a_frames_source_line_does_not_survive_its_frame(self):
        stripped = strip_framework_frames(
            '  File "/opt/site-packages/x.py", line 1, in f\n'
            "    raise SomethingInternal()\n"
            "ValueError: the real error\n"
        )
        assert "SomethingInternal" not in stripped
        assert "ValueError: the real error" in stripped


class TestSummaryLines:
    def test_only_result_lines_are_reported(self):
        """Captured log records start with ERROR too, and are not results.

        Seven of them once filled the 10-annotation budget and evicted the
        traceback.
        """
        lines = summary_lines(PYTEST_OUTPUT)
        assert len(lines) == 2
        assert all("::" in line for line in lines)
        assert not any("unhandled error on GET" in line for line in lines)

    def test_no_summary_section_yields_nothing(self):
        assert summary_lines("no sections here at all") == []


class TestEscaping:
    def test_newlines_become_workflow_escapes(self):
        assert escape("a\nb") == "a%0Ab"

    def test_percent_is_escaped_before_newlines_not_after(self):
        """Otherwise the %0A sequences get double-encoded into %250A."""
        assert escape("100%\nnext") == "100%25%0Anext"

    def test_carriage_returns_are_removed(self):
        assert "\r" not in escape("a\r\nb")
