"""Shim and hidden-failure guard for the GBLK implementation.

Enforces that the three ``geopfa/prob/gblk_*.py`` modules and
``geopfa/spatial_lkx.py`` never acquire shim, xfail, skip, TODO, or silent
fallback behavior.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SCOPE_FILES: tuple[str, ...] = (
    "geopfa/prob/gblk_assemble.py",
    "geopfa/prob/gblk_backend.py",
    "geopfa/prob/gblk_runner.py",
    "geopfa/spatial_lkx.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (_REPO_ROOT / path).read_text(encoding="utf-8")


def _strip_inline_comment(line: str) -> str:
    """Return ``line`` with any trailing ``#`` comment removed.

    Respects single/double-quoted string literals so ``"#"`` inside a
    string is not treated as a comment start.
    """
    if "#" not in line:
        return line
    in_str = False
    quote = ""
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                in_str = False
        elif ch in {"'", '"'}:
            in_str = True
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _strip_docstrings_and_comments(source: str) -> str:
    """Return ``source`` with docstrings and comments removed.

    Docstrings are legitimate places to describe an intentional
    fallback contract or reference historic bugs; the guard rules must
    only apply to executable code.
    """
    tree = ast.parse(source)
    docstring_spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_spans.append(
                    (first.lineno, first.end_lineno or first.lineno)
                )

    lines = source.splitlines()
    for start, end in docstring_spans:
        for i in range(start - 1, end):
            lines[i] = ""
    return "\n".join(_strip_inline_comment(line) for line in lines)


@pytest.mark.parametrize("relpath", _SCOPE_FILES)
def test_no_todo_or_hack_markers(relpath: str) -> None:
    """No TODO / FIXME / XXX / HACK markers in scope files.

    Notes
    -----
    Docstrings and comments are stripped before the check so that a
    documentation reference to a historic FIXME does not falsely trip
    the guard; markers surviving the strip are executable-code markers.
    """
    source = _read(relpath)
    executable = _strip_docstrings_and_comments(source)
    pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    hits = pattern.findall(executable)
    assert not hits, (
        f"{relpath}: shim/marker patterns leaked into executable code: {hits}"
    )


@pytest.mark.parametrize("relpath", _SCOPE_FILES)
def test_no_bare_or_broad_except(relpath: str) -> None:
    """Reject `except:` and `except Exception:` in scope files.

    Broad exception handlers silently swallow programming errors and
    are the classic shim pattern documented in RR-010 / RR-011.
    """
    tree = ast.parse(_read(relpath))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc = node.type
        if exc is None:
            offenders.append((node.lineno, "bare except:"))
            continue
        if isinstance(exc, ast.Name) and exc.id in {
            "Exception",
            "BaseException",
        }:
            offenders.append((node.lineno, f"except {exc.id}:"))
    assert not offenders, f"{relpath}: broad except handlers: {offenders}"


@pytest.mark.parametrize("relpath", _SCOPE_FILES)
def test_no_not_implemented_placeholders(relpath: str) -> None:
    """Reject `raise NotImplementedError` placeholders in scope files."""
    tree = ast.parse(_read(relpath))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        if (isinstance(exc, ast.Name) and exc.id == "NotImplementedError") or (
            isinstance(exc, ast.Attribute)
            and exc.attr == "NotImplementedError"
        ):
            offenders.append(node.lineno)
    assert not offenders, (
        f"{relpath}: NotImplementedError placeholders at lines {offenders}"
    )


@pytest.mark.parametrize("relpath", _SCOPE_FILES)
def test_no_pytest_skip_or_xfail(relpath: str) -> None:
    """Reject pytest.skip / pytest.xfail / skip markers in scope files.

    These belong exclusively to test files, never to library code.
    """
    executable = _strip_docstrings_and_comments(_read(relpath))
    forbidden = (
        "pytest.skip",
        "pytest.xfail",
        "pytest.mark.skip",
        "pytest.mark.xfail",
    )
    hits = [tok for tok in forbidden if tok in executable]
    assert not hits, f"{relpath}: forbidden pytest gate tokens: {hits}"


@pytest.mark.parametrize("relpath", _SCOPE_FILES)
def test_no_blanket_warning_filters(relpath: str) -> None:
    """Reject `warnings.filterwarnings("ignore")` in scope files.

    Blanket ``ignore`` filters mask upstream regressions in
    ``latticekrigx`` / ``scipy.sparse`` that this codebase must surface,
    not swallow.
    """
    executable = _strip_docstrings_and_comments(_read(relpath))
    pattern = re.compile(r"filterwarnings\(\s*['\"]ignore['\"]\s*(?:\)|,)")
    hits = pattern.findall(executable)
    assert not hits, f"{relpath}: blanket 'ignore' warning filters: {hits}"


def test_spatial_lkx_defines_no_fallback_functions() -> None:
    """A failed spatial fit must not switch silently to another model."""
    source = _read("geopfa/spatial_lkx.py")
    tree = ast.parse(source)
    fallback_funcs = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and "fallback" in node.name.lower()
    ]
    assert fallback_funcs == []
