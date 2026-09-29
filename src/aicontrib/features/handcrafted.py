"""Hand-made features of a piece of code, for the second stage (aicontrib.model.stage2).

The embedding reads the code as tokens and misses some traits that set AI-written code apart: how long
and wordy its comments are, trailing whitespace (the tokenizer drops it), typographic symbols (em dashes,
arrows, curly quotes), TODOs (mostly human), defensive null checks. Each feature is one number per text;
FEATURES maps its name to its function. They were chosen by measuring 41 candidates on the agent-commit
data (README, "Second stage").

Comments are found with a small per-line scanner that skips string literals, so "//" in a URL isn't a
comment: `#` and triple-quoted docstrings for Python, `//` and `/* */` for the C-like languages.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

import numpy as np

TODO = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
# Em/en dashes, arrows, bullets, ellipsis, curly quotes, check marks, math signs.
TYPOGRAPHIC = re.compile("[—–→←↔⇒•…“”‘’✓✗"
                         "≈≤≥×]")
NULL_CHECK = re.compile(r"(===?|!==?)\s*(null|undefined|nullptr)\b|\bis (not )?None\b|\?\.|\?\?|\bnullptr\b|"
                        r"ArgumentNullException|ThrowIfNull")
ANY = re.compile(r":\s*any\b|as any\b|<any>")
LOG_CALL = re.compile(r"\b(console\.(log|warn|error|info|debug)|logger\.|logging\.|log\.(info|warn|error|debug)|"
                      r"print\(|Console\.Write|Debug\.Log|std::cout|std::cerr|printf\(|_logger\.|LOG\()")
TOKEN = re.compile(r"\w+|[^\w\s]")


class Parsed:
    """A text split into lines, each with its code part and its comment part."""

    def __init__(self, code: str, language: str | None):
        self.lines = code.split("\n")
        self.code_parts, self.comments = _split_code_comments(self.lines, language)
        self.nonblank = [i for i, line in enumerate(self.lines) if line.strip()]
        self.code_lines = [self.code_parts[i] for i in self.nonblank if self.code_parts[i].strip()]
        self.comment_texts = [self.comments[i] for i in self.nonblank if self.comments[i]]
        self.code_text = "\n".join(self.code_parts)
        self.raw = code


def _split_code_comments(lines: list[str], language: str | None) -> tuple[list[str], list[str]]:
    python = language == "Python"
    code_parts, comment_parts = [], []
    in_block = None  # the delimiter closing an open block comment / docstring
    for line in lines:
        code_buf, com_buf = [], []
        i, n = 0, len(line)
        quote = None
        while i < n:
            if in_block:
                end = line.find(in_block, i)
                com_buf.append(line[i:] if end < 0 else line[i:end])
                i, in_block = (n, in_block) if end < 0 else (end + len(in_block), None)
                continue
            ch = line[i]
            if quote:
                if ch == "\\":
                    code_buf.append(line[i:i + 2])
                    i += 2
                elif line.startswith(quote, i):
                    code_buf.append(quote)
                    i += len(quote)
                    quote = None
                else:
                    code_buf.append(ch)
                    i += 1
                continue
            if python:
                if line.startswith(('"""', "'''"), i):
                    delim = line[i:i + 3]
                    if "".join(code_buf).strip():  # not first on the line: a multi-line string, i.e. code
                        quote = delim
                        code_buf.append(delim)
                        i += 3
                        continue
                    end = line.find(delim, i + 3)  # a docstring
                    com_buf.append(line[i + 3:] if end < 0 else line[i + 3:end])
                    i, in_block = (n, delim) if end < 0 else (end + 3, None)
                    continue
                if ch == "#":
                    com_buf.append(line[i + 1:])
                    break
            else:
                if line.startswith("//", i):
                    com_buf.append(line[i + 2:].lstrip("/"))
                    break
                if line.startswith("/*", i):
                    end = line.find("*/", i + 2)
                    com_buf.append(line[i + 2:] if end < 0 else line[i + 2:end])
                    i, in_block = (n, "*/") if end < 0 else (end + 2, None)
                    continue
            if ch in "\"'" or (ch == "`" and not python):
                quote = ch
            code_buf.append(ch)
            i += 1
        code_parts.append("".join(code_buf))
        comment_parts.append(" ".join(c.strip(" *") for c in com_buf).strip())
    return code_parts, comment_parts


def _per_nonblank(p: Parsed, n: float) -> float:
    return n / max(1, len(p.nonblank))


def _per_code_line(p: Parsed, n: float) -> float:
    return n / max(1, len(p.code_lines))


def _max_indent_depth(p: Parsed) -> float:
    # Code lines only: the " *" lines of block comments would make one space look like an indent level.
    indents = [len(p.lines[i]) - len(p.lines[i].lstrip(" \t")) for i in p.nonblank if p.code_parts[i].strip()]
    unit = min([x for x in indents if x > 0], default=1)
    return math.log1p(max(indents, default=0) / unit)


def _token_entropy(p: Parsed) -> float:
    counts = Counter(TOKEN.findall(p.code_text))
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values()) if total else 0.0


FEATURES: dict[str, Callable[[Parsed], float]] = {
    # Comments: AI writes more, longer, full-sentence comments; humans leave TODOs.
    "mean_comment_words": lambda p: float(np.mean([len(c.split()) for c in p.comment_texts])) if p.comment_texts else 0.0,
    "comment_char_ratio": lambda p: sum(len(c) for c in p.comment_texts) / max(1, len(p.raw)),
    "comment_line_ratio": lambda p: _per_nonblank(p, sum(1 for i in p.nonblank if p.comments[i] and not p.code_parts[i].strip())),
    "todo_count": lambda p: math.log1p(len(TODO.findall(p.raw))),
    # Formatting: humans' editors leave trailing whitespace, which the embedding's tokenizer can't see.
    "trailing_ws_ratio": lambda p: _per_nonblank(p, sum(1 for i in p.nonblank if p.lines[i] != p.lines[i].rstrip())),
    "blank_line_ratio": lambda p: 1 - len(p.nonblank) / max(1, len(p.lines)),
    # Characters a keyboard doesn't type easily.
    "typographic_symbols": lambda p: math.log1p(len(TYPOGRAPHIC.findall(p.raw))),
    "non_ascii_ratio": lambda p: sum(ch > "\x7f" for ch in p.raw) / max(1, len(p.raw)),
    # Defensive and typed code.
    "null_check_density": lambda p: _per_code_line(p, len(NULL_CHECK.findall(p.code_text))),
    "any_density": lambda p: _per_code_line(p, len(ANY.findall(p.code_text))),
    "log_call_density": lambda p: _per_code_line(p, len(LOG_CALL.findall(p.code_text))),
    # Shape.
    "max_indent_depth": _max_indent_depth,
    "token_entropy": _token_entropy,
}


def feature_matrix(codes: list[str], languages: list[str | None], names: list[str]) -> np.ndarray:
    """One row per text, one column per feature name. languages: canonical names (e.g. "Python")."""
    unknown = [n for n in names if n not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown stage2 features {unknown}. Available: {', '.join(FEATURES)}")
    out = np.zeros((len(codes), len(names)), dtype=np.float64)
    for i, (code, language) in enumerate(zip(codes, languages)):
        parsed = Parsed(code, language)
        out[i] = [FEATURES[n](parsed) for n in names]
    return out
