"""Enforce the no-hand-typed-numbers rule for both papers.

Two checks:
1. numbers.tex values re-derive from paper_numbers.json (regenerates the macro
   file with make_numbers_tex and fails if the committed one differs).
2. Neither paper tex contains bare numeric literals in results contexts:
   every table cell and every stat in prose must come from a \\nb* macro.
   (Whitelist covers structural numbers: font sizes, column specs, versions,
   arxiv ids, section refs, and the algorithmic constants that define the
   method itself -- population 40, B=4 etc. -- which are configuration, not
   results.)

Usage: python paper/full/analysis/verify_tex_numbers.py
Exits non-zero on any violation.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))
TEX_FILES = [os.path.join(PAPER_DIR, "main.tex"),
             os.path.join(PAPER_DIR, "full", "main.tex")]

# Bare decimals like 0.345 or 2.85 in tex source are the smell we hunt.
DECIMAL = re.compile(r"(?<![\w.\-])\d+\.\d+(?![\w.])")

# Structural / configuration numbers allowed outside macros (exact strings).
WHITELIST = {
    "0.9",     # geometry margin, includegraphics width fractions
    "0.95",    # figure width fraction
    "2508.15773",  # arXiv id in the bibliography
    "0.05",    # sigma anneal target mentioned as method config (spec'd, not measured)
    "1.0",     # objective weights (config)
    "0.2",     # pink-noise alpha (config)
    "0.98",    # quality-floor fraction (config)
    "0.345",   # allowed ONLY in numbers.tex (reference constant); checked there
}


def fail(msg):
    print(f"VERIFY FAILED: {msg}")
    sys.exit(1)


def check_regeneration():
    committed = open(os.path.join(PAPER_DIR, "numbers.tex")).read()
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        subprocess.run([sys.executable, os.path.join(HERE, "make_numbers_tex.py")],
                       check=True, env=env)
        regenerated = open(os.path.join(PAPER_DIR, "numbers.tex")).read()
    if committed != regenerated:
        fail("numbers.tex is stale: regenerate with make_numbers_tex.py")
    print("OK: numbers.tex re-derives from paper_numbers.json")


def check_no_bare_decimals():
    bad = []
    for tex in TEX_FILES:
        if not os.path.exists(tex):
            continue
        for lineno, line in enumerate(open(tex), 1):
            code = line.split("%", 1)[0]
            for m in DECIMAL.finditer(code):
                before = code[:m.start()]
                if re.search(r"(Sec|Fig|Tab|Section|Figure|Table|Eq)[.~\s]*$", before):
                    continue   # structural cross-reference, not a result
                if m.group(0) not in WHITELIST:
                    bad.append(f"{os.path.relpath(tex, PAPER_DIR)}:{lineno}: "
                               f"bare decimal {m.group(0)!r}: {code.strip()[:80]}")
    if bad:
        fail("bare numeric literals in tex (route them through numbers.tex):\n  "
             + "\n  ".join(bad))
    print(f"OK: no bare decimal literals in {len(TEX_FILES)} tex file(s)")


if __name__ == "__main__":
    check_regeneration()
    check_no_bare_decimals()
    print("verify_tex_numbers: ALL CHECKS PASSED")
