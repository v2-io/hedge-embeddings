#!/usr/bin/env python3
r"""
convert_to_tex.py — re-runnable paper.md → paper.tex pipeline for TACL submission.

Design
------
Pipeline (all in-memory; no temp files written that aren't artifacts):

1. Read paper.md.
2. Strip blockquoted intent-note blocks. The paper is in active drafting and
   uses Markdown blockquotes (`> ...`) as working-doc intent notes for
   not-yet-drafted sections. We drop any maximal run of lines starting with
   `>` (along with the contiguous blank lines that separate consecutive `>`
   blocks within a single quote group). The intent is: render only finished
   prose; skip working notes entirely.
3. Split the markdown into:
     - title       (first H1)
     - abstract    (paragraph(s) under `## Abstract`, up to the next `---` or `##`)
     - body        (everything from `## 1.` through end, minus the abstract,
                    references, and supplementary intent-note sections)
   References / Supplementary / Open dependencies sections are skipped for
   now — they are entirely intent-note blockquotes at this stage of drafting.
4. Strip leading numeric prefixes from headings ("## 1. Introduction" →
   "## Introduction"; "### 2.1 Linear features" → "### Linear features").
   LaTeX auto-numbers sections, so retaining the prefix would produce
   "1 1. Introduction".
5. Run pandoc on the cleaned body fragment to produce a LaTeX body.
   Flags chosen:
     -f markdown+smart   (smart quotes / em-dashes / ellipses)
     -t latex
     --wrap=preserve     (don't reflow; keep line structure for diff-friendliness)
     --no-highlight      (no syntax highlighting; we don't have code blocks
                          and don't want pandoc to pull in listings/minted)
     --top-level-division=section
     (no --citeproc; citations stay in `(Author, Year)` form for now —
      will be replaced by \citep / \citet once a .bib file exists.)
6. Wrap the body in a TACL skeleton: documentclass, \usepackage{tacl2021v1},
   anonymous \author block, \maketitle, abstract environment, \input{body},
   \bibliography placeholder, \end{document}.
7. Write paper.tex.
8. Optionally compile with pdflatex (twice for refs); report success/failure.

Idempotence
-----------
- No timestamps in output.
- Pandoc invocation is deterministic.
- Heading-prefix stripping is regex-based and stable.

Known limitations (deliberately out of scope for this drafting pass)
--------------------------------------------------------------------
- Citations remain in `(Author et al., Year)` markdown form. They need to be
  rewritten to `\citep{key}` / `\citet{key}` once a .bib file is built.
  This is the largest pending TODO. Suggested next step: build refs.bib from
  the reference intent-list in paper.md §References, then run a sed/regex
  pass mapping "(Author et al., YYYY)" → \citep{authorYYYY}.
- Tables: paper.md has no tables yet. Pandoc converts pipe tables to LaTeX
  tabular by default; will need iteration when concrete tables land
  (e.g., the per-model × per-dataset results table for §4.2).
- Figures: no \includegraphics calls in paper.md yet. When figure references
  appear (e.g., `figures/fig1_vogel_crossval.pdf`), they should be wrapped
  in figure environments. Add a markdown-level convention (image syntax
  `![caption](figures/foo.pdf)`) and pandoc will convert it correctly.
- Math: pandoc handles `$...$` and `$$...$$` natively. Verified on the
  \hat\rho expressions in §2.3. Display equations should also work.
- Em-dashes / curly quotes / ellipsis: pandoc's `+smart` extension
  handles these.
- The author block is hardcoded to "Anonymous Submission" per the TACL
  double-blind policy; for the camera-ready version, swap to real names
  and add `acceptedWithA` to the tacl2021v1 package options.

Usage
-----
    python3 convert_to_tex.py [--no-compile]

Re-run as often as paper.md changes. Output is paper.tex (committed) and
paper.pdf (gitignored).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PAPER_MD = REPO_ROOT / "paper.md"
PAPER_TEX = REPO_ROOT / "paper.tex"
TEX_DIR = REPO_ROOT / "tex"


# ---------------------------------------------------------------------------
# Step 2: strip blockquoted intent notes
# ---------------------------------------------------------------------------

def strip_intent_blockquotes(md: str) -> str:
    r"""Drop runs of lines starting with `>` (including blank lines that sit
    inside a single multi-paragraph blockquote) from the markdown source.

    A "blockquote group" here is a maximal run of lines that match either:
      - `^>` (a quoted line, possibly empty after the `>`), or
      - `^\s*$` immediately followed (after any other blanks) by another `^>`.
    To keep the implementation simple and predictable, we use a two-pass
    state machine: collect lines, drop any line whose stripped form starts
    with `>`, AND drop trailing blank lines that separate the blockquote
    from the next non-blockquote content (to avoid leaving double blank
    lines that pandoc would render as paragraph breaks)."""
    out_lines: list[str] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(">"):
            # We're entering a blockquote run. Skip every quoted line and any
            # blank lines that are sandwiched between quoted lines (multi-
            # paragraph blockquote). Stop when we hit a non-blank, non-quoted
            # line.
            j = i
            while j < len(lines):
                stripped = lines[j].lstrip()
                if stripped.startswith(">"):
                    j += 1
                    continue
                if stripped == "":
                    # peek ahead: if the *next* non-blank line is another
                    # blockquote, this blank is internal to the group; skip.
                    k = j + 1
                    while k < len(lines) and lines[k].strip() == "":
                        k += 1
                    if k < len(lines) and lines[k].lstrip().startswith(">"):
                        j = k
                        continue
                    # otherwise this blank ends the group
                    break
                break
            i = j
            # ensure we don't leave more than one consecutive blank in output
            if out_lines and out_lines[-1].strip() == "":
                # already ends in blank; consume any leading blanks at i
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
            continue
        out_lines.append(line)
        i += 1
    # collapse 3+ consecutive blank lines down to 1 blank line
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Step 3: split title / abstract / body / drop tail sections
# ---------------------------------------------------------------------------

# Sections we drop entirely from the LaTeX body. These are currently 100%
# intent-note blockquote at this drafting stage; they'll be re-included
# section-by-section as prose lands. The `## References` block in particular
# becomes the .bib file later; the conversion pipeline shouldn't try to
# render it as prose.
DROP_SECTIONS = {"References", "Supplementary", "Open dependencies"}


def split_paper(md: str) -> tuple[str, str, str]:
    """Return (title, abstract_md, body_md).

    - title: the H1 line (without the leading `# `).
    - abstract_md: the paragraph(s) immediately under `## Abstract`,
      up to the next horizontal rule or H2.
    - body_md: everything from the first numbered H2 (`## 1. ...`) through
      end of document, with the dropped sections removed.
    """
    # Title
    m = re.search(r"(?m)^# (.+)$", md)
    if not m:
        sys.exit("ERROR: no H1 title found in paper.md")
    title = m.group(1).strip()

    # Abstract (between `## Abstract` and the next `---` or `## `)
    m_abs = re.search(
        r"(?ms)^## Abstract\s*\n(.+?)(?=^---\s*$|^## )", md
    )
    if not m_abs:
        sys.exit("ERROR: no `## Abstract` section found in paper.md")
    abstract = m_abs.group(1).strip()

    # Body: from the first `## N. Title` onward
    m_body = re.search(r"(?ms)^## \d+\..*", md)
    if not m_body:
        sys.exit("ERROR: no numbered `## N. ...` heading found in paper.md")
    body = md[m_body.start():]

    # Drop sections by name. We split on H2 boundaries; a "section" runs
    # from one H2 to the next (or EOF). If the H2 title (after stripping
    # any leading "N. " prefix) is in DROP_SECTIONS, drop the whole block.
    sections = re.split(r"(?m)(^## .*$)", body)
    # sections is a list like: ['', heading1, content1, heading2, content2, ...]
    out_parts: list[str] = []
    if sections and sections[0].strip() == "":
        sections = sections[1:]
    for i in range(0, len(sections), 2):
        if i + 1 >= len(sections):
            break
        heading = sections[i]
        content = sections[i + 1]
        # Strip leading "## " and any "N." or "N.M" number prefix to identify
        # the section by its plain title.
        head_text = re.sub(r"^##\s+", "", heading)
        head_text = re.sub(r"^\d+(\.\d+)?\.?\s*", "", head_text).strip()
        if head_text in DROP_SECTIONS:
            continue
        out_parts.append(heading)
        out_parts.append(content)
    body = "\n".join(out_parts).strip() + "\n"

    return title, abstract, body


# ---------------------------------------------------------------------------
# Step 4: strip leading numeric prefixes from headings
# ---------------------------------------------------------------------------

def strip_heading_numbers(md: str) -> str:
    """`## 1. Introduction` → `## Introduction`,
       `### 2.1 Linear features` → `### Linear features`.
    LaTeX auto-numbers sections; keeping the prefix would double-number."""
    md = re.sub(r"(?m)^(##) \d+\.\s+", r"\1 ", md)
    md = re.sub(r"(?m)^(###) \d+\.\d+\s+", r"\1 ", md)
    return md


# ---------------------------------------------------------------------------
# Step 5: pandoc invocation
# ---------------------------------------------------------------------------

def pandoc_to_latex(md: str, shift_headings: int = 0) -> str:
    """Run pandoc on a markdown fragment, return LaTeX body.

    `shift_headings`: passed to --shift-heading-level-by. The body markdown
    uses `##` for top-level sections, but pandoc treats `##` as level-2 (=
    \\subsection by default). We shift by -1 so that `##` → `\\section`,
    `###` → `\\subsection`, and `####` → `\\subsubsection`.
    """
    if shutil.which("pandoc") is None:
        sys.exit("ERROR: pandoc not found on PATH. brew install pandoc.")
    cmd = [
        "pandoc",
        "-f", "markdown+smart",
        "-t", "latex",
        "--wrap=preserve",
        "--no-highlight",
        "--top-level-division=section",
    ]
    if shift_headings:
        cmd.append(f"--shift-heading-level-by={shift_headings}")
    proc = subprocess.run(
        cmd, input=md, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"ERROR: pandoc failed:\n{proc.stderr}")
    return _fix_longtable_for_two_column(proc.stdout)


def _fix_longtable_for_two_column(latex: str) -> str:
    """Convert pandoc's longtable environments to plain tabular wrapped in
    table+centering, because TACL is two-column and longtable requires
    one-column mode.  Idempotent: a second pass is a no-op.

    Pandoc emits roughly:
        \\begin{longtable}[]{@{}<colspec>@{}}
        \\toprule\\noalign{}
        ... header ...
        \\midrule\\noalign{}
        \\endhead
        \\bottomrule\\noalign{}
        \\endlastfoot
        ... rows ...
        \\end{longtable}

    We rewrite to:
        \\begin{table}[t]
        \\centering
        \\begin{tabular}{<colspec>}
        \\toprule
        ... header ...
        \\midrule
        ... rows ...
        \\bottomrule
        \\end{tabular}
        \\end{table}
    """
    import re
    pattern = re.compile(
        r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}",
        re.DOTALL,
    )

    def replace_one(match: "re.Match[str]") -> str:
        colspec = match.group(1)
        body = match.group(2)
        # Strip longtable-specific markers that are illegal in tabular.
        body = body.replace(r"\noalign{}", "")
        body = body.replace(r"\endhead", "")
        body = body.replace(r"\endlastfoot", "")
        body = body.replace(r"\endfirsthead", "")
        body = body.replace(r"\endfoot", "")
        # Drop the in-table caption marker; we let the user add one outside if desired.
        body = re.sub(r"\\caption\{.*?\}\\\\\n", "", body, flags=re.DOTALL)
        # Collapse 3+ blank lines to a single one for readability.
        body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")
        return (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\begin{tabular}{" + colspec + "}\n"
            + body + "\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

    return pattern.sub(replace_one, latex)


# ---------------------------------------------------------------------------
# Step 6: TACL template wrapper
# ---------------------------------------------------------------------------

PREAMBLE = r"""% paper.tex — generated by convert_to_tex.py from paper.md.
% DO NOT EDIT THIS FILE BY HAND. Edit paper.md and re-run convert_to_tex.py.
%
% TACL submission, double-blind. Style files in tex/ (tacl2021v1.sty,
% acl_natbib.bst). Compile with: lualatex paper.tex (twice for refs).
% The convert_to_tex.py driver invokes lualatex by default; override with
% the LATEX_ENGINE env var.

\documentclass[11pt,a4paper]{article}
\usepackage{latexsym}
\usepackage{url}
\usepackage{graphicx}
\usepackage{array}
\usepackage{calc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath,amssymb}

% Lualatex-native font setup (replaces pdflatex's \usepackage{times} +
% T1 fontenc + utf8 inputenc).  TeX Gyre Termes is the OpenType
% Times-equivalent.  CJK fallback registers macOS-default fonts for
% Japanese/Korean/Chinese/Arabic/Devanagari glyphs that appear in §6 L2
% so they render rather than disappearing as missing-glyph holes.
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
% TODO: CJK fallback for Japanese chars in §6 L2 (ありそうな, まさか).
% Attempted via luaotfload.add_fallback + RawFeature on TeX Gyre Termes,
% but ran into luaotfload font-resolution issues we did not resolve in
% this session.  For now those glyphs render as missing-glyph boxes;
% see paper.md §6 L2 for the affected text.  At camera-ready, either
% (a) get the fallback working with a minimal known-good font list,
% (b) switch to xelatex which handles CJK fallback more straightforwardly,
% or (c) romanize the Japanese examples (lossy but lossless w.r.t. the
% scientific point being made).

% Lualatex handles UTF-8 math/Greek glyphs (ρ Δ ≤ ≥ → ↔ ×) natively,
% so the pdflatex-era \newunicodechar declarations are no longer needed.
% Keeping the legacy block as a no-op fallback for the (unlikely) case
% that someone runs this through pdflatex via the LATEX_ENGINE override.
\usepackage{newunicodechar}

% Unicode characters that appear in the markdown source. pdflatex doesn't
% have native UTF-8 support for math/Greek glyphs; we map them explicitly.
% If a new Unicode char appears in paper.md and breaks compilation, add a
% \newunicodechar line for it here.
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{²}{\ensuremath{^{2}}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{ε}{\ensuremath{\epsilon}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{η}{\ensuremath{\eta}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{Ω}{\ensuremath{\Omega}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{Π}{\ensuremath{\Pi}}
\newunicodechar{Θ}{\ensuremath{\Theta}}
% Section sign (§) is already in T1 fonts.

% Pandoc emits \tightlist for compact lists; define it as a no-op so we
% don't have to enable pandoc's full default preamble.
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

% Pandoc emits \pandocbounded around figures with size constraints.
% The default (identity) lets figures render at native size, which
% overflows the column boundary in TACL's two-column layout.  Redefine
% to scale down figures wider than \linewidth while leaving smaller
% figures untouched.
\providecommand{\pandocbounded}[1]{%
  \begingroup
  \sbox0{#1}%
  \ifdim\wd0>\linewidth
    \resizebox{\linewidth}{!}{\usebox0}%
  \else
    \usebox0%
  \fi
  \endgroup
}

% TACL style file lives in tex/. Tell LaTeX to look there.
\makeatletter
\def\input@path{{tex/}}
\makeatother
\usepackage[]{tacl2021v1}

% Figures live in figures/.
\graphicspath{{figures/}}

\title{__TITLE__}

\author{Anonymous Submission \\
  \texttt{anonymized@example.com}}

\date{}

\begin{document}
\maketitle

\begin{abstract}
__ABSTRACT__
\end{abstract}

__BODY__

% References: TACL uses acl_natbib. A refs.bib file is not yet built; the
% draft still has citations in (Author et al., Year) markdown form. When
% refs.bib lands, uncomment the two lines below and re-run.
%
% \bibliography{refs}
% \bibliographystyle{tex/acl_natbib}

\end{document}
"""


def assemble_tex(title: str, abstract_tex: str, body_tex: str) -> str:
    return (
        PREAMBLE
        .replace("__TITLE__", title)
        .replace("__ABSTRACT__", abstract_tex.strip())
        .replace("__BODY__", body_tex.strip())
    )


# ---------------------------------------------------------------------------
# Step 8: LaTeX compile
# ---------------------------------------------------------------------------
#
# Engine: lualatex by default.  Lualatex handles UTF-8 natively (CJK
# characters, math symbols, em-dashes etc. all pass through without
# the \newunicodechar mappings pdflatex requires) and supports OpenType
# fonts via fontspec.  The TACL .sty file is engine-agnostic at quick
# inspection (no \pdfoutput / \pdftex / \ifpdf branches), so lualatex
# should compile against the standard template.  Override via the
# LATEX_ENGINE env var if needed (e.g. LATEX_ENGINE=pdflatex).

import os

def try_compile(tex_path: Path) -> tuple[bool, str]:
    """Run the configured LaTeX engine twice. Return (ok, last_log_tail)."""
    engine = os.environ.get("LATEX_ENGINE", "lualatex")
    if shutil.which(engine) is None:
        return False, f"{engine} not on PATH (brew install --cask mactex or basictex)"
    log_tail = ""
    for pass_num in (1, 2):
        proc = subprocess.run(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(tex_path.name),
            ],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        log_tail = proc.stdout[-2000:] if proc.stdout else proc.stderr[-2000:]
        if proc.returncode != 0:
            return False, log_tail
    return True, log_tail


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip the pdflatex pass (just emit paper.tex).",
    )
    args = p.parse_args(argv)

    if not PAPER_MD.exists():
        sys.exit(f"ERROR: {PAPER_MD} not found.")
    if not (TEX_DIR / "tacl2021v1.sty").exists():
        sys.exit(
            f"ERROR: {TEX_DIR}/tacl2021v1.sty not found. "
            "Re-run the template-fetch step (see README of this script)."
        )

    md = PAPER_MD.read_text(encoding="utf-8")
    md = strip_intent_blockquotes(md)
    title, abstract_md, body_md = split_paper(md)
    body_md = strip_heading_numbers(body_md)

    # Sanitize standalone `---` (markdown horizontal rule). Pandoc renders
    # these in LaTeX as a centered rule which looks ugly between sections.
    # Drop them; the section commands provide the visual breaks.
    body_md = re.sub(r"(?m)^---\s*$", "", body_md)
    body_md = re.sub(r"\n{3,}", "\n\n", body_md)

    abstract_tex = pandoc_to_latex(abstract_md)
    body_tex = pandoc_to_latex(body_md, shift_headings=-1)

    tex = assemble_tex(title, abstract_tex, body_tex)
    PAPER_TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER_TEX} ({len(tex)} bytes, {tex.count(chr(10))} lines)")

    if args.no_compile:
        return 0

    engine = os.environ.get("LATEX_ENGINE", "lualatex")
    ok, log_tail = try_compile(PAPER_TEX)
    if ok:
        print(f"{engine}: OK (paper.pdf built)")
    else:
        print(f"{engine}: FAILED")
        print("--- last 2000 chars of log ---")
        print(log_tail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
