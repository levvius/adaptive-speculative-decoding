#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix stale front matter in the thesis: РЕФЕРАТ counts + TOC page numbers.

The document was expanded but its front matter was never refreshed: the abstract
(РЕФЕРАТ) still claims an old page/figure/table count, and 58 of 63 static
table-of-contents page numbers lag the real layout by +1…+8 pages. This script
renders the *current* ``.docx`` with LibreOffice to learn the real page of each
heading, then patches in place:

  * РЕФЕРАТ run — pages → actual total, figures → 6, tables → 9
    (sources = 19, appendices = 4, chapters = 4 are already correct);
  * each TOC entry's page-number run → the heading's actual rendered page.

Only digit text changes. The clickable TOC hyperlinks/bookmarks added by
``papers/add_toc_links.py`` (which wrap the *title* run, not the page run) and
every word of body content are left untouched. **No experimental value is
modified.** Counts/pages are recomputed from the render, so re-running is
idempotent.

Caveat: pages come from the LibreOffice rendering of this file; confirm against
Word on finalize (the front section matches exactly, so they should agree).

Usage:  python3 papers/fix_frontmatter.py [--docx PATH] [--dry-run]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

PARA = re.compile(r"<w:p\b.*?</w:p>", re.S)
WT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)

REFERAT_MARK = "Выпускная квалификационная работа:"
FIG_COUNT, TAB_COUNT = 6, 9   # verified from captions (Рисунок/Таблица N.M)
FRONT_SKIP = 6                # skip title/abstract/TOC pages when locating headings


def runs_text(s):
    return "".join(WT.findall(s))


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def toc_title(p):
    return norm(runs_text(p.split("<w:tab/>", 1)[0]))


def _replace_wt_inner(fragment, where, transform):
    """Rewrite the inner text of <w:t> elements matching predicate ``where``."""
    def f(m):
        full, inner = m.group(0), m.group(1)
        if not where(inner):
            return full
        open_tag = full[: full.index(">") + 1]
        return open_tag + transform(inner) + "</w:t>"
    return WT.sub(f, fragment)


def patch_referat(p, total):
    def transform(s):
        s = re.sub(r"\d+(?=\s*с\.)", str(total), s, count=1)
        s = re.sub(r"\d+(?=\s*рис\.)", str(FIG_COUNT), s, count=1)
        s = re.sub(r"\d+(?=\s*табл\.)", str(TAB_COUNT), s, count=1)
        return s
    return _replace_wt_inner(p, lambda inner: REFERAT_MARK in inner, transform)


def patch_toc_page(p, page):
    """Set the page-number run (the <w:t> right after <w:tab/>) to ``page``."""
    i = p.find("<w:tab/>")
    if i == -1:
        return p, False
    m = WT.search(p, i)
    if not m:
        return p, False
    open_tag = m.group(0)[: m.group(0).index(">") + 1]
    new = p[: m.start()] + open_tag + str(page) + "</w:t>" + p[m.end():]
    return new, (m.group(1) != str(page))


def render_pages(docx):
    """Return (list-of-normalised-page-text, total_pages) via LibreOffice+poppler."""
    tmp = tempfile.mkdtemp(prefix="fm_")
    try:
        subprocess.run(
            ["soffice", "--headless", "-env:UserInstallation=file:///tmp/lo_fm",
             "--convert-to", "pdf", "--outdir", tmp, docx],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pdf = os.path.join(tmp, os.path.splitext(os.path.basename(docx))[0] + ".pdf")
        txt = os.path.join(tmp, "t.txt")
        subprocess.run(["pdftotext", "-layout", pdf, txt], check=True)
        pages = open(txt, encoding="utf-8").read().split("\f")
        npages = [norm(re.sub(r"\.{2,}", "", pg)).casefold() for pg in pages]
        total = len(pages)
        try:
            out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, check=True).stdout
            mm = re.search(r"Pages:\s+(\d+)", out)
            if mm:
                total = int(mm.group(1))
        except Exception:
            pass
        return npages, total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def actual_page(npages, title):
    key = norm(title).casefold()[:46]
    for i in range(FRONT_SKIP, len(npages)):
        if key and key in npages[i]:
            return i + 1
    return None


def process(xml, npages, total):
    edits, changed_toc, missing, ref_done = {}, 0, [], False
    for m in PARA.finditer(xml):
        off, p = m.start(), m.group(0)
        if REFERAT_MARK in runs_text(p):
            edits[off] = patch_referat(p, total)
            ref_done = True
        elif 'w:leader="dot"' in p and toc_title(p):
            title = toc_title(p)
            ap = actual_page(npages, title)
            if ap is None:
                missing.append(title)
                continue
            new, did = patch_toc_page(p, ap)
            if did:
                changed_toc += 1
            edits[off] = new
    out, pos = [], 0
    for m in PARA.finditer(xml):
        out.append(xml[pos:m.start()])
        out.append(edits.get(m.start(), m.group(0)))
        pos = m.end()
    out.append(xml[pos:])
    return "".join(out), ref_done, changed_toc, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docx", default=os.path.join(os.path.dirname(__file__), "Козин_newera.docx"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.docx):
        sys.exit("docx not found: %s" % args.docx)

    print("rendering current docx to learn real page layout…")
    npages, total = render_pages(args.docx)
    print("rendered total pages: %d" % total)

    with zipfile.ZipFile(args.docx) as z:
        names = z.namelist()
        doc = z.read("word/document.xml").decode("utf-8")
        blobs = {n: z.read(n) for n in names}

    new_doc, ref_done, changed_toc, missing = process(doc, npages, total)
    print("РЕФЕРАТ updated: %s -> %d с., %d рис., %d табл." % (ref_done, total, FIG_COUNT, TAB_COUNT))
    print("TOC page numbers changed: %d" % changed_toc)
    for t in missing:
        print("  UNLOCATED (left as-is):", t)

    if args.dry_run:
        print("[dry-run] nothing written.")
        return
    if not ref_done and changed_toc == 0:
        print("nothing to change.")
        return

    bak = args.docx + ".frontmatter.bak"
    if not os.path.exists(bak):
        shutil.copy2(args.docx, bak)
        print("backup ->", bak)
    blobs["word/document.xml"] = new_doc.encode("utf-8")
    tmp = args.docx + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    os.replace(tmp, args.docx)
    print("wrote", args.docx)


if __name__ == "__main__":
    main()
