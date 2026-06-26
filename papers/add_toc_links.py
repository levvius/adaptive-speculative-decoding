#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make the thesis table of contents clickable, in place.

The thesis ``papers/ИС61_fpm_КозинАА_2026.docx`` carries a static table of contents:
each entry is a paragraph with a right ``dot``-leader tab plus a page number,
and every section heading in the body is just a bold paragraph (no Word
heading style, no bookmark, no hyperlink). This script turns that static TOC
into a navigable one **without regenerating the document** — the working copy
contains ~250 KB of manual edits that exist only in the .docx, so we must not
rebuild it from ``build_newera_docx.py``.

For each TOC entry we:
  1. find the matching body heading (first bold paragraph after the TOC whose
     text equals the entry title, case-insensitively — the body uses UPPERCASE
     for appendices and lowercase ``jointadaspec`` while the TOC is title-case);
  2. drop a ``<w:bookmarkStart/>``…``<w:bookmarkEnd/>`` around that heading and
     add a ``<w:outlineLvl/>`` so the Word navigation pane and PDF bookmarks
     work too;
  3. wrap the TOC entry's title run(s) in ``<w:hyperlink w:anchor="...">`` so a
     click jumps straight to the heading. The page number stays outside the
     link and pagination is untouched (no field, no F9 update needed).

Internal anchor hyperlinks need no relationship entry, so no other part of the
package changes. Nothing in the document text or any number is modified.

Usage:
    python3 papers/add_toc_links.py [--docx PATH] [--dry-run]
"""

import argparse
import os
import re
import shutil
import sys
import zipfile

PARA = re.compile(r"<w:p\b.*?</w:p>", re.S)
WT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
RUN_START = re.compile(r"<w:r\b")
IND_LEFT = re.compile(r'<w:ind w:left="(\d+)"')

# TOC indent (twips) -> outline level. 0 = chapter, 1 = section, 2 = subsection.
LEVEL_BY_INDENT = {0: 0, 360: 1, 720: 2}

ANCHOR_PREFIX = "_jtoc_"
FIRST_BOOKMARK_ID = 1000  # safely above the few bookmarks already in the doc


def runs_text(xml_fragment):
    return "".join(WT.findall(xml_fragment))


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def key(s):
    return norm(s).casefold()


def toc_title(para):
    """Title = text of the run(s) before the page-number run (the one with <w:tab/>)."""
    return norm(runs_text(para.split("<w:tab/>", 1)[0]))


def toc_level(para):
    m = IND_LEFT.search(para.split("</w:pPr>", 1)[0])
    if not m:
        return 0
    return LEVEL_BY_INDENT.get(int(m.group(1)), 2)


def wrap_toc_entry(para, anchor):
    """Wrap the title run(s) of a TOC paragraph in an internal hyperlink."""
    head, sep, rest = para.partition("</w:pPr>")
    if not sep:
        return None
    assert rest.endswith("</w:p>")
    body = rest[: -len("</w:p>")]
    tab = body.find("<w:tab/>")
    if tab == -1:
        return None
    run_starts = [m.start() for m in RUN_START.finditer(body) if m.start() < tab]
    if not run_starts:
        return None
    split = run_starts[-1]  # start of the page-number run
    title_runs, page_runs = body[:split], body[split:]
    new_body = '<w:hyperlink w:anchor="%s">%s</w:hyperlink>%s' % (
        anchor,
        title_runs,
        page_runs,
    )
    return head + sep + new_body + "</w:p>"


def bookmark_heading(para, anchor, bid, level):
    """Add an outline level + bracket the heading runs with a bookmark."""
    head, sep, rest = para.partition("</w:pPr>")
    if not sep:
        return None
    ppr = head  # "<w:p ...><w:pPr>...."  (still open, without the closing tag)
    outline = '<w:outlineLvl w:val="%d"/>' % level
    # outlineLvl must sit just before the paragraph-mark <w:rPr> (schema order),
    # else immediately before the closing </w:pPr>.
    rpr = ppr.find("<w:rPr>")
    if rpr != -1:
        ppr = ppr[:rpr] + outline + ppr[rpr:]
    else:
        ppr = ppr + outline
    assert rest.endswith("</w:p>")
    runs = rest[: -len("</w:p>")]
    start = '<w:bookmarkStart w:id="%d" w:name="%s"/>' % (bid, anchor)
    end = '<w:bookmarkEnd w:id="%d"/>' % bid
    return ppr + sep + start + runs + end + "</w:p>"


def process(xml):
    paras = [(m.start(), m.group(0)) for m in PARA.finditer(xml)]
    toc = [(off, p) for off, p in paras if 'w:leader="dot"' in p and toc_title(p)]
    last_toc = max(off for off, _ in toc)
    body = [(off, p) for off, p in paras if off > last_toc and "<w:b/>" in p]

    edits = {}          # paragraph start offset -> replacement string
    used = set()        # body offsets already claimed
    matched, misses = 0, []
    bid = FIRST_BOOKMARK_ID

    for n, (off, p) in enumerate(toc, start=1):
        title = toc_title(p)
        hit = next(
            (boff for boff, bp in body
             if boff not in used and key(runs_text(bp)) == key(title)),
            None,
        )
        if hit is None:
            misses.append(title)
            continue
        anchor = "%s%03d" % (ANCHOR_PREFIX, n)
        new_toc = wrap_toc_entry(p, anchor)
        bp = dict(body)[hit]
        new_body = bookmark_heading(bp, anchor, bid, toc_level(p))
        if new_toc is None or new_body is None:
            misses.append(title)
            continue
        edits[off] = new_toc
        edits[hit] = new_body
        used.add(hit)
        matched += 1
        bid += 1

    # Single offset-keyed pass: swap only edited paragraphs, keep all else byte-for-byte.
    out, pos = [], 0
    for m in PARA.finditer(xml):
        out.append(xml[pos:m.start()])
        out.append(edits.get(m.start(), m.group(0)))
        pos = m.end()
    out.append(xml[pos:])
    return "".join(out), matched, len(toc), misses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docx", default=os.path.join(os.path.dirname(__file__), "ИС61_fpm_КозинАА_2026.docx"))
    ap.add_argument("--dry-run", action="store_true", help="report matches, write nothing")
    args = ap.parse_args()

    if not os.path.exists(args.docx):
        sys.exit("docx not found: %s" % args.docx)

    with zipfile.ZipFile(args.docx) as z:
        names = z.namelist()
        doc = z.read("word/document.xml").decode("utf-8")
        blobs = {n: z.read(n) for n in names}

    if 'w:anchor="%s' % ANCHOR_PREFIX in doc:
        sys.exit("Already linked (found %s anchors) — nothing to do." % ANCHOR_PREFIX)

    new_doc, matched, total, misses = process(doc)
    print("TOC entries: %d | linked: %d | unmatched: %d" % (total, matched, len(misses)))
    for t in misses:
        print("  UNMATCHED:", t)

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    if matched == 0:
        sys.exit("Refusing to write: no entries linked.")

    bak = args.docx + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(args.docx, bak)
        print("backup ->", bak)

    blobs["word/document.xml"] = new_doc.encode("utf-8")
    tmp = args.docx + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:  # preserve original part order
            z.writestr(n, blobs[n])
    os.replace(tmp, args.docx)
    print("wrote", args.docx)


if __name__ == "__main__":
    main()
