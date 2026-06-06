#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn the rendered defense PDF into a single self-contained HTML deck.

Each slide of ``papers/dist/JointAdaSpec_defense.pdf`` is rasterised with
poppler's ``pdftoppm`` and embedded (base64) into one portable HTML file with a
tiny keyboard/click viewer — so the third download format opens in any browser
with no server, no network and no external assets. If ``pdftoppm`` is missing,
falls back to LibreOffice's HTML export.

Run:  python3 papers/build_html_deck.py
"""

import base64
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "dist", "JointAdaSpec_defense.pdf")
PPTX = os.path.join(HERE, "dist", "JointAdaSpec_defense.pptx")
OUT = os.path.join(HERE, "dist", "JointAdaSpec_defense.html")
DPI = 120

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JointAdaSpec — защита ВКР · Козин А.А.</title>
<style>
 html,body{margin:0;height:100%;background:#0b1020;
   font-family:Calibri,Carlito,'DejaVu Sans',sans-serif;overflow:hidden}
 #stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
 #slide{max-width:100vw;max-height:100vh;box-shadow:0 0 48px rgba(0,0,0,.55)}
 #hud{position:fixed;bottom:10px;right:16px;color:#8A8F98;font-size:14px;
   font-variant-numeric:tabular-nums}
 #hint{position:fixed;bottom:10px;left:16px;color:#5A6172;font-size:12px}
 button.nav{position:fixed;top:0;height:100%;width:14%;border:0;background:transparent;cursor:pointer}
 #prev{left:0}#next{right:0}
</style></head><body>
<div id="stage"><img id="slide" alt="слайд"></div>
<button class="nav" id="prev" aria-label="назад"></button>
<button class="nav" id="next" aria-label="вперёд"></button>
<div id="hud"></div>
<div id="hint">← →&nbsp;·&nbsp;пробел&nbsp;·&nbsp;клик&nbsp;·&nbsp;F — на весь экран</div>
<script>
const S=__SLIDES__;let i=0;
const img=document.getElementById('slide'),hud=document.getElementById('hud');
function show(n){i=Math.max(0,Math.min(S.length-1,n));img.src=S[i];hud.textContent=(i+1)+' / '+S.length;}
function go(d){show(i+d);}
document.getElementById('next').onclick=()=>go(1);
document.getElementById('prev').onclick=()=>go(-1);
addEventListener('keydown',e=>{
 if(['ArrowRight','PageDown',' '].includes(e.key)){go(1);e.preventDefault();}
 else if(['ArrowLeft','PageUp'].includes(e.key))go(-1);
 else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);
 else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
});
show(0);
</script></body></html>
"""


def build_from_pngs(pngs):
    uris = []
    for p in pngs:
        with open(p, "rb") as f:
            uris.append("data:image/png;base64," + base64.b64encode(f.read()).decode("ascii"))
    arr = "[\n" + ",\n".join('"%s"' % u for u in uris) + "\n]"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__SLIDES__", arr))
    print("wrote %s (%d slides, %.1f MB)" % (OUT, len(pngs), os.path.getsize(OUT) / 1e6))


def main():
    if shutil.which("pdftoppm") and os.path.exists(PDF):
        tmp = tempfile.mkdtemp(prefix="jds_html_")
        try:
            subprocess.run(["pdftoppm", "-png", "-r", str(DPI), PDF, os.path.join(tmp, "s")], check=True)
            pngs = sorted(glob.glob(os.path.join(tmp, "s-*.png")))
            if not pngs:
                sys.exit("pdftoppm produced no pages")
            build_from_pngs(pngs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    # fallback: LibreOffice Impress HTML export
    if shutil.which("soffice") and os.path.exists(PPTX):
        print("pdftoppm unavailable — falling back to LibreOffice HTML export")
        subprocess.run(["soffice", "--headless",
                        "-env:UserInstallation=file:///tmp/lo_slides_html",
                        "--convert-to", "html", "--outdir", os.path.dirname(OUT), PPTX], check=True)
        return
    sys.exit("Need either pdftoppm+PDF or soffice+PPTX to build HTML.")


if __name__ == "__main__":
    main()
