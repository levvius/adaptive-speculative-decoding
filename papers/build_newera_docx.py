# -*- coding: utf-8 -*-
"""
Сборка первой редакции магистерской ВКР Козина А.А. в papers/Козин_newera.docx.

Подход воспроизводит build_thesis_docx.py: пакет .docx собирается напрямую из
OOXML без сторонних зависимостей (python-docx/lxml не установлены). За скелет
берётся ИС61_fpm_КозинАА_2026.docx — из него наследуются стиль Normal
(Times New Roman, интервал 1,5), геометрия страницы по ГОСТ (поля 30/15/20/20 мм)
и колонтитулы с нумерацией. Тело word/document.xml переписывается полностью.

В первом проходе пишутся: титульный лист, реферат, содержание всей работы,
введение, полностью раздел 1, список источников. Разделы 2-4, заключение и
приложения представлены заголовками с аннотированным планом (без авторской прозы).

Все числовые результаты взяты из FACT-блока papers/ВКР_тезисы_и_структура.md
(locked-артефакты 2026-05-14 ... 2026-05-27).
"""
import os
import re
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "ИС61_fpm_КозинАА_2026.docx")
OUT = os.path.join(HERE, "Козин_newera.docx")

# Геометрия текстового поля (twips): ширина = 11906 - 1701 - 851 = 9354.
TXT_W = 9354
CENTER_TAB = TXT_W // 2
RIGHT_TAB = TXT_W

IMG_REL_ID = "rId100"
IMG_NAME = "image_newera1.png"
IMG_REL_ID2 = "rId101"
IMG_NAME2 = "image_newera2.png"

# --------------------------------------------------------------------------
# Низкоуровневые конструкторы OOXML
# --------------------------------------------------------------------------
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs(text, bold=False, italic=False, sz=28):
    rpr = ""
    if bold:
        rpr += "<w:b/><w:bCs/>"
    if italic:
        rpr += "<w:i/><w:iCs/>"
    rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    return (f'<w:r><w:rPr>{rpr}</w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def P(text, indent=True, jc="both", bold=False, italic=False, sz=28, space_after=None):
    """Абзац тела: Times New Roman 14 pt, интервал 1,5 (из стиля Normal)."""
    ind = '<w:ind w:firstLine="709"/>' if indent else '<w:ind w:firstLine="0"/>'
    spc = f'<w:spacing w:after="{space_after}"/>' if space_after is not None else ""
    rpr = '<w:sz w:val="{0}"/><w:szCs w:val="{0}"/>'.format(sz)
    if bold:
        rpr = "<w:b/><w:bCs/>" + rpr
    if italic:
        rpr = "<w:i/><w:iCs/>" + rpr
    return (f'<w:p><w:pPr>{spc}{ind}<w:jc w:val="{jc}"/>'
            f'<w:rPr>{rpr}</w:rPr></w:pPr>{_runs(text, bold, italic, sz)}</w:p>')


def EMPTY():
    return '<w:p><w:pPr><w:rPr><w:sz w:val="28"/></w:rPr></w:pPr></w:p>'


def H1(text, page_break=True):
    """Заголовок верхнего уровня: ПРОПИСНЫЕ, по центру, bold, с новой страницы."""
    pb = "<w:pageBreakBefore/>" if page_break else ""
    return (f'<w:p><w:pPr>{pb}<w:spacing w:before="0" w:after="240"/>'
            f'<w:ind w:firstLine="0"/><w:jc w:val="center"/>'
            f'<w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>'
            f'{_runs(text.upper(), bold=True)}</w:p>')


def H2(text):
    """Глава X.Y: с абзацного отступа, bold, без точки."""
    return (f'<w:p><w:pPr><w:spacing w:before="240" w:after="120"/>'
            f'<w:ind w:firstLine="709"/><w:jc w:val="both"/><w:keepNext/>'
            f'<w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>'
            f'{_runs(text, bold=True)}</w:p>')


def H3(text):
    """Подглава X.Y.Z: с абзацного отступа, bold."""
    return (f'<w:p><w:pPr><w:spacing w:before="160" w:after="80"/>'
            f'<w:ind w:firstLine="709"/><w:jc w:val="both"/><w:keepNext/>'
            f'<w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>'
            f'{_runs(text, bold=True)}</w:p>')


def FORMULA(body, number):
    """Формула по центру; номер (раздел.номер) у правого края."""
    tabs = (f'<w:tabs><w:tab w:val="center" w:pos="{CENTER_TAB}"/>'
            f'<w:tab w:val="right" w:pos="{RIGHT_TAB}"/></w:tabs>')
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:before="120" w:after="120"/>'
            f'<w:ind w:firstLine="0"/><w:jc w:val="left"/>'
            f'<w:rPr><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:sz w:val="28"/></w:rPr><w:tab/></w:r>'
            f'{_runs(body, italic=True)}'
            f'<w:r><w:rPr><w:sz w:val="28"/></w:rPr><w:tab/></w:r>'
            f'{_runs("(" + number + ")")}</w:p>')


# --------------------------------------------------------------------------
# Конструктор OOXML-math — настоящие формулы Word через <m:oMath>.
# Каждый run несёт шрифт Cambria Math 14 pt; буквы автоматически курсивные,
# операторы и имена функций — прямые (<m:nor/>).
# --------------------------------------------------------------------------
_MFONT = ('<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
          '<w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>')
_MCTRL = ('<m:ctrlPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
          '<w:i/><w:iCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></m:ctrlPr>')


def mrun(t):
    """Математический run: буквы и цифры (буквы — авто-курсив)."""
    return f'<m:r>{_MFONT}<m:t xml:space="preserve">{esc(t)}</m:t></m:r>'


def op(t):
    """Прямой (upright) run: операторы, имена функций, словесные вставки."""
    return (f'<m:r><m:rPr><m:nor/></m:rPr>{_MFONT}'
            f'<m:t xml:space="preserve">{esc(t)}</m:t></m:r>')


def _f(x):
    """Готовый OMML-фрагмент пропускаем; голую строку оборачиваем как run."""
    return x if isinstance(x, str) and x.lstrip().startswith('<m:') else mrun(x)


def mrow(*parts):
    return "".join(_f(p) for p in parts)


def msub(b, sub):
    return (f'<m:sSub><m:sSubPr>{_MCTRL}</m:sSubPr>'
            f'<m:e>{_f(b)}</m:e><m:sub>{_f(sub)}</m:sub></m:sSub>')


def msup(b, sup):
    return (f'<m:sSup><m:sSupPr>{_MCTRL}</m:sSupPr>'
            f'<m:e>{_f(b)}</m:e><m:sup>{_f(sup)}</m:sup></m:sSup>')


def msubsup(b, sub, sup):
    return (f'<m:sSubSup><m:sSubSupPr>{_MCTRL}</m:sSubSupPr>'
            f'<m:e>{_f(b)}</m:e><m:sub>{_f(sub)}</m:sub>'
            f'<m:sup>{_f(sup)}</m:sup></m:sSubSup>')


def mfrac(num, den):
    return (f'<m:f><m:fPr>{_MCTRL}</m:fPr>'
            f'<m:num>{_f(num)}</m:num><m:den>{_f(den)}</m:den></m:f>')


def mnary(chrs, sub, sup, e):
    sub_x = f'<m:sub>{_f(sub)}</m:sub>' if sub else '<m:sub/>'
    sup_x = f'<m:sup>{_f(sup)}</m:sup>' if sup else '<m:sup/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="{chrs}"/><m:limLoc m:val="undOvr"/>'
            f'<m:subHide m:val="{0 if sub else 1}"/>'
            f'<m:supHide m:val="{0 if sup else 1}"/>{_MCTRL}</m:naryPr>'
            f'{sub_x}{sup_x}<m:e>{_f(e)}</m:e></m:nary>')


def mdelim(e, beg="(", end=")"):
    return (f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/>'
            f'{_MCTRL}</m:dPr><m:e>{_f(e)}</m:e></m:d>')


def msqrt(e):
    return (f'<m:rad><m:radPr><m:degHide m:val="1"/>{_MCTRL}</m:radPr>'
            f'<m:deg/><m:e>{_f(e)}</m:e></m:rad>')


def macc(e, chr="̂"):
    """Диакритика над выражением (по умолчанию — крышка, как в V̂, P̂)."""
    return (f'<m:acc><m:accPr><m:chr m:val="{chr}"/>{_MCTRL}</m:accPr>'
            f'<m:e>{_f(e)}</m:e></m:acc>')


def EQ(omml, number):
    """Нумерованная формула: математика по центру, номер (N) у правого края."""
    tabs = (f'<w:tabs><w:tab w:val="center" w:pos="{CENTER_TAB}"/>'
            f'<w:tab w:val="right" w:pos="{RIGHT_TAB}"/></w:tabs>')
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:before="120" w:after="120"/>'
            f'<w:ind w:firstLine="0"/><w:jc w:val="left"/>'
            f'<w:rPr><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:sz w:val="28"/></w:rPr><w:tab/></w:r>'
            f'<m:oMath>{omml}</m:oMath>'
            f'<w:r><w:rPr><w:sz w:val="28"/></w:rPr><w:tab/></w:r>'
            f'<w:r><w:rPr><w:sz w:val="28"/></w:rPr>'
            f'<w:t xml:space="preserve">({esc(number)})</w:t></w:r></w:p>')


def _inline(part):
    """Часть абзаца: OMML-фрагмент → инлайн <m:oMath>; строка → текстовый run."""
    if isinstance(part, str) and part.lstrip().startswith('<m:'):
        return f'<m:oMath>{part}</m:oMath>'
    return _runs(part)


def PM(parts, indent=True, jc="both"):
    """Абзац с инлайн-математикой: parts — список строк (текст) и OMML-фрагментов."""
    ind = '<w:ind w:firstLine="709"/>' if indent else '<w:ind w:firstLine="0"/>'
    body = "".join(_inline(p) for p in parts)
    return (f'<w:p><w:pPr>{ind}<w:jc w:val="{jc}"/>'
            f'<w:rPr><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:pPr>{body}</w:p>')


def WHERE(intro, items):
    """Блок «где …»: вводная строка, затем по строке на символ.

    Элемент — строка (как есть) или кортеж (omml_символ, описание) для инлайн-математики.
    """
    out = [P(intro, indent=False, jc="left", space_after=40)]
    for it in items:
        if isinstance(it, tuple):
            sym, desc = it
            inner = f'<m:oMath>{sym}</m:oMath>' + _runs(desc)
        else:
            inner = _runs(it)
        out.append('<w:p><w:pPr><w:ind w:left="709" w:firstLine="0"/>'
                   '<w:spacing w:after="40"/><w:jc w:val="left"/>'
                   '<w:rPr><w:sz w:val="28"/></w:rPr></w:pPr>'
                   + inner + '</w:p>')
    return "".join(out)


def TBL_CAPTION(text):
    return P(text, indent=False, jc="left", space_after=60)


def FIG_CAPTION(text):
    return P(text, indent=False, jc="center", space_after=120)


def LISTING(caption, code):
    """Блок программного кода: подпись «Листинг N — …» сверху, затем моноширинный текст.

    code — строка с переносами \\n или список строк; отступы сохраняются.
    """
    lines = code.split("\n") if isinstance(code, str) else list(code)
    mono = ('<w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" '
            'w:cs="Courier New"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>')
    runs = []
    for i, ln in enumerate(lines):
        if i:
            runs.append(f'<w:r>{mono}<w:br/></w:r>')
        runs.append(f'<w:r>{mono}<w:t xml:space="preserve">{esc(ln)}</w:t></w:r>')
    para = ('<w:p><w:pPr><w:spacing w:before="60" w:after="160" w:line="240" '
            'w:lineRule="auto"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
            '<w:shd w:val="clear" w:color="auto" w:fill="F5F5F5"/>'
            f'<w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>'
            f'<w:sz w:val="22"/></w:rPr></w:pPr>{"".join(runs)}</w:p>')
    return P(caption, indent=False, jc="left", space_after=40) + para


def TBL(rows, widths, align=None):
    """Таблица с тонкими границами; первая строка — серая шапка."""
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [
        f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>'
        '<w:jc w:val="center"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
        '<w:tblCellMar><w:left w:w="60" w:type="dxa"/>'
        '<w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
    ]
    for ri, row in enumerate(rows):
        header = ri == 0
        out.append("<w:tr>")
        for ci, cell in enumerate(row):
            shd = ('<w:shd w:val="clear" w:color="auto" w:fill="E8E8E8"/>'
                   if header else "")
            b = "<w:b/><w:bCs/>" if header else ""
            jc = "center" if (header or align is None) else align[ci]
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/>{shd}</w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="{jc}"/><w:ind w:firstLine="0"/>'
                '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>'
                '<w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:pPr>'
                f'<w:r><w:rPr>{b}<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(cell)}</w:t></w:r></w:p></w:tc>'
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def TOC_LINE(text, page, bold=False, level=0):
    """Строка содержания: текст + точечный заполнитель + номер страницы."""
    left = {0: 0, 1: 360, 2: 720}[level]
    tabs = f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{RIGHT_TAB}"/></w:tabs>'
    rpr = '<w:sz w:val="28"/><w:szCs w:val="28"/>'
    if bold:
        rpr = "<w:b/><w:bCs/>" + rpr
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:after="40"/>'
            f'<w:ind w:left="{left}" w:firstLine="0"/><w:jc w:val="left"/>'
            f'<w:rPr>{rpr}</w:rPr></w:pPr>'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:tab/></w:r>'
            f'<w:r><w:rPr>{rpr}</w:rPr><w:t>{esc(str(page))}</w:t></w:r></w:p>')


def CENTER(text, bold=False, sz=28, caps=False, space_after=120):
    t = text.upper() if caps else text
    return P(t, indent=False, jc="center", bold=bold, sz=sz, space_after=space_after)


def FIG_IMAGE(cx, cy, rel_id=IMG_REL_ID, pid=100, name="Рисунок 1"):
    """Inline-рисунок по центру; ссылается на relationship rel_id."""
    drawing = (
        '<w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{pid}" name="{name}"/>'
        '<wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" r:embed="{rel_id}"/>'
        '<a:stretch xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
    )
    return ('<w:p><w:pPr><w:spacing w:before="120" w:after="60"/>'
            '<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            + drawing + '</w:p>')


def block(parts):
    return "".join(parts)


# ==========================================================================
# СХЕМАТИЧЕСКИЙ РИСУНОК 1.1 (matplotlib)
# ==========================================================================
def make_figure(path):
    """Схема цикла спекулятивного декодирования. Возвращает (px_w, px_h) или None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(9.2, 3.1), dpi=170)
        ax.set_xlim(0, 9.2); ax.set_ylim(0, 3.1); ax.axis("off")

        def box(x, y, w, h, text, fc, ec="black"):
            p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                               linewidth=1.2, edgecolor=ec, facecolor=fc)
            ax.add_patch(p)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

        # Черновая модель: 4 предложенных токена.
        ax.text(0.05, 2.62, "Черновая модель q:", fontsize=10, fontweight="bold", va="center")
        labels_d = ["y₁", "y₂", "y₃", "y₄"]
        for i, t in enumerate(labels_d):
            box(2.7 + i * 1.15, 2.35, 0.95, 0.55, t, "#eef2ff")
        # Целевая модель: один параллельный проход проверки.
        ax.text(0.05, 1.42, "Целевая модель p:", fontsize=10, fontweight="bold", va="center")
        verdict = [("y₁", "#d7f3dd", "принят"), ("y₂", "#d7f3dd", "принят"),
                   ("y₃", "#d7f3dd", "принят"), ("y₄", "#f9d7d7", "отклонён")]
        for i, (t, fc, _) in enumerate(verdict):
            box(2.7 + i * 1.15, 1.15, 0.95, 0.55, t, fc)
        # Ресемплинг 1 токена из остаточного распределения.
        box(2.7 + 4 * 1.15, 1.15, 1.25, 0.55, "y₄′ ~ (p−q)₊", "#fff2cc")
        # Вертикальные стрелки проверки.
        for i in range(4):
            x = 2.7 + i * 1.15 + 0.475
            ax.add_patch(FancyArrowPatch((x, 2.33), (x, 1.72), arrowstyle="-|>",
                                         mutation_scale=11, linewidth=1.0, color="#555"))
        # Подписи итога.
        ax.text(2.7 + 1.5 * 1.15, 0.78, "принятый префикс (3 токена)",
                ha="center", fontsize=9, color="#2a7a3a")
        ax.text(2.7 + 4 * 1.15 + 0.62, 0.78, "+1 ресемпл",
                ha="center", fontsize=9, color="#9a7d00")
        ax.text(4.6, 0.18, "один проход целевой модели подтверждает до γ+1 токенов",
                ha="center", fontsize=9, style="italic", color="#333")
        fig.tight_layout(pad=0.3)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        w, h = fig.canvas.get_width_height()
        plt.close(fig)
        return w, h
    except Exception as e:  # noqa
        print(f"[figure] не удалось построить рисунок: {e}")
        return None


def make_figure_pipeline(path):
    """Схема конвейера JointAdaSpec: 4 стадии, конфиги, отчёты. Возвращает (px_w, px_h)|None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(9.4, 3.7), dpi=170)
        ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.7); ax.axis("off")

        def box(x, y, w, h, text, fc, fs=9.0):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.2, edgecolor="black", facecolor=fc))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

        xs = [0.25, 2.55, 4.85, 7.15]
        w, h, y = 1.95, 0.78, 1.5
        stages = ["01\nСбор трейсов", "02\nРешение MDP", "03\nБенчмарк",
                  "04\nПроверка C1–C4"]
        for txt, x in zip(stages, xs):
            box(x, y, w, h, txt, "#eef2ff")
        artifacts = ["traces.parquet", "policy.npz", "results.jsonl"]
        for i in range(3):
            x0, x1 = xs[i] + w, xs[i + 1]
            ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2),
                         arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#555"))
            ax.text((x0 + x1) / 2, y + h + 0.12, artifacts[i], ha="center",
                    fontsize=8, color="#333", style="italic")
        ax.add_patch(FancyArrowPatch((xs[3] + w, y + h / 2), (xs[3] + w + 0.65, y + h / 2),
                     arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#555"))
        ax.text(xs[3] + w + 0.32, y + h + 0.12, "conditions.json", ha="center",
                fontsize=8, color="#333", style="italic")
        # configs сверху — пунктир ко всем стадиям
        box(3.35, 3.0, 2.7, 0.55, "configs/ (Hydra, YAML)", "#fff2cc", fs=9)
        for x in xs:
            ax.add_patch(FancyArrowPatch((4.7, 3.0), (x + w / 2, y + h),
                         arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                         color="#b8860b", linestyle="--"))
        # reports снизу — читает стадии 2–4
        box(2.95, 0.2, 3.5, 0.6, "reports/templates\n(Парето, пороги, ablation)",
            "#d7f3dd", fs=8.5)
        for x in xs[1:]:
            ax.add_patch(FancyArrowPatch((x + w / 2, y), (4.7, 0.8),
                         arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                         color="#2a7a3a", linestyle="--"))
        fig.tight_layout(pad=0.3)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        wpx, hpx = fig.canvas.get_width_height()
        plt.close(fig)
        return wpx, hpx
    except Exception as e:  # noqa
        print(f"[figure-3.1] не удалось построить рисунок: {e}")
        return None


# ==========================================================================
# КОНТЕНТ
# ==========================================================================
KEYWORDS = ("СПЕКУЛЯТИВНОЕ ДЕКОДИРОВАНИЕ, БОЛЬШИЕ ЯЗЫКОВЫЕ МОДЕЛИ, УСКОРЕНИЕ "
            "ИНФЕРЕНСА, МАРКОВСКИЙ ПРОЦЕСС ПРИНЯТИЯ РЕШЕНИЙ, VALUE ITERATION, "
            "АДАПТИВНОЕ УПРАВЛЕНИЕ, ДЛИНА ЧЕРНОВИКА, ПОРОГ ВЕРИФИКАЦИИ, JOINTADASPEC.")

# --------------------- Титульный лист ---------------------
TITLE = block([
    CENTER("Министерство науки и высшего образования Российской Федерации", sz=24, space_after=40),
    CENTER("Федеральное государственное бюджетное образовательное учреждение "
           "высшего образования", sz=24, space_after=40),
    CENTER("«Кубанский государственный университет» (КубГУ)", bold=True, sz=24, space_after=40),
    CENTER("Факультет прикладной математики", sz=24, space_after=40),
    CENTER("Кафедра математического моделирования", sz=24, space_after=600),
    CENTER("ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА", bold=True, space_after=40),
    CENTER("(МАГИСТЕРСКАЯ ДИССЕРТАЦИЯ)", bold=True, space_after=360),
    CENTER("СОВМЕСТНОЕ АДАПТИВНОЕ СПЕКУЛЯТИВНОЕ ДЕКОДИРОВАНИЕ", bold=True, space_after=40),
    CENTER("БОЛЬШИХ ЯЗЫКОВЫХ МОДЕЛЕЙ НА ОСНОВЕ МАРКОВСКОГО", bold=True, space_after=40),
    CENTER("ПРОЦЕССА ПРИНЯТИЯ РЕШЕНИЙ (JOINTADASPEC)", bold=True, space_after=700),
    P("Работу выполнил _______________________________ А.А. Козин", indent=False, jc="both", space_after=120),
    P("Направление подготовки 01.04.02 Прикладная математика и информатика", indent=False, jc="both", space_after=120),
    P("Направленность (профиль) Математическое и программное обеспечение "
      "вычислительных машин и систем", indent=False, jc="both", space_after=300),
    P("Научный руководитель _______________________________", indent=False, jc="both", space_after=600),
    CENTER("Краснодар 2026", space_after=0),
])

# --------------------- Реферат ---------------------
REFERAT = block([
    H1("Реферат"),
    P("Выпускная квалификационная работа (магистерская диссертация): 78 с., "
      "4 раздела, 10 рис., 10 табл., 19 источников, 4 приложения.", indent=False, jc="both"),
    P(KEYWORDS, indent=False, jc="both", bold=False),
    P("Объектом исследования является процесс инференса авторегрессионных больших "
      "языковых моделей, использующих спекулятивное декодирование.", indent=True),
    P("Предметом исследования являются методы совместной адаптивной оптимизации "
      "длины черновика и порога верификации в спекулятивном декодировании.", indent=True),
    P("Цель работы — разработка и экспериментальное исследование метода адаптивного "
      "спекулятивного декодирования, который совместно управляет длиной черновика и "
      "порогом нечёткой верификации для повышения пропускной способности инференса при "
      "контролируемом качестве генерации.", indent=True),
    P("Методы исследования: формализация управления в виде табличного марковского "
      "процесса принятия решений, решение методом разреженной итерации по ценности "
      "(value iteration), парный критерий Макнемара и бутстреп-оценка 95 %-х "
      "доверительных интервалов при сравнении методов.", indent=True),
    P("Основные результаты. Предложен метод JointAdaSpec, формализующий совместное "
      "управление двумя осями спекулятивного декодирования как задачу оптимального "
      "управления. На паре моделей Qwen2.5-14B-Instruct → Qwen2.5-0.5B-Instruct получен "
      "статистически значимый прирост точности +4,07 п.п. exact match на наборе GSM8K "
      "(p = 0,0205, критерий Макнемара, n = 1500 парных наблюдений) при пропускной "
      "способности 10,84 ток/с, что в 2,22 раза выше ванильного спекулятивного "
      "декодирования. Доказано, что совместная и каскадная политики статистически "
      "неразличимы (Δ = −0,13 п.п., p = 0,96), что является точным следствием "
      "доказанной теоремы о value gap. На паре с низким соотношением мощностей "
      "(7B/1,5B, 4,7×) прирост качества не подтверждён — установлено граничное условие "
      "применимости метода.", indent=True),
    P("Научная новизна состоит в том, что впервые длина черновика и порог нечёткой "
      "верификации рассматриваются как связанные переменные управления в единой "
      "оптимизационной постановке, для которой получены аналитические гарантии "
      "корректности и субоптимальности каскадных эвристик.", indent=True),
    P("Практическая значимость определяется тем, что метод восстанавливает пропускную "
      "способность, теряемую ванильным спекулятивным декодированием, без обучения "
      "дополнительных нейросетевых модулей и реализован в виде воспроизводимого "
      "программного комплекса с открытым исходным кодом.", indent=True),
])

# --------------------- Содержание ---------------------
SODER = block([
    H1("Содержание"),
    TOC_LINE("ВВЕДЕНИЕ", 7, bold=True),
    TOC_LINE("1 Теоретические основы и методы ускорения инференса больших языковых моделей", 12, bold=True),
    TOC_LINE("1.1 Большие языковые модели и задача инференса", 12, level=1),
    TOC_LINE("1.1.1 Декодер-трансформер и авторегрессионная генерация", 12, level=2),
    TOC_LINE("1.1.2 Последовательное узкое место, латентность и пропускная способность", 13, level=2),
    TOC_LINE("1.1.3 Кэш ключей и значений и режим, ограниченный памятью", 14, level=2),
    TOC_LINE("1.2 Спекулятивное декодирование", 15, level=1),
    TOC_LINE("1.2.1 Черновая модель, целевая модель и верификатор", 15, level=2),
    TOC_LINE("1.2.2 Точная верификация через модифицированный rejection sampling", 16, level=2),
    TOC_LINE("1.2.3 Вероятность принятия и формула ускорения", 18, level=2),
    TOC_LINE("1.3 Ослабленная верификация и компромисс «скорость — качество»", 19, level=1),
    TOC_LINE("1.3.1 Нечёткая верификация и порог принятия", 19, level=2),
    TOC_LINE("1.3.2 Компромисс между пропускной способностью и качеством", 20, level=2),
    TOC_LINE("1.3.3 Ограниченность фиксированных гиперпараметров", 21, level=2),
    TOC_LINE("1.4 Обзор современных методов спекулятивного декодирования", 21, level=1),
    TOC_LINE("1.4.1 Обучаемая верификация значимых токенов", 21, level=2),
    TOC_LINE("1.4.2 SpecExec и точная древовидная спекуляция", 22, level=2),
    TOC_LINE("1.4.3 Облегчённые черновые механизмы: Medusa, EAGLE и родственные", 22, level=2),
    TOC_LINE("1.4.4 Адаптивный выбор длины черновика", 23, level=2),
    TOC_LINE("1.4.5 Ослабленная адаптивная верификация и фронт Парето", 24, level=2),
    TOC_LINE("1.4.6 Систематизация и выявленный пробел", 24, level=2),
    TOC_LINE("1.5 Марковские процессы принятия решений как аппарат адаптивного управления", 25, level=1),
    TOC_LINE("1.5.1 Определение MDP и уравнение оптимальности Беллмана", 25, level=2),
    TOC_LINE("1.5.2 Метод value iteration: сходимость и вычислительная сложность", 27, level=2),
    TOC_LINE("1.5.3 Монотонные и пороговые политики", 27, level=2),
    TOC_LINE("1.6 Исследовательский пробел и постановка задачи", 28, level=1),
    TOC_LINE("1.6.1 Ограничения существующих адаптивных методов", 28, level=2),
    TOC_LINE("1.6.2 Идея совместного адаптивного управления", 29, level=2),
    TOC_LINE("1.6.3 Формальная постановка задачи совместной оптимизации", 29, level=2),
    TOC_LINE("2 Метод JointAdaSpec и его теоретический анализ", 31, bold=True),
    TOC_LINE("2.1 MDP-формулировка совместной оптимизации", 31, level=1),
    TOC_LINE("2.1.1 Пространство состояний и его дискретизация", 31, level=2),
    TOC_LINE("2.1.2 Пространство действий", 33, level=2),
    TOC_LINE("2.1.3 Функция переходов и функция награды", 34, level=2),
    TOC_LINE("2.2 Теоретический анализ метода", 36, level=1),
    TOC_LINE("2.2.1 Выборочная сложность оценщика переходов (теорема A)", 36, level=2),
    TOC_LINE("2.2.2 Беллман-инвариантность аддитивного штрафа качества (теорема B)", 37, level=2),
    TOC_LINE("2.2.3 Субоптимальность каскадных политик (теорема C)", 38, level=2),
    TOC_LINE("2.2.4 Точный value gap совместной и каскадной политик (теорема D)", 40, level=2),
    TOC_LINE("2.2.5 Скаляризация Парето-фронта (теорема 2.4)", 41, level=2),
    TOC_LINE("2.3 Алгоритм решения MDP", 41, level=1),
    TOC_LINE("2.4 Инференс-алгоритм с выученной политикой", 43, level=1),
    TOC_LINE("3 Программная реализация", 45, bold=True),
    TOC_LINE("3.1 Архитектура программного комплекса", 45, level=1),
    TOC_LINE("3.2 Модуль сбора трейсов", 47, level=1),
    TOC_LINE("3.3 Модуль оценки MDP и решения value iteration", 48, level=1),
    TOC_LINE("3.4 Модуль инференса с политикой JointAdaSpec", 50, level=1),
    TOC_LINE("3.5 Реализация baseline-методов", 51, level=1),
    TOC_LINE("3.6 Инструменты, тестирование и воспроизводимость", 52, level=1),
    TOC_LINE("4 Экспериментальное исследование", 55, bold=True),
    TOC_LINE("4.1 Методика экспериментов и метрики", 55, level=1),
    TOC_LINE("4.2 Исследуемые пары моделей и наборы данных", 56, level=1),
    TOC_LINE("4.3 Основной результат: пара 14B/0.5B", 57, level=1),
    TOC_LINE("4.4 Нуль-результат и триангуляция: пара 7B/1.5B", 58, level=1),
    TOC_LINE("4.5 Адаптивность против фиксированного порога (теорема E)", 60, level=1),
    TOC_LINE("4.6 Анализ компромисса по κ и нелинейности качества (теорема G)", 61, level=1),
    TOC_LINE("4.7 Сравнение с AutoJudge и обсуждение результатов", 62, level=1),
    TOC_LINE("ЗАКЛЮЧЕНИЕ", 65, bold=True),
    TOC_LINE("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", 68, bold=True),
    TOC_LINE("ПРИЛОЖЕНИЕ А. Доказательства теорем", 71, bold=True),
    TOC_LINE("ПРИЛОЖЕНИЕ Б. Архитектура репозитория и листинги", 75, bold=True),
    TOC_LINE("ПРИЛОЖЕНИЕ В. Дополнительные экспериментальные данные", 76, bold=True),
    TOC_LINE("ПРИЛОЖЕНИЕ Г. Манифесты воспроизводимости", 78, bold=True),
])

# --------------------- Введение ---------------------
VVEDENIE = block([
    H1("Введение"),
    P("Развёртывание больших языковых моделей (large language model, LLM) сместило "
      "основную долю вычислительных затрат с обучения на инференс — обслуживание "
      "пользовательских запросов. Обучение выполняется однократно, тогда как инференс "
      "повторяется при каждом обращении, и по оценкам операторов крупных сервисов на "
      "него приходится от 60 до 90 % полной стоимости жизненного цикла модели. "
      "Доминирующей операцией здесь служит авторегрессионная генерация: каждый "
      "следующий токен вычисляется отдельным прямым проходом по целевой модели и "
      "зависит от всех предыдущих. Ответ длиной в несколько сотен токенов требует "
      "стольких же последовательных, не распараллеливаемых обращений к модели с "
      "десятками миллиардов параметров — именно эта цепочка, а не объём арифметики, "
      "определяет наблюдаемую задержку."),
    P("Узкое место усугубляется режимом работы памяти. На шаге декодирования "
      "обрабатывается ровно один новый токен, но из памяти считываются все веса слоёв "
      "и весь накопленный кэш ключей и значений (key-value cache, KV-cache). "
      "Арифметическая интенсивность при этом мала, и шаг оказывается ограниченным "
      "пропускной способностью памяти (memory-bound), а не производительностью "
      "вычислителя; даже на ускорителях класса NVIDIA RTX 5090 с пропускной "
      "способностью порядка 1,8 ТБ/с арифметические блоки простаивают в ожидании "
      "данных. Отсюда ключевое следствие: проверка нескольких токенов за один проход "
      "почти не дороже проверки одного. На этом резерве и построено спекулятивное "
      "декодирование."),
    P("Спекулятивное декодирование (speculative decoding, SD), предложенное Leviathan "
      "и соавторами [10] и независимо Chen и соавторами [4], стало промышленным "
      "стандартом ускорения инференса без потери качества и реализовано в системах "
      "vLLM, TensorRT-LLM и SGLang. Дешёвая черновая модель за один проход предлагает "
      "блок из нескольких токенов, а дорогая целевая модель проверяет их одним "
      "параллельным проходом; правило modified rejection sampling гарантирует, что "
      "итоговое распределение принятых токенов совпадает с распределением целевой "
      "модели. Выигрыш достигается тогда, когда черновая модель угадывает достаточную "
      "долю токенов и за один дорогой проход подтверждается сразу несколько. Однако "
      "исходная формулировка использует фиксированную длину черновика и строгое правило "
      "приёма, что эмпирически неоптимально."),
    P("Литература 2023–2026 годов развивала спекулятивное декодирование по двум во "
      "многом независимым направлениям. Первое — адаптивный выбор длины черновика, где "
      "число предлагаемых токенов меняется по наблюдаемым признакам генерации: "
      "SpecDec++ [8] формулирует остановку черновика как задачу оптимальной остановки "
      "с обучаемым классификатором, SVIP и AdaEDL опираются на онлайн-оценку энтропии "
      "чернового распределения, DISCO использует контекстные эвристики, BanditSpec — "
      "бандитские стратегии. Второе направление — ослабленная (нечёткая) верификация, "
      "где условие приёма смягчается ради более высокой пропускной способности: judge "
      "decoding и AutoJudge [6] обучают верификатор отбирать значимые расхождения, "
      "Fuzzy SD [7] вводит скалярный порог приёма, MARS настраивает ослабление по "
      "локальным признакам. Теоретическую границу Парето между скоростью и качеством "
      "для ослабленных методов описали Yin и соавторы [18]. Оба направления управляют "
      "одной осью при фиксированной другой и не рассматривают их совместно."),
    P("Принципиальное ограничение существующих подходов состоит в том, что и длина "
      "черновика, и порог верификации, как правило, задаются фиксированными "
      "гиперпараметрами, подобранными по усреднённому поведению на валидационной "
      "выборке. Между тем оптимальная длина черновика и допустимая степень ослабления "
      "проверки зависят от текущего состояния генерации: на «лёгких» участках "
      "(типовые окончания, синтаксически предопределённые позиции) выгодны длинный "
      "черновик и мягкая проверка, на «трудных» (выбор числа в арифметической задаче, "
      "начало нового шага рассуждения) — короткий черновик и строгая проверка. Любое "
      "фиксированное значение оказывается компромиссом, не оптимальным ни для одного "
      "конкретного состояния. Более того, последовательное (каскадное) применение двух "
      "одномерных адаптивных правил не имеет гарантий оптимальности: оптимальное "
      "значение одного параметра функционально зависит от значения другого, поэтому "
      "оси нельзя оптимизировать порознь без потери совместного оптимума."),
    P("Целью настоящей работы является разработка и экспериментальное исследование "
      "метода адаптивного спекулятивного декодирования, который формализует совместное "
      "управление длиной черновика и порогом нечёткой верификации как задачу "
      "оптимального управления в дискретном марковском процессе принятия решений "
      "(Markov decision process, MDP) и решает её методом итерации по ценности "
      "(value iteration). Для достижения цели поставлены следующие задачи:"),
    P("1) проанализировать современные методы спекулятивного декодирования, выявить "
      "исследовательский пробел, связанный с отсутствием совместной оптимизации двух "
      "осей управления, и формализовать задачу совместного управления;",
      indent=True, jc="both"),
    P("2) построить MDP-модель совместной оптимизации — пространства состояний и "
      "действий, функции переходов и награды — и доказать её ключевые теоретические "
      "свойства;", indent=True, jc="both"),
    P("3) реализовать программный комплекс из стадий сбора трейсов, оценки параметров "
      "MDP, решения value iteration и инференса с выученной политикой, включая "
      "единообразные реализации базовых методов;", indent=True, jc="both"),
    P("4) провести воспроизводимую экспериментальную оценку качества и пропускной "
      "способности относительно базовых методов на современных открытых моделях "
      "семейства Qwen2.5 [13] и установить границы применимости подхода.",
      indent=True, jc="both"),
    P("Объектом исследования выступает процесс инференса авторегрессионных больших "
      "языковых моделей со спекулятивным декодированием. Предметом исследования "
      "являются методы совместной адаптивной оптимизации длины черновика и порога "
      "верификации, а также их теоретические свойства и эмпирическая эффективность."),
    P("Методологическую основу работы составляют теория марковских процессов принятия "
      "решений и динамического программирования [12, 14], аппарат приближённого "
      "обучения с подкреплением [9] и методы статистической проверки гипотез. Сравнение "
      "методов проводится на парных наблюдениях с применением критерия Макнемара и "
      "бутстреп-оценки 95 %-х доверительных интервалов; качество измеряется метрикой "
      "exact match на наборе математических задач GSM8K [5], скорость — пропускной "
      "способностью в токенах в секунду."),
    P("Научная новизна работы состоит в том, что длина черновика и порог нечёткой "
      "верификации впервые рассматриваются не как независимо настраиваемые "
      "гиперпараметры, а как связанные переменные управления в единой MDP-постановке с "
      "малым пространством состояний, допускающим точное решение методом value "
      "iteration без обучения нейросетевой политики. Для этой постановки получены "
      "аналитические результаты: оценка выборочной сложности оценщика переходов, "
      "условие Беллман-инвариантности аддитивного штрафа качества, линейная по мере "
      "нарушений оценка субоптимальности каскадных политик и точное выражение для "
      "разрыва ценности между совместной и каскадной политиками."),
    P("Практическая значимость определяется тем, что предложенный метод восстанавливает "
      "пропускную способность, теряемую ванильным спекулятивным декодированием на "
      "одиночном ускорителе, не требует обучения дополнительных нейросетевых модулей "
      "(политика хранится в виде таблицы и стоит O(1) на шаг декодирования) и совместим "
      "с существующими инференс-фреймворками. Реализация оформлена как воспроизводимый "
      "программный комплекс с фиксацией версий, начальных значений генераторов и "
      "манифестов. Полученные результаты очерчивают и границы применимости подхода, что "
      "важно для инженерных решений о его внедрении."),
    P("Работа состоит из введения, четырёх разделов, заключения, списка использованных "
      "источников и четырёх приложений. Первый раздел вводит теоретические основы "
      "инференса LLM и спекулятивного декодирования, даёт обзор современных методов и "
      "аппарата MDP, формулирует исследовательский пробел. Второй раздел излагает метод "
      "JointAdaSpec и его теоретический анализ. Третий раздел описывает программную "
      "реализацию, четвёртый — экспериментальную оценку. В заключении сформулированы "
      "основные результаты и направления дальнейших исследований."),
])


# --------------------- Раздел 1 ---------------------
def razdel1(fig_block):
    return block([
        H1("1 Теоретические основы и методы ускорения инференса больших языковых моделей"),
        P("Настоящий раздел вводит понятийный и математический аппарат, на котором "
          "строится работа. Сначала рассматривается инференс больших языковых моделей и "
          "природа его узкого места (1.1), затем — спекулятивное декодирование как "
          "способ обойти это узкое место без потери качества (1.2) и его ослабленные "
          "варианты, порождающие компромисс «скорость — качество» (1.3). Раздел 1.4 "
          "систематизирует современные методы, раздел 1.5 излагает аппарат марковских "
          "процессов принятия решений, а раздел 1.6 формулирует исследовательский "
          "пробел и постановку задачи, мотивирующую метод JointAdaSpec."),

        H2("1.1 Большие языковые модели и задача инференса"),
        H3("1.1.1 Декодер-трансформер и авторегрессионная генерация"),
        P("Большая языковая модель (large language model, LLM) — это нейросетевая "
          "модель, аппроксимирующая распределение вероятностей над последовательностями "
          "токенов естественного языка. Современные LLM семейств LLaMA, Qwen и DeepSeek "
          "насчитывают от единиц до сотен миллиардов параметров и почти без исключения "
          "построены на архитектуре декодер-трансформера (decoder-only transformer), "
          "восходящей к механизму внимания [17]. Модель состоит из стопки одинаковых "
          "блоков, каждый из которых сочетает причинно-маскированное самовнимание "
          "(causal self-attention) и позиционно-независимую полносвязную подсеть; "
          "причинная маска запрещает позиции обращаться к будущим токенам, что и делает "
          "модель пригодной для левостороннего (слева направо) порождения текста. Размер "
          "словаря V для перечисленных семейств составляет порядка 10⁵ токенов."),
        P("Пусть x — входная последовательность (промпт), а y = (y_1, …, y_n) — "
          "порождаемый ответ над словарём V. Декодер-трансформер задаёт распределение "
          "ответа авторегрессионно, то есть как произведение условных распределений "
          "каждого токена при условии всех предыдущих:"),
        EQ(mrow(
            msub("p", "θ"), mdelim(mrow("y", op(" | "), "x")), op(" = "),
            mnary("∏", "t=1", "n",
                  mrow(msub("p", "θ"),
                       mdelim(mrow(msub("y", "t"), op(" | "), "x", op(", "),
                                   msub("y", "<t"))))),
        ), "1.1"),
        WHERE("где", [
            (mrow("θ"), " — параметры модели;"),
            (mrow(msub("y", "<t")), " — префикс ранее сгенерированных токенов;"),
            (mrow(msub("p", "θ"), mdelim(mrow(op("·"), op(" | "), "x", op(", "),
                                              msub("y", "<t")))),
             " — распределение над словарём V на шаге t."),
        ]),
        P("Из выражения (1.1) следует ключевая для всей работы особенность: распределение "
          "токена y_t зависит от уже выбранного значения предыдущего токена. Сгенерировать "
          "токен "
          "можно лишь после того, как зафиксирован предыдущий, поэтому шаги порождения "
          "образуют строго последовательную цепочку. Именно эта зависимость, а не объём "
          "арифметики самого по себе, определяет стоимость генерации."),
        P("Из условного распределения (1.1) ответ получают одной из стратегий "
          "декодирования. Жадное декодирование (greedy) выбирает на каждом шаге наиболее "
          "вероятный токен; стохастическое семплирование выбирает токен случайно, часто "
          "с температурным масштабированием логитов, регулирующим разнообразие. Для "
          "дальнейшего изложения существенно, что спекулятивное декодирование "
          "воспроизводит выбранную стратегию целевой модели без искажений, поэтому "
          "рассуждение ведётся в терминах целевого распределения p и переносится на оба "
          "режима там, где различие между ними несущественно."),

        H3("1.1.2 Последовательное узкое место, латентность и пропускная способность"),
        P("Авторегрессионная генерация ответа длины n требует n последовательных прямых "
          "проходов (forward pass) по целевой модели. В отличие от стадии обработки "
          "промпта (prefill), где все входные токены проходят через модель одним "
          "параллельным проходом, стадия декодирования (decode) принципиально "
          "последовательна: распараллелить её на уровне токенов в рамках стандартного "
          "подхода невозможно, поскольку вход очередного шага есть выход предыдущего. "
          "Это ограничение принято называть последовательным узким местом декодирования "
          "(sequential decode bottleneck)."),
        P("Производительность инференса характеризуют двумя различными метриками. "
          "Латентность (latency) — время от запроса до получения ответа; для одиночной "
          "последовательности она пропорциональна числу шагов n и времени одного прохода "
          "по целевой модели. Пропускная способность (throughput) — число токенов в "
          "секунду (ток/с), генерируемых системой. Эти метрики не эквивалентны: "
          "пакетирование (batching) множества запросов повышает суммарную пропускную "
          "способность за счёт параллелизма по запросам, но не сокращает число "
          "последовательных шагов и потому не улучшает латентность отдельного ответа. "
          "Спекулятивное декодирование атакует узкое место с другой стороны — сокращает "
          "число обращений к целевой модели на один порождённый токен."),
        P("Количественно различие стадий выражается арифметической интенсивностью — "
          "отношением числа операций к объёму считанных из памяти данных. На стадии "
          "prefill это отношение велико, и стадия ограничена производительностью "
          "вычислителя; на стадии decode при batch size, равном единице, оно падает до "
          "единиц операций на байт, и стадия ограничена памятью. Пакетирование запросов "
          "повышает интенсивность декодирования и потому суммарную пропускную "
          "способность, однако, как отмечено выше, не сокращает число последовательных "
          "шагов отдельного ответа и не улучшает его латентность."),

        H3("1.1.3 Кэш ключей и значений и режим, ограниченный памятью"),
        P("Чтобы на каждом шаге не пересчитывать представления всего префикса заново, "
          "используется кэш ключей и значений (key-value cache, KV-cache): векторы "
          "ключей и значений механизма внимания, вычисленные для уже обработанных "
          "токенов, сохраняются и переиспользуются на последующих шагах. Кэширование "
          "снижает арифметическую сложность шага декодирования с квадратичной до "
          "линейной по длине префикса, однако порождает иное ограничение."),
        P("На шаге декодирования модель обрабатывает ровно один новый токен, тогда как "
          "для вычисления внимания необходимо считать из памяти весь KV-кэш и все "
          "веса слоёв. Отношение числа арифметических операций к объёму перемещаемых "
          "данных (арифметическая интенсивность) при этом мало, и шаг оказывается "
          "ограниченным пропускной способностью памяти (memory-bound), а не "
          "производительностью вычислителя (compute-bound). Графический ускоритель в "
          "этом режиме недозагружен: его арифметические блоки простаивают в ожидании "
          "данных. Практическое следствие двояко. С одной стороны, обработка нескольких "
          "токенов за один проход почти не увеличивает время шага по сравнению с "
          "обработкой одного — резерв вычислителя позволяет «бесплатно» проверить "
          "сразу несколько кандидатов. Именно на этом наблюдении основано спекулятивное "
          "декодирование. С другой стороны, рост KV-кэша с длиной контекста увеличивает "
          "объём перемещаемых данных и потому остаётся существенным фактором стоимости."),

        H2("1.2 Спекулятивное декодирование"),
        H3("1.2.1 Черновая модель, целевая модель и верификатор"),
        P("Спекулятивное декодирование (speculative decoding, SD) ускоряет инференс, "
          "вводя в схему вторую, существенно более дешёвую модель [10, 4]. Целевая модель "
          "(target model) с распределением p — это та модель, выборку из которой "
          "требуется получить; обычно она велика (единицы–десятки миллиардов параметров) "
          "и определяет качество. Черновая модель (draft model) с распределением q "
          "меньше на один–два порядка и служит для дешёвого выдвижения гипотез. "
          "Верификатор (verifier) — процедура, которая по предложенным черновиком "
          "токенам и распределениям обеих моделей решает, какие из них принять."),
        P("Один цикл SD устроен так. Черновая модель авторегрессионно порождает блок из "
          "γ токенов-кандидатов (γ называют длиной черновика). Затем целевая модель "
          "одним параллельным проходом вычисляет свои распределения p для всех γ позиций "
          "блока сразу — это возможно, поскольку кандидаты уже известны и причинная "
          "маска допускает их совместную обработку. Наконец, верификатор сопоставляет p "
          "и q и принимает максимально длинный согласованный префикс кандидатов, после "
          "чего цикл повторяется. Выигрыш возникает оттого, что один дорогой проход "
          "целевой модели подтверждает сразу несколько токенов."),
        P("Существенным практическим условием применимости SD является совместимость "
          "токенизаторов черновой и целевой моделей. Правило приёма сопоставляет "
          "вероятности, присвоенные одному и тому же токену двумя моделями, поэтому обе "
          "обязаны использовать идентичное отображение текста в идентификаторы токенов; "
          "иначе сравнение p(y_i) и q(y_i) теряет смысл. На практике это ограничивает "
          "выбор пар: черновую и целевую модели берут из одного семейства с общим "
          "словарём — в настоящей работе из семейства Qwen2.5 [13], где это условие "
          "выполнено по построению."),

        H3("1.2.2 Точная верификация через модифицированный rejection sampling"),
        P("Нетривиальность SD в том, что наивная проверка (принять кандидат, если он "
          "совпал с argmax целевой модели) исказила бы распределение. Корректность "
          "обеспечивает правило модифицированного отклоняющего семплирования (modified "
          "rejection sampling) [10]. Кандидат y_i, выбранный черновиком из q, "
          "принимается с вероятностью"),
        EQ(mrow(
            msub("a", "i"), op(" = "), op("min"),
            mdelim(mrow("1", op(", "),
                        mfrac(mrow("p", mdelim(msub("y", "i"))),
                              mrow("q", mdelim(msub("y", "i")))))),
        ), "1.2"),
        WHERE("где", [
            "p(y_i) — вероятность токена y_i по целевой модели;",
            "q(y_i) — вероятность того же токена по черновой модели.",
        ]),
        P("Если кандидат отклонён, на его позиции токен пересемплируется из нормированного "
          "остаточного распределения с плотностью, пропорциональной (p − q)_+ = "
          "max(0, p − q), а оставшиеся кандидаты блока отбрасываются. Можно показать, что "
          "при таком правиле итоговое распределение принятого (или пересемплированного) "
          "токена в точности совпадает с распределением целевой модели p. Тем самым "
          "спекулятивное декодирование является не приближённым, а точным методом: оно "
          "ускоряет генерацию, не меняя порождаемого распределения. Это утверждение, "
          "которое принято называть свойством точности спекулятивного декодирования, "
          "доказывается индукцией по числу токенов блока с использованием марковости "
          "процесса генерации; полное доказательство вынесено в приложение А. Схема "
          "одного цикла приведена на рисунке 1.1."),
        fig_block,
        P("Принципиально, что выигрыш по скорости достигается без какого-либо "
          "переобучения целевой модели и без догадок о «правильности» токенов: правило "
          "(1.2) самостоятельно отбраковывает кандидаты, в которых черновик и цель "
          "расходятся. Качество ответа в точности соответствует целевой модели, а "
          "черновик влияет лишь на скорость."),

        H3("1.2.3 Вероятность принятия и формула ускорения"),
        P("Ускорение определяется тем, насколько часто кандидаты черновика принимаются. "
          "Обозначим через α ожидаемую вероятность принятия одного токена — меру "
          "согласованности черновой и целевой моделей. Если принятия независимы и "
          "равновероятны, то ожидаемое число токенов, подтверждаемых за один проход "
          "целевой модели при длине черновика γ, равно"),
        EQ(mrow(
            op("E"), mdelim("β", "[", "]"), op(" = "),
            mfrac(mrow("1", op(" − "), msup("α", mrow("γ", op("+"), "1"))),
                  mrow("1", op(" − "), "α")),
        ), "1.3"),
        WHERE("где", [
            "α ∈ [0, 1] — ожидаемая вероятность принятия токена;",
            "γ — длина черновика (число кандидатов в блоке);",
            "β — число токенов, порождённых за один цикл.",
        ]),
        P("Поскольку каждый цикл стоит одного прохода целевой модели и γ проходов "
          "(дешёвой) черновой, итоговый коэффициент ускорения относительно "
          "пошаговой генерации равен E[β] / (1 + γ c), где c — отношение стоимости "
          "прохода черновой модели к стоимости прохода целевой. Отсюда видны два "
          "источника проигрыша. При низком α множитель (1.3) близок к единице, и "
          "накладные расходы на черновик не окупаются; при малом отношении размеров "
          "моделей величина c велика, и знаменатель растёт, съедая выигрыш от "
          "подтверждённых токенов. Оба эффекта наблюдаются эмпирически: на паре "
          "Qwen2.5-7B → 1.5B с соотношением мощностей "
          "лишь 4,7× ванильное спекулятивное декодирование оказывается даже медленнее "
          "прямой генерации целевой моделью. Это не дефект метода, а прямое следствие "
          "формулы (1.3): спекуляция выгодна лишь при достаточно высоком α и достаточно "
          "большом разрыве в стоимости моделей."),
        P("Из формулы (1.3) вытекает и теоретический потолок ускорения точного "
          "спекулятивного декодирования. При α → 1 множитель E[β] стремится к γ + 1, и "
          "коэффициент ускорения приближается к (γ + 1) / (1 + γc), что при больших γ не "
          "превосходит 1/c — обратной величины относительной стоимости чернового "
          "прохода. Преодолеть этот потолок, оставаясь в классе точных методов, "
          "невозможно: дальнейший выигрыш достижим либо повышением согласованности α за "
          "счёт более удачной черновой модели, либо контролируемым ослаблением правила "
          "приёма, которое рассматривается в разделе 1.3."),
        P("Эмпирически относительная стоимость c прямо определяет применимость метода. На "
          "основной паре (целевая модель 14B, черновая 0,5B) отношение мощностей около "
          "28-кратного, величина c мала, и потолок 1/c высок; на дополнительной паре "
          "(7B и 1,5B) отношение лишь 4,7-кратное, c велико, и потолок низок. Этим и "
          "объясняется наблюдаемое в разделе 4 явление: на второй паре ванильное "
          "спекулятивное декодирование оказывается медленнее прямой генерации, поскольку "
          "дешёвого черновика, способного окупить накладные расходы, при таком отношении "
          "попросту нет."),

        H2("1.3 Ослабленная верификация и компромисс «скорость — качество»"),
        H3("1.3.1 Нечёткая верификация и порог принятия"),
        P("Точное правило (1.2) консервативно: оно отклоняет кандидат всякий раз, когда "
          "целевая модель присваивает ему меньшую вероятность, чем черновая. Однако на "
          "практике многие такие токены семантически приемлемы, и их отклонение лишь "
          "снижает скорость. Нечёткая (ослабленная) верификация (fuzzy verification) "
          "смягчает условие принятия, вводя порог верификации (verification threshold) "
          "T ≥ 1: кандидат принимается, если"),
        EQ(mrow(
            mfrac(mrow("p", mdelim(msub("y", "i"))),
                  mrow("q", mdelim(msub("y", "i")))),
            op(" ≥ "), mfrac("1", "T"),
        ), "1.4"),
        WHERE("где", [
            "T ≥ 1 — порог верификации (множитель допуска);",
            "при T = 1 правило (1.4) совпадает с точным условием, и SD остаётся точным;",
            "при T > 1 допускается принятие токенов, недооценённых целевой моделью.",
        ]),
        P("Тем самым порог T непрерывно интерполирует между точным спекулятивным "
          "декодированием (T = 1) и всё более агрессивным приёмом черновых токенов "
          "(T → ∞). При T > 1 метод становится приближённым (lossy): распределение "
          "порождаемых токенов отклоняется от распределения целевой модели. Именно в "
          "такой скалярной форме порог приёма введён в методе Fuzzy SD [7]; его удобство "
          "для настоящей работы в том, что единственная вещественная величина T "
          "полностью параметризует ось верификации и потому естественно включается в "
          "пространство действий управляющего процесса (раздел 1.5 и далее)."),

        H3("1.3.2 Компромисс между пропускной способностью и качеством"),
        P("Порог T задаёт прямой компромисс между пропускной способностью и качеством. "
          "Чем больше T, тем выше доля принятых токенов α, а значит, согласно (1.3), и "
          "число токенов за цикл — пропускная способность растёт. Одновременно растёт и "
          "доля токенов, в которых система доверилась черновой модели вопреки целевой, — "
          "качество ответа отклоняется от эталонного. На «лёгких» участках "
          "последовательности (типовые окончания, синтаксически предопределённые "
          "позиции) ослабление проверки почти безвредно; на «трудных» (выбор числа в "
          "арифметической задаче, начало нового шага рассуждения) то же ослабление "
          "может изменить смысл ответа. Качество, таким образом, нелинейно зависит от "
          "доли принятий, и эта зависимость неоднородна по ходу генерации."),
        P("Множество достижимых пар «пропускная способность — качество» при изменении T "
          "образует границу Парето: ни одну из двух величин нельзя улучшить, не ухудшив "
          "другую. Теоретическую структуру этой границы для семейств методов с "
          "параметризованным ослаблением верификации исследовали Yin и соавторы [18], "
          "показавшие, что при разумных предположениях фронт монотонен. Настоящая работа "
          "опирается на этот результат дважды: как на язык описания компромисса "
          "(раздел 4) и как на ориентир для скаляризации двухкритериальной задачи "
          "управления (раздел 2)."),

        H3("1.3.3 Ограниченность фиксированных гиперпараметров"),
        P("И длина черновика γ, и порог T в большинстве реализаций задаются "
          "фиксированными значениями, подобранными по усреднённому поведению на "
          "валидационной выборке. Между тем из изложенного следует, что оптимальные γ и "
          "T зависят от текущего состояния генерации. Когда черновая модель уверена и "
          "согласована с целевой, выгодны длинный черновик и мягкий порог; когда "
          "уверенность мала, разумнее короткий черновик и строгая проверка, чтобы не "
          "тратить проходы впустую и не портить качество. Любое фиксированное значение "
          "есть компромисс, не оптимальный ни для одного конкретного состояния. Это "
          "наблюдение и мотивирует переход к адаптивному управлению, в котором γ и T "
          "выбираются как функции состояния."),

        H2("1.4 Обзор современных методов спекулятивного декодирования"),
        P("Развитие спекулятивного декодирования в 2023–2026 годах удобно разложить по "
          "трём направлениям: совершенствование правила приёма, конструирование более "
          "точной черновой модели и адаптивное управление гиперпараметрами цикла. Ниже "
          "методы рассмотрены в этом порядке, после чего сведены в таблицу 1.1 по двум "
          "осям управления, существенным для настоящей работы, — длине черновика и "
          "критерию верификации."),
        H3("1.4.1 Обучаемая верификация значимых токенов"),
        P("Первое направление заменяет вероятностное правило (1.2) обучаемым критерием, "
          "оценивающим не числовое расхождение моделей, а его значимость для итогового "
          "ответа. В judge decoding роль верификатора играет классификатор, принимающий "
          "черновой токен, если тот семантически допустим, даже при низком отношении "
          "p / q. Метод AutoJudge [6] доводит эту идею до автоматического обучения: "
          "метки добываются без ручной разметки по эквивалентности ответов на наборе "
          "GSM8K [5] — токен считается незначимым, если его замена не меняет "
          "правильность решения, — после чего на размеченных примерах обучается "
          "логистическая регрессия с калибровкой на целевую полноту. На инференсе "
          "классификатор пропускает «незначимые» расхождения и повышает долю принятий; "
          "в собственных измерениях (раздел 4) на паре 7B/1.5B AutoJudge достигает "
          "61,7 % exact match. Родственный SelfJudge устраняет отдельную фазу обучения, "
          "используя в роли judge саму черновую модель ценой дополнительных проходов. "
          "Общее свойство группы — высокий эффективный acceptance rate при приближённом "
          "(lossy) характере генерации и, как правило, потребности в обучении отдельного "
          "верификатора под каждую задачу."),
        H3("1.4.2 SpecExec и точная древовидная спекуляция"),
        P("Метод SpecExec [15] увеличивает число подтверждаемых за цикл токенов, "
          "переходя от линейного черновика к дереву кандидатов. Черновая модель строит "
          "префиксное дерево наиболее вероятных продолжений, целевая модель проверяет "
          "все ветви одним проходом с переиспользованием KV-кэша вдоль рёбер дерева, "
          "после чего выбирается наилучший допустимый путь. За счёт пакетного внимания "
          "верификация дерева стоит примерно столько же, сколько верификация линейной "
          "цепочки той же глубины, тогда как ожидаемое число принятых токенов выше. "
          "SpecExec сохраняет точное распределение целевой модели и особенно эффективен "
          "при больших бюджетах параллелизма, однако управляет структурой дерева, а не "
          "порогом приёма, и потому ортогонален оси верификации."),
        H3("1.4.3 Облегчённые черновые механизмы: Medusa, EAGLE и родственные"),
        P("Третья группа улучшает не правило проверки, а сам механизм выдвижения "
          "гипотез, повышая согласованность α при низкой стоимости c. В методе "
          "Medusa [3] к целевой модели пристраивается несколько дополнительных голов "
          "(decoding heads), параллельно предсказывающих токены на несколько позиций "
          "вперёд; их кандидаты объединяются древовидным вниманием. EAGLE [11] выполняет "
          "авторегрессию не на уровне токенов, а на уровне скрытых представлений "
          "(hidden states) целевой модели, что заметно уменьшает дрейф распределений "
          "между черновиком и целью. К этому же направлению относятся Kangaroo, где "
          "черновиком служит подмножество слоёв целевой модели с лёгким адаптером, и "
          "LayerSkip, реализующий self-speculative decoding через ранний выход из "
          "промежуточных слоёв. Все они повышают α, но требуют специального обучения и "
          "не вводят управляемого порога качества. По отношению к настоящей работе эти "
          "методы ортогональны: предложенное управление парой (длина, порог) применимо "
          "поверх любой из таких черновых моделей."),
        P("Перечисленные подходы различаются балансом между ростом согласованности α и "
          "ростом относительной стоимости c. Medusa почти не увеличивает c (головы дёшевы), "
          "но прирост α ограничен независимостью голов; EAGLE на уровне скрытых "
          "представлений достигает большего α ценой обучения авторегрессионной черновой "
          "сети, а версия EAGLE-3 дополнительно улучшает обучение и архитектуру по сравнению "
          "с ранними EAGLE-1 и EAGLE-2. Kangaroo и LayerSkip переиспользуют слои самой "
          "целевой модели, снижая c за счёт более тесной связи с целью. Для настоящей работы "
          "существенно, что любой из этих механизмов лишь смещает рабочую точку (α, c), не "
          "затрагивая управление парой (длина, порог), и потому сочетается с предлагаемым "
          "методом, а не конкурирует с ним."),
        H3("1.4.4 Адаптивный выбор длины черновика"),
        P("Ближайшее к настоящей работе направление управляет длиной черновика γ, "
          "сохраняя строгое правило приёма. В методе SpecDec++ [8] решение «продолжать "
          "ли черновик» формулируется как задача оптимальной остановки: обучаемый "
          "классификатор предсказывает вероятность принятия следующего токена по "
          "скрытым представлениям черновой модели, и генерация прерывается при падении "
          "этой вероятности ниже порога. Методы SVIP и AdaEDL опираются на онлайн-оценку "
          "энтропии чернового распределения — высокая энтропия служит предиктором низкой "
          "вероятности принятия; по существу оба реализуют одномерный пороговый критерий "
          "в координатах энтропии. DISCO выбирает длину по контекстным эвристикам без "
          "обучаемой политики, а BanditSpec трактует выбор γ как задачу о многоруком "
          "бандите с наградой в виде числа принятых токенов за единицу времени. Все эти "
          "методы адаптируют ровно одну ось — длину γ — при фиксированном (как правило, "
          "точном) правиле проверки, и потому их выигрыш ограничен сверху потолком 1/c "
          "из подраздела 1.2.3."),
        H3("1.4.5 Ослабленная адаптивная верификация и фронт Парето"),
        P("Зеркально к предыдущей группе действуют методы, фиксирующие длину черновика "
          "и управляющие осью верификации. Fuzzy SD [7] вводит скалярный порог приёма T "
          "(подраздел 1.3.1), переводящий метод в lossy-режим; MARS обобщает эту идею "
          "через matched acceptance rule, настраивая степень ослабления по локальным "
          "признакам распределения целевой модели. MARS — наиболее близкий к настоящей "
          "работе метод по оси верификации; принципиальное отличие предлагаемого подхода "
          "в том, что порог T оптимизируется не изолированно, а совместно с длиной γ. "
          "Теоретическую рамку для всей группы задаёт результат Yin и соавторов [18] о "
          "структуре границы Парето между ускорением и расхождением распределений: при "
          "разумных предположениях фронт монотонен, что и обосновывает скаляризацию "
          "двухкритериальной задачи через множитель Лагранжа в разделе 2."),
        H3("1.4.6 Систематизация и выявленный пробел"),
        P("Рассмотренные методы удобно классифицировать по двум независимым осям "
          "управления — характеру выбора длины черновика (фиксированная, эвристическая, "
          "обучаемая) и строгости верификации (строгая, ослабленная фиксированная, "
          "ослабленная адаптивная). Соответствующая систематизация приведена в "
          "таблице 1.1."),
        TBL_CAPTION("Таблица 1.1 — Систематизация методов спекулятивного декодирования "
                    "по осям управления длиной черновика и критерием верификации"),
        TBL(
            [["Длина \\ Верификация", "Строгая (MRS)",
              "Ослабленная фиксированная", "Ослабленная адаптивная"],
             ["Фиксированная", "SD [10, 4]",
              "Judge Decoding, AutoJudge [6], SelfJudge, Fuzzy SD [7]", "MARS"],
             ["Эвристическая", "DISCO", "—", "—"],
             ["Обучаемая / адаптивная", "SpecDec++ [8], SVIP, AdaEDL, BanditSpec",
              "—", "JointAdaSpec (наст. работа)"]],
            [2154, 1500, 3200, 2500],
            align=["left", "center", "left", "center"],
        ),
        P("Таблица 1.1 делает пробел явным. Ячейка на пересечении адаптивного управления "
          "длиной и ослабленной адаптивной верификации остаётся незаполненной: ни одна "
          "из рассмотренных работ не управляет обеими осями совместно как связанными "
          "переменными единой оптимизационной задачи. Частные комбинации встречаются "
          "(например, адаптивную длину применяют поверх фиксированного порога Fuzzy SD), "
          "однако такое каскадное сочетание не обеспечивает совместного оптимума, "
          "поскольку оптимальное значение одного параметра функционально зависит от "
          "другого. Именно эту нишу занимает предлагаемый метод JointAdaSpec, причём, в "
          "отличие от AutoJudge, Medusa и EAGLE, он не требует обучения нейросетевых "
          "модулей — управляющая политика хранится в виде таблицы."),

        H2("1.5 Марковские процессы принятия решений как аппарат адаптивного управления"),
        H3("1.5.1 Определение MDP и уравнение оптимальности Беллмана"),
        P("Естественным формализмом для последовательного выбора действий под "
          "неопределённостью служит марковский процесс принятия решений (Markov "
          "decision process, MDP) [12]. MDP задаётся кортежем (S, A, P, r, γ), где S — "
          "множество состояний, A — множество действий, P(s′ | s, a) — вероятность "
          "перехода в s′ при выборе a в s, r(s, a) — мгновенная награда, а γ ∈ [0, 1) — "
          "коэффициент дисконтирования, придающий больший вес близким во времени "
          "наградам. Политика π сопоставляет состоянию действие; её ценность есть "
          "математическое ожидание дисконтированной суммы наград при следовании π "
          "из состояния s. Оптимальная политика максимизирует ценность во всех "
          "состояниях, а её функция ценности удовлетворяет уравнению оптимальности "
          "Беллмана:"),
        EQ(mrow(
            msup("V", "*"), mdelim("s"), op(" = "),
            msub(op("max"), mrow("a", op(" ∈ "), "A")),
            mdelim(mrow(
                "r", mdelim(mrow("s", op(", "), "a")), op(" + "), "γ",
                mnary("∑", "s′", None,
                      mrow("P", mdelim(mrow("s′", op(" | "), "s", op(", "), "a")),
                           " ", msup("V", "*"), mdelim("s′"))),
            ), "[", "]"),
        ), "1.5"),
        WHERE("где", [
            "V*(s) — оптимальная ценность состояния s;",
            "r(s, a) — награда за действие a в состоянии s;",
            "γ — коэффициент дисконтирования;",
            "P(s′ | s, a) — вероятность перехода в состояние s′.",
        ]),
        P("Удобно ввести функцию ценности действия (Q-функцию): величина Q*(s, a) "
          "складывается из мгновенной награды r(s, a) и дисконтированной ожидаемой "
          "ценности следующего состояния. Оптимальное действие в состоянии s доставляет "
          "максимум Q*(s, a) по действиям a; тем самым, зная V*, политику извлекают "
          "жадно. Применительно "
          "к спекулятивному декодированию состояние описывает текущую ситуацию генерации, "
          "а действие — выбор длины черновика и порога, что и превращает задачу "
          "адаптивного управления в задачу решения MDP. Оговорим коллизию обозначений: "
          "символ γ традиционно используют и для длины черновика (разделы 1.2–1.4), и для "
          "коэффициента дисконтирования (настоящий раздел). Во избежание неоднозначности "
          "при построении MDP-формулировки JointAdaSpec в разделе 2 коэффициент "
          "дисконтирования переобозначается через λ, а γ сохраняется за длиной черновика."),
        P("Качество MDP-модели целиком определяется выбором состояния: оно должно быть "
          "достаточно информативным, чтобы оптимальное действие зависело лишь от него "
          "(свойство марковости), и достаточно компактным, чтобы пространство S "
          "оставалось малым и допускало точное решение. В разделе 2 состояние "
          "составляется из трёх наблюдаемых на каждом цикле признаков — энтропии "
          "чернового распределения как меры уверенности черновика, накопленного "
          "расхождения Кульбака — Лейблера между черновым и целевым распределениями как "
          "меры их рассогласования и позиции внутри текущего блока, — и каждый из них "
          "дискретизируется в конечное число интервалов. Такой выбор делает состояние "
          "наблюдаемым в ходе обычного цикла SD без дополнительных вычислений."),
        H3("1.5.2 Метод value iteration: сходимость и вычислительная сложность"),
        P("Стандартный способ найти V* — итерация по ценности (value iteration), "
          "представляющая собой последовательное применение оператора Беллмана:"),
        EQ(mrow(
            msub("V", mrow("k", op("+"), "1")), mdelim("s"), op(" = "),
            msub(op("max"), mrow("a", op(" ∈ "), "A")),
            mdelim(mrow(
                "r", mdelim(mrow("s", op(", "), "a")), op(" + "), "γ",
                mnary("∑", "s′", None,
                      mrow("P", mdelim(mrow("s′", op(" | "), "s", op(", "), "a")),
                           " ", msub("V", "k"), mdelim("s′"))),
            ), "[", "]"),
        ), "1.6"),
        WHERE("где", [
            "V_k — приближение функции ценности на итерации k.",
        ]),
        P("Оператор Беллмана является сжатием с коэффициентом γ в равномерной норме, "
          "поэтому итерации (1.6) сходятся к единственной неподвижной точке V* "
          "геометрически — с линейной скоростью, определяемой коэффициентом сжатия "
          "γ [12]. Метод восходит к "
          "динамическому программированию [2]; одна итерация требует O(|S| · |A|) "
          "операций при разреженной функции переходов, а число итераций до достижения "
          "точности ε составляет порядка log(1/ε) / (1 − γ). При конечных и невысоких по "
          "размерности S и A он даёт точное решение без аппроксимации функции ценности, "
          "что выгодно отличает его от градиентных методов обучения с подкреплением [14]. "
          "Качество табличного решения, однако, ограничено точностью оценки P и r по "
          "конечной выборке: оценки выборочной сложности для подобных оценщиков, "
          "построенных по выборке из генеративной модели, известны [1] и используются во "
          "втором разделе при обосновании теоремы о точности оценщика переходов."),
        H3("1.5.3 Монотонные и пороговые политики"),
        P("Для многих прикладных MDP оптимальная политика имеет регулярную структуру: "
          "если состояния и действия упорядочены, а награда и переходы удовлетворяют "
          "условиям монотонности и супермодулярности [16], то оптимальное действие "
          "монотонно по состоянию, а сама политика приобретает пороговый вид — действие "
          "меняется при пересечении некоторой границы в пространстве состояний [12]. "
          "Пороговые политики удобны и вычислительно, и интерпретационно; именно к ним "
          "сводятся одномерные адаптивные методы из подраздела 1.4.4: SVIP и AdaEDL "
          "реализуют пороговое правило в координате энтропии чернового распределения, "
          "SpecDec++ — в координате обучаемой оценки вероятности принятия. Настоящая "
          "работа обобщает эту одномерную пороговую структуру на двумерное управление "
          "парой (длина, порог). Вместе с тем выполнение условий супермодулярности не "
          "гарантировано: при их нарушении пороговая — а значит, и каскадная, "
          "рассматривающая оси по очереди — структура перестаёт быть оптимальной. "
          "Проверка этих условий и количественная оценка последствий их нарушения "
          "составляют предмет теоретического анализа во втором разделе."),

        H2("1.6 Исследовательский пробел и постановка задачи"),
        H3("1.6.1 Ограничения существующих адаптивных методов"),
        P("Проведённый обзор позволяет точно сформулировать пробел. Методы адаптивной "
          "длины (SpecDec++ и родственные) управляют осью γ при фиксированном правиле "
          "проверки; методы ослабленной верификации (нечёткое SD, AutoJudge) управляют "
          "осью качества при фиксированной или эвристической длине. Ни одна из работ не "
          "рассматривает γ и T как связанные переменные единой оптимизационной задачи. "
          "Между тем оси взаимозависимы: выгодная длина черновика зависит от того, "
          "насколько строг порог приёма, и наоборот. Последовательное (каскадное) "
          "применение двух одномерных правил — выбрать сначала длину, затем порог (или "
          "наоборот) — кажется естественным, но не имеет гарантий оптимальности именно "
          "из-за этой взаимозависимости."),
        H3("1.6.2 Идея совместного адаптивного управления"),
        P("Предлагаемый подход состоит в том, чтобы трактовать выбор пары (длина "
          "черновика, порог верификации) на каждом цикле декодирования как действие в "
          "марковском процессе принятия решений, состояние которого описывает текущую "
          "ситуацию генерации (уверенность черновика, накопленное расхождение моделей, "
          "позицию в блоке). Оптимальная политика такого MDP по построению учитывает "
          "связь осей: она назначает совместно оптимальную пару (γ, T) для каждого "
          "состояния. Каскадные эвристики при этом оказываются частным случаем — "
          "подмножеством совместных политик, — что позволяет строго сравнить их с "
          "совместным решением. Такой взгляд переводит инженерную задачу подбора "
          "гиперпараметров в задачу оптимального управления с доказуемыми свойствами."),
        H3("1.6.3 Формальная постановка задачи совместной оптимизации"),
        P("Формально требуется построить дискретный MDP (S, A, P, r, γ), в котором "
          "состояние агрегирует наблюдаемые признаки цикла спекулятивного декодирования, "
          "действие задаёт пару (длина черновика, уровень порога верификации), награда "
          "поощряет число принятых токенов и штрафует затраченное время и потерю "
          "качества, а решение уравнения (1.5) методом (1.6) даёт совместно оптимальную "
          "управляющую политику. От искомого решения требуется: сохранять принцип "
          "спекулятивного декодирования (управляемое отклонение от целевого "
          "распределения через порог T); допускать точное и воспроизводимое вычисление "
          "политики без обучения нейросетевых модулей; и поддаваться теоретическому "
          "анализу — в части корректности оценки параметров, влияния штрафа качества и "
          "соотношения совместной и каскадной политик. Построению такого MDP, его "
          "решению и теоретическому анализу посвящён второй раздел работы."),
        P("Двухкритериальность задачи — одновременная максимизация пропускной "
          "способности и качества — разрешается скаляризацией: потеря качества входит в "
          "награду как аддитивный штраф с неотрицательным множителем κ. Изменение κ "
          "перемещает оптимум вдоль границы Парето из подраздела 1.3.2: при κ = 0 "
          "максимизируется только скорость, при росте κ растёт вес качества. Тем самым "
          "единая постановка охватывает целое семейство компромиссов, а не одну "
          "фиксированную точку, что отличает её от методов с заранее заданным порогом."),
    ])


# --------------------- Разделы 2-4: заголовки + аннотированный план ---------------------
def _plan(text):
    return P(text, indent=True, jc="both", italic=True)


RAZDEL2 = block([
    H1("2 Метод JointAdaSpec и его теоретический анализ"),
    P("Первый раздел установил исследовательский пробел: ни один из существующих методов "
      "не управляет длиной черновика и порогом верификации совместно как связанными "
      "переменными единой оптимизационной задачи. Настоящий раздел закрывает этот пробел. "
      "Сначала строится дискретный марковский процесс принятия решений, состояние которого "
      "агрегирует наблюдаемые признаки цикла спекулятивного декодирования, а действие "
      "задаёт пару (длина, порог) (2.1); затем проводится теоретический анализ свойств "
      "оптимальной политики (2.2); наконец, описываются алгоритм решения MDP методом "
      "итерации по ценности (2.3) и алгоритм инференса с выученной политикой (2.4). Во "
      "всём разделе коэффициент дисконтирования обозначается через λ, а символ γ сохранён "
      "за длиной черновика (см. замечание об обозначениях в подразделе 1.5.1)."),

    H2("2.1 MDP-формулировка совместной оптимизации"),
    H3("2.1.1 Пространство состояний и его дискретизация"),
    P("Состояние MDP должно быть достаточно информативным, чтобы оптимальная пара (длина, "
      "порог) зависела лишь от него (свойство марковости), и достаточно компактным для "
      "точного табличного решения. Эти требования удовлетворяет тройка наблюдаемых на "
      "каждом шаге величин: энтропия чернового распределения, расхождение чернового и "
      "целевого распределений и позиция в текущем блоке."),
    P("Энтропия чернового распределения H. Для текущей позиции черновика энтропия Шеннона "
      "распределения q вычисляется как"),
    EQ(mrow("H", mdelim(msub("q", "i")), op(" = "), op("−"),
            mnary("∑", mrow("x", op(" ∈ "), "V"), None,
                  mrow(msub("q", "i"), mdelim("x"), " ", op("log"), " ",
                       msub("q", "i"), mdelim("x")))), "2.1"),
    WHERE("где", [
        (mrow(msub("q", "i")), " — распределение черновой модели на позиции i;"),
        (mrow("V"), " — словарь токенов."),
    ]),
    P("Энтропия H служит индикатором локальной неуверенности черновой модели: при низкой "
      "энтропии черновик уверен в следующем токене, что повышает шансы согласия с целевой "
      "моделью и принятия токена. Именно этот признак лежит в основе одномерных методов "
      "SVIP и AdaEDL (подраздел 1.4.4); его включение сохраняет содержательную "
      "преемственность с литературой. В работе используется верхняя граница H_max = 6,0 "
      "нат, покрывающая свыше 99 % наблюдаемых на трейсах значений."),
    P("Дивергенция Кульбака — Лейблера K. Расхождение между черновым и целевым "
      "распределениями измеряется дивергенцией"),
    EQ(mrow("K", mdelim(mrow(msub("q", "i"), op(", "), msub("p", "i"))), op(" = "),
            op("KL"), mdelim(mrow(msub("q", "i"), op(" ‖ "), msub("p", "i"))), op(" = "),
            mnary("∑", mrow("x", op(" ∈ "), "V"), None,
                  mrow(msub("q", "i"), mdelim("x"), " ", op("log"), " ",
                       mfrac(mrow(msub("q", "i"), mdelim("x")),
                             mrow(msub("p", "i"), mdelim("x")))))), "2.2"),
    P("Если энтропия H характеризует абсолютную неуверенность черновой модели, то K "
      "характеризует именно расхождение распределений. Высокое K при низком H отвечает "
      "ситуации «черновик уверен, но неверно»: точное правило (1.2) отвергнет такой токен "
      "с высокой вероятностью. Совместное наблюдение пары (H, K) несёт существенно больше "
      "информации о предстоящем принятии, чем любая одномерная статистика. Использовано "
      "направление KL(q ‖ p), а не обратное: кандидаты выбираются из q, поэтому расхождение "
      "«q относительно p» точнее предсказывает поведение верификации. Верхняя граница "
      "K_max = 8,0 нат."),
    P("Совместное наблюдение пары (H, K) различает четыре качественно разных режима. "
      "Низкие H и K отвечают согласию уверенных моделей — выгоден длинный черновик с "
      "мягким порогом. Высокое H при низком K — обе модели неуверены, но их распределения "
      "близки. Низкое H при высоком K — черновик уверен, но расходится с целью: типичная "
      "ловушка, в которой ослаблять проверку опасно. Высокие H и K — общая "
      "неопределённость, при которой осторожнее короткий черновик. Ни одна одномерная "
      "статистика эти режимы не разделяет, что и обосновывает двумерное состояние."),
    P("Позиция в черновике k. Дискретный счётчик уже сгенерированных в текущем блоке "
      "токенов необходим по двум содержательным причинам. Маргинальная выгода от "
      "продолжения черновика падает с ростом k в силу геометрического убывания вероятности "
      "достичь позиции k + 1, а верхняя граница γ_max создаёт жёсткое ограничение, которое "
      "политика обязана учитывать, чтобы не выродиться в правило «продолжать всегда». Без k "
      "процесс перестаёт быть марковским. Принято γ_max = 8."),
    P("Итоговое пространство состояний есть произведение"),
    EQ(mrow("S", op(" = "),
            mdelim(mrow("0", op(", "), msub("H", "max")), "[", "]"), op(" × "),
            mdelim(mrow("0", op(", "), msub("K", "max")), "[", "]"), op(" × "),
            mdelim(mrow("0", op(", "), "1", op(", "), "…", op(", "), msub("γ", "max")),
                   "{", "}")), "2.3"),
    P("Для табличного решения непрерывные оси H и K дискретизируются равномерной сеткой "
      "20 × 20, что вместе с девятью значениями k ∈ {0, …, 8} даёт |S| = 20 · 20 · 9 = "
      "3600 состояний. Альтернативные признаки отвергнуты по содержательным причинам: "
      "top-1 вероятность и margin черновика сильно коллинеарны энтропии H; скрытое "
      "представление черновой модели имеет размерность в тысячи и исключает табличный MDP; "
      "глобальный счётчик принятых токенов с начала генерации не является марковским."),

    H3("2.1.2 Пространство действий"),
    P("Действие на каждом шаге двумерно и отражает два независимых рычага управления — "
      "продолжать ли генерацию черновика и какой порог применить при верификации "
      "следующего токена. Действие по оси длины бинарно:"),
    EQ(mrow(msub("a", "length"), op(" ∈ "), msub("A", "length"), op(" = "),
            mdelim(mrow(op("stop"), op(", "), op("continue")), "{", "}")), "2.4"),
    P("Действие stop останавливает генерацию черновика и переводит цикл к фазе "
      "верификации; continue порождает следующий черновой токен. При k = γ_max действие "
      "принудительно равно stop. Действие по оси верификации — скалярный порог T из "
      "семейства нечёткой верификации (1.4); непрерывный диапазон [1, T_max] дискретизуется "
      "по геометрической сетке"),
    EQ(mrow(msub("a", "verif"), op(" ∈ "), msub("A", "verif"), op(" = "),
            mdelim(mrow("1", op(", "), msubsup("T", "max", "1/M"), op(", "),
                        msubsup("T", "max", "2/M"), op(", "), "…", op(", "),
                        msub("T", "max")), "{", "}")), "2.5"),
    P("Геометрическая сетка выбрана потому, что влияние порога на вероятность принятия "
      "мультипликативно: переход от T = 1 к T = 1,1 и переход от T = 5 к T = 5,5 "
      "содержательно различны. При T_max = 4,0 и M = 7 получаются восемь уровней "
      "{1,00; 1,22; 1,49; 1,82; 2,22; 2,71; 3,30; 4,00}; ограничение T_max = 4 обусловлено "
      "тем, что при больших порогах распределение порождаемых токенов отклоняется от "
      "целевого настолько, что деградация качества становится неприемлемой. Совместное "
      "пространство действий есть произведение осей"),
    EQ(mrow("A", op(" = "), msub("A", "length"), op(" × "), msub("A", "verif"),
            op(", "), mdelim("A", "|", "|"), op(" = "), "2", op(" · "), "8", op(" = "),
            "16"), "2.6"),
    P("Размер |A| = 16 достаточен для разрешения совместной политики при стоимости одной "
      "итерации решения порядка O(|S| · |A|), что составляет несколько секунд на одном "
      "процессоре."),

    H3("2.1.3 Функция переходов и функция награды"),
    P("Функция переходов P(s′ | s, a) не допускает аналитической формы: признаки нового "
      "состояния (H′, K′) функционально определяются распределениями q и p, которые сами "
      "суть результат прямого прохода через многослойные сети. Поэтому вместо аналитической "
      "формы используется эмпирическая оценка, получаемая из трейсов спекулятивного "
      "декодирования (процедура описана в подразделе 2.3). Содержательно различимы переходы "
      "после continue (порождается следующий черновой токен ценой одного вызова черновой "
      "модели) и после stop (выполняется верификация ценой одного вызова целевой модели и "
      "начинается новый блок)."),
    P("Формально различимы четыре типа переходов: продолжение черновика (continue при "
      "k < γ_max), его остановка (stop), принудительная остановка на границе (continue "
      "при k = γ_max сводится к stop) и переход к началу нового блока после верификации. "
      "Существенное для теоремы 2.3 условие стохастической монотонности (C2) содержательно "
      "означает, что сдвиг текущего состояния к большим (H, K) делает более вероятным и "
      "большее (H′, K′) следующего состояния. Эта монотонность отражает локальную "
      "автокоррелированность сложности текста: высокая неуверенность черновика в текущей "
      "позиции свидетельствует о локально сложном участке последовательности. Эмпирическая "
      "проверка условий монотонности вынесена в подраздел 4.x."),
    P("Функция награды выводится из операционной цели метода — максимума числа принятых "
      "токенов в единицу времени при ограничении на качество. Обозначив через η пропускную "
      "способность (принятых токенов в секунду), а через D — расхождение распределения "
      "порождаемых токенов с распределением целевой модели, содержательную задачу "
      "записывают как условную оптимизацию"),
    EQ(mrow("η", mdelim("π"), op(" → "), op("max"), op(",      "),
            "D", mdelim("π"), op(" ≤ "), "ε"), "2.7"),
    P("где ε — допустимый уровень потери качества. Применяя метод множителей Лагранжа к "
      "ограничению (2.7), переходят к безусловной задаче с пошаговой наградой"),
    EQ(mrow("r", mdelim(mrow("s", op(", "), "a")), op(" = "),
            "η", mdelim(mrow("s", op(", "), "a")), op(" − "), "κ", op(" · "),
            "D", mdelim(mrow("s", op(", "), "a"))), "2.8"),
    P("в которой множитель κ ≥ 0 задаёт цену единицы потери качества и тем самым выбирает "
      "точку на границе Парето из подраздела 1.3.2: при κ = 0 максимизируется чистая "
      "скорость, с ростом κ растёт вес качества. Оптимальная политика есть решение "
      "уравнения оптимальности Беллмана с этим вознаграждением"),
    EQ(mrow(msup("V", "*"), mdelim("s"), op(" = "),
            msub(op("max"), mrow("a", op(" ∈ "), "A")),
            mdelim(mrow("r", mdelim(mrow("s", op(", "), "a")), op(" + "), "λ",
                        mnary("∑", "s′", None,
                              mrow("P", mdelim(mrow("s′", op(" | "), "s", op(", "), "a")),
                                   " ", msup("V", "*"), mdelim("s′")))), "[", "]")),
       "2.9"),
    P("Существенно, что штраф качества входит в награду через слагаемое, зависящее от "
      "действия (выбранного порога T): как показывает теорема B (подраздел 2.2.2), штраф, "
      "зависящий только от состояния, не влияет на оптимальную политику. Именно "
      "связанность награды с действием делает управление качеством содержательным."),
    P("Конкретная форма награды в реализации (подраздел 3.3) такова: число принятых токенов "
      "поощряется, затраченное время штрафуется небольшим множителем c_time = 0,01, а потеря "
      "качества — слагаемым с множителем κ. Аддитивный штраф качества, согласующийся с "
      "теоремой B, параметризуется коэффициентами при дивергенции K и позиции k (поля "
      "quality_risk_K и quality_risk_k конфигурации); по умолчанию оба равны нулю, что "
      "соответствует базовой постановке (2.8). Малое c_time гарантирует, что доминирующим "
      "членом награды остаётся число принятых токенов, а штрафы лишь корректируют выбор на "
      "«трудных» состояниях."),

    H2("2.2 Теоретический анализ метода"),
    P("Теоретическая часть отвечает на три вопроса: насколько точно политику можно выучить "
      "из конечной выборки трейсов (2.2.1), как корректно ввести штраф качества (2.2.2) и в "
      "каком отношении находятся совместная и каскадная политики (2.2.3–2.2.5). "
      "Формулировки теорем приводятся с указанием идеи доказательства; полные выкладки "
      "вынесены в приложение А."),

    H3("2.2.1 Выборочная сложность оценщика переходов (теорема A)"),
    P("Табличная политика вычисляется по эмпирической оценке функции переходов, поэтому "
      "качество решения ограничено точностью этой оценки на конечной выборке. Следующая "
      "теорема даёт равномерную по состояниям оценку отклонения выученной функции ценности "
      "от истинной."),
    PM(["Теорема A (выборочная сложность). Пусть ", macc("P"),
        " — оценка функции переходов по выборке, в которой каждая пара (состояние, "
        "действие) наблюдалась не менее ", msub("n", "min"),
        " раз, и пусть применяется сглаживание Лапласа с параметром α. Тогда с "
        "вероятностью не менее 1 − δ"], indent=True),
    EQ(mrow(
        msub(mdelim(mrow(macc("V"), op(" − "), msup("V", "*")), "‖", "‖"), "∞"),
        op(" ≤ "),
        mfrac(mrow("2", "R"), msup(mdelim(mrow("1", op(" − "), "λ")), "2")),
        msqrt(mfrac(mrow("2", " ", op("ln"),
                         mdelim(mfrac(mrow("2", mdelim("S", "|", "|"),
                                           mdelim("A", "|", "|")), "δ"))),
                    msub("n", "min"))),
        op(" + "),
        mfrac(mrow("α", mdelim("S", "|", "|"), "R"),
              mrow(mdelim(mrow("1", op(" − "), "λ")),
                   mdelim(mrow(msub("n", "min"), op(" + "), "α"))))), "2.10"),
    WHERE("где", [
        (mrow("R"), " — верхняя граница модуля награды;"),
        (mrow("λ"), " — коэффициент дисконтирования;"),
        (mrow("δ"), " — уровень доверия."),
    ]),
    P("Первое слагаемое — статистическая ошибка (концентрация Хёфдинга по выборке "
      "переходов), убывающая как корень из числа наблюдений; второе — смещение, вносимое "
      "сглаживанием Лапласа и убывающее с ростом n_min. Практический вывод: для контроля "
      "ошибки достаточно гарантировать минимальную заполненность каждой пары "
      "(состояние, действие) в трейсах, что и закладывается в процедуру сбора (2.3)."),
    P("Идея доказательства. Для фиксированной пары (s, a) оценка функции переходов есть "
      "нормированный счётчик наблюдений, и по неравенству Хёфдинга её отклонение от "
      "истинного распределения убывает как корень из числа наблюдений. Объединение оценок "
      "по всем |S| · |A| парам (union bound) даёт логарифмический множитель ln(2|S| · "
      "|A| / δ). Перенос ошибки оценки переходов на ошибку функции ценности опирается на "
      "λ-сжимаемость оператора Беллмана и вносит множитель 1 / (1 − λ)²: одна степень "
      "отвечает за горизонт планирования, вторая — за накопление ошибки вдоль итераций. "
      "Смещение от сглаживания Лапласа учитывается отдельным слагаемым, линейным по α и "
      "убывающим с ростом n_min. Полные выкладки приведены в приложении А."),

    H3("2.2.2 Беллман-инвариантность аддитивного штрафа качества (теорема B)"),
    P("Конструкция награды (2.8) вводит штраф качества как слагаемое, зависящее от "
      "действия. Следующее утверждение объясняет, почему это необходимо, и одновременно "
      "корректирует более раннюю мультипликативную форму штрафа, нарушавшую сжимаемость "
      "оператора Беллмана."),
    PM(["Теорема B (Беллман-инвариантность). Пусть к награде добавлен штраф вида ",
        mrow("g", mdelim("s")), ", зависящий только от состояния. Тогда оптимальная "
        "политика MDP не изменяется: добавка сдвигает функцию ценности на постоянную по "
        "действиям величину и сокращается в операторе максимума. Форму оптимальной "
        "политики способен изменить лишь штраф, связанный с действием, ",
        mrow("g", mdelim(mrow("s", op(", "), "a"))), "."], indent=True),
    P("Содержательное следствие двояко. Прежде всего, штраф качества обязан зависеть от "
      "выбранного порога T — иначе, по доказанному, он не влияет на политику; это и "
      "заложено в (2.8). Кроме того, аддитивная форма штрафа сохраняет λ-сжимаемость "
      "оператора Беллмана, а с ней и гарантии сходимости итерации по ценности "
      "(теорема 1.4), тогда как мультипликативная форма их разрушала. Именно поэтому в "
      "окончательной формулировке "
      "метода используется аддитивный, связанный с действием штраф."),
    P("Идея доказательства. После подстановки штрафа g(s) в уравнение Беллмана слагаемое "
      "g(s) не зависит от действия и выносится из-под знака максимума; argmax по действиям, "
      "а с ним и оптимальная политика, не меняется, а функция ценности лишь сдвигается на "
      "дисконтированную сумму штрафов вдоль траектории. Содержательно штраф, не различающий "
      "действий, наказывает само нахождение в состоянии, а не выбор, и потому не управляет. "
      "Ранняя мультипликативная форма штрафа (множитель 1 − κD при награде) нарушала это "
      "рассуждение и вдобавок лишала оператор Беллмана λ-сжимаемости, ставя под сомнение "
      "сходимость; переход к аддитивной форме (2.8) устранил оба дефекта."),

    H3("2.2.3 Субоптимальность каскадных политик (теорема C)"),
    P("Каскадной называется политика, полученная последовательным решением двух одномерных "
      "задач — сначала по одной оси, затем по другой (например, «сначала длина, затем "
      "порог»). Множество каскадных политик является подмножеством совместных, "
      "Π_cascade ⊂ Π_joint, поскольку любое каскадное решение реализуемо и в совместном "
      "пространстве. Отсюда немедленно следует слабое доминирование."),
    PM(["Теорема 2.3 (слабое доминирование). Так как Π_cascade ⊂ Π_joint, для оптимальных "
        "функций ценности выполнено ", mrow(msub("V", "joint"), mdelim("s"), op(" ≥ "),
        msub("V", "cascade"), mdelim("s")),
        " во всех состояниях s: совместная оптимизация не может быть хуже каскадной."],
       indent=True),
    P("Каскад допускает два порядка — «длина, затем порог» и «порог, затем длина»; каждый "
      "фиксирует одну ось оптимальной при некотором фиксированном значении другой и затем "
      "оптимизирует вторую. Обе получаемые политики принадлежат Π_joint, поэтому слабое "
      "доминирование (теорема 2.3) покрывает любой из порядков, и в дальнейшем под "
      "каскадной понимается лучшая из двух."),
    P("Насколько совместная политика может превосходить каскадную, зависит от структуры "
      "MDP. Если выполнены условия монотонности и супермодулярности (C1)–(C4) — монотонность "
      "награды по (H, K), стохастическая монотонность переходов, супермодулярность награды "
      "и переходов и, главное, совместная супермодулярность действий по состоянию (C4), — то "
      "оптимальная политика имеет пороговую структуру (теорема Топкиса о монотонности "
      "решения параметрической задачи [16]) и совпадает с каскадной. При нарушении (C4) "
      "появляется зазор, ограниченный сверху мерой нарушения."),
    PM(["Теорема C (оценка субоптимальности каскада). Пусть ", msubsup("μ", "J", "*"),
        mdelim("B"), " — стационарная масса множества B состояний, в которых нарушено "
        "условие (C4). Тогда"], indent=True),
    EQ(mrow(msub("V", "joint"), mdelim("s"), op(" − "), msub("V", "cascade"), mdelim("s"),
            op(" ≤ "),
            mfrac(mrow("2", " ", msub("R", "max"), " ", msubsup("μ", "J", "*"),
                       mdelim("B")), mrow("1", op(" − "), "λ"))), "2.11"),
    P("Оценка (2.11) линейна по массе нарушения μ*_J(B): чем реже нарушается (C4), тем ближе "
      "каскадная политика к совместной. Эмпирическая проверка (подраздел 4.x) обнаруживает, "
      "однако, что на обеих рассмотренных парах моделей (C4) нарушается почти всюду — "
      "μ*_J(B) ≈ 0,90, — поэтому правая часть (2.11) оказывается слишком грубой "
      "(практически вакуумной) оценкой. Точное объяснение наблюдаемой близости совместной и "
      "каскадной политик даёт теорема D."),

    H3("2.2.4 Точный value gap совместной и каскадной политик (теорема D)"),
    P("Грубость оценки (2.11) означает не то, что зазор велик, а то, что мера нарушения — "
      "неподходящая характеристика. Точную величину зазора даёт его представление через "
      "преимущество (advantage) каскадной политики на множестве нарушения."),
    PM(["Теорема D (точный разрыв ценности). Обозначив преимущество ",
        mrow(msup("A", msub("π", "C")), mdelim(mrow("s", op(", "), "a"))),
        op(" = "), mrow(msup("Q", msub("π", "C")), mdelim(mrow("s", op(", "), "a")),
        op(" − "), msup("V", msub("π", "C")), mdelim("s")),
        " каскадной политики, для разрыва ценности справедливо точное равенство"],
       indent=True),
    EQ(mrow(msub("V", "joint"), op(" − "), msub("V", "cascade"), op(" = "),
            mfrac("1", mrow("1", op(" − "), "λ")), op(" · "),
            msub(op("E"), mrow("s", op(" ~ "), msubsup("μ", "J", "*"))),
            mdelim(mrow(msup("A", msub("π", "C")),
                        mdelim(mrow("s", op(", "), msub("π", "J"), mdelim("s")))),
                   "[", "]")), "2.12"),
    P("Эмпирически усреднённое по множеству нарушения преимущество оказывается "
      "пренебрежимо малым: его средняя величина составляет около 6 · 10⁻⁵ на паре "
      "14B/0.5B и около 10⁻³ на паре 7B/1.5B. Поэтому, несмотря на повсеместное нарушение "
      "(C4), точный разрыв ценности мал: стационарно взвешенная величина V_joint − V_cascade "
      "равна +0,005 на паре 14B/0.5B и +0,10 на паре 7B/1.5B, а слабое доминирование "
      "(теорема 2.3) выполнено точно — на 100 % состояний обеих пар. Содержательный вывод: "
      "практическое совпадение совместной и каскадной политик есть не артефакт настройки, а "
      "предсказание теории — нарушение (C4) повсеместно, но безвредно."),
    P("Идея вывода. Равенство (2.12) есть применение леммы о разности производительностей "
      "(performance-difference lemma) [9] к паре политик π_J и π_C: разность их ценностей "
      "равна дисконтированному стационарному среднему преимущества одной политики "
      "относительно другой. Поскольку вне множества B каскадная политика уже совместно "
      "оптимальна, среднее сосредоточено на B, и величину зазора определяет не размер B, а "
      "среднее преимущество на нём — которое эмпирически близко к нулю. Это и объясняет, "
      "почему грубая оценка (2.11) вакуумна, а точный зазор мал."),

    H3("2.2.5 Скаляризация Парето-фронта (теорема 2.4)"),
    P("Награда (2.8) скаляризует двухкритериальную задачу (скорость, качество) через "
      "множитель κ. Следующее утверждение связывает выбор κ с положением оптимума на "
      "границе Парето достижимых пар (η, D)."),
    PM(["Теорема 2.4 (скаляризация). Для линейной скаляризации ",
        mrow("J", op("("), "π", op(") = "), "η", mdelim("π"), op(" − "), "κ",
             " ", "D", mdelim("π")),
        " при κ ≥ 0 оптимальные политики"], indent=True),
    EQ(mrow(msubsup("π", "κ", "*"), op(" ∈ "), msub(op("argmax"), "π"),
            mdelim(mrow("η", mdelim("π"), op(" − "), "κ", " ", "D", mdelim("π")),
                   "[", "]")), "2.13"),
    P("лежат на верхней выпуклой оболочке множества достижимых пар (η, D), а перебор κ ≥ 0 "
      "обходит эту оболочку. Тем самым единая постановка охватывает целое семейство "
      "компромиссов «скорость — качество», а не одну фиксированную точку. Следует честно "
      "оговорить: строгая выпуклость достижимого множества на конечной выборке "
      "(n = 300 на значение κ) чисто не проверяется — соответствующая развёртка по κ "
      "приведена и обсуждается в подразделе 4.6."),
    P("Идея доказательства. Множество достижимых пар (η, D) по всем стохастическим "
      "стационарным политикам выпукло: рандомизация политик даёт выпуклые комбинации их "
      "операционных характеристик. Линейный функционал η − κD достигает максимума на "
      "выпуклом множестве в опорной точке его верхней границы — Парето-фронта, — причём "
      "наклон опорной прямой равен κ. Перебор κ от нуля к бесконечности обходит фронт от "
      "точки максимальной скорости к точке максимального качества; промежуточные κ дают "
      "промежуточные компромиссы."),

    H2("2.3 Алгоритм решения MDP"),
    P("Решение MDP состоит из трёх стадий: дискретизации пространства, оценки параметров из "
      "трейсов и итерации по ценности. Непрерывные оси (H, K) разбиваются равномерной "
      "сеткой 20 × 20, что задаёт индексы дискретного состояния s = (i_H, i_K, k). По "
      "набору одношаговых переходов (s, a, r, s′), собранных на удержанной части обучающих "
      "промптов, функция переходов оценивается частотно со сглаживанием Лапласа"),
    EQ(mrow(macc("P"), mdelim(mrow("s′", op(" | "), "s", op(", "), "a")), op(" = "),
            mfrac(mrow("N", mdelim(mrow("s", op(", "), "a", op(", "), "s′")), op(" + "),
                       "α"),
                  mrow("N", mdelim(mrow("s", op(", "), "a")), op(" + "), "α",
                       mdelim("S", "|", "|")))), "2.14"),
    WHERE("где", [
        (mrow("N", mdelim(mrow("s", op(", "), "a", op(", "), "s′"))),
         " — число наблюдённых переходов s → s′ при действии a;"),
        (mrow("α"), " — параметр сглаживания (псевдосчёт)."),
    ]),
    P("Сглаживание гарантирует определённость переходов для редко наблюдаемых пар и "
      "контролирует смещение согласно теореме A. Награда r(s, a) усредняется по тем же "
      "трейсам. Оптимальная функция ценности находится итерацией по ценности — повторным "
      "применением оператора Беллмана"),
    EQ(mrow(msub("V", mrow("j", op(" + "), "1")), mdelim("s"), op(" = "),
            msub(op("max"), mrow("a", op(" ∈ "), "A")),
            mdelim(mrow("r", mdelim(mrow("s", op(", "), "a")), op(" + "), "λ",
                        mnary("∑", "s′", None,
                              mrow(macc("P"),
                                   mdelim(mrow("s′", op(" | "), "s", op(", "), "a")),
                                   " ", msub("V", "j"), mdelim("s′")))), "[", "]")),
       "2.15"),
    P("Итерации продолжаются до сближения соседних приближений функции ценности по "
      "равномерной норме (ниже порога tol); при λ = 0,99 сходимость достигается за "
      "порядка 830 итераций. "
      "Благодаря разреженности функции переходов (из каждого состояния достижимо небольшое "
      "число преемников) стоимость одной итерации составляет O(|S| · |A|), а всё решение "
      "занимает секунды на одном процессоре."),
    P("Сводно процедура решения такова: 1) дискретизировать признаки (H, K) сеткой "
      "20 × 20 и сформировать индексы состояний; 2) по трейсам накопить счётчики переходов "
      "и средние награды для каждой пары (s, a); 3) оценить функцию переходов по (2.14) и "
      "положить начальное приближение ценности нулевым; 4) повторять обновление (2.15), "
      "пока соседние приближения не сблизятся по равномерной норме ниже порога tol; "
      "5) извлечь детерминированную политику по правилу (2.16). Каскадная политика "
      "получается заменой последнего шага на последовательную оптимизацию двух осей."),
    P("Совместная и каскадная политики решаются по одному и тому же набору трейсов, что "
      "обеспечивает корректность их сравнения в разделе 4. Детерминированность всего "
      "пайплайна (фиксированные начальные значения генераторов, отсутствие обучаемых "
      "нейросетевых модулей) делает выученную политику воспроизводимой по тем же входным "
      "трейсам."),

    H2("2.4 Инференс-алгоритм с выученной политикой"),
    P("На инференсе выученная политика применяется без какого-либо дополнительного обучения "
      "и без обращения к нейросетевым модулям. На каждом шаге цикла спекулятивного "
      "декодирования по текущим наблюдаемым признакам вычисляется индекс состояния s, после "
      "чего действие выбирается табличным поиском"),
    EQ(mrow("π", mdelim("s"), op(" = "), msub(op("argmax"), mrow("a", op(" ∈ "), "A")),
            " ", msup("Q", "*"), mdelim(mrow("s", op(", "), "a"))), "2.16"),
    P("за время O(1). Компонента a_length определяет, продолжать ли черновик; компонента "
      "a_verif задаёт порог T, с которым следующий токен проходит нечёткую верификацию по "
      "правилу (1.4): кандидат принимается, если p / q ≥ 1 / T. Дивергенция K, входящая в "
      "состояние, вычисляется с задержкой на один шаг (по уже наблюдённым распределениям "
      "предыдущей позиции, K_prev), что устраняет лишний прямой проход и сохраняет "
      "марковость в реализуемой форме. Накладные расходы политики — целочисленная "
      "дискретизация двух признаков и одно обращение к таблице — пренебрежимо малы на фоне "
      "прямого прохода модели."),
    P("Выученная на основной паре политика поддаётся интерпретации. В большинстве состояний "
      "она предписывает продолжать черновик и принимать токен при пороге, близком к "
      "T ≈ 1,0, то есть выполнять почти точную верификацию; более строгий порог выбирается "
      "лишь в состояниях с высоким накопленным расхождением K, где доверять черновику "
      "опасно. Таким образом, выученное управление оказывается умеренно адаптивным, а не "
      "агрессивным, что согласуется с честной картиной прироста качества из раздела 4. "
      "Соответствующая пороговая поверхность и карта вероятности продолжения приведены в "
      "приложении В (рисунки В.2 и В.3)."),
    P("Инференс-цикл на каждом блоке устроен так: 1) по текущим признакам (энтропия, "
      "задержанная дивергенция K_prev, позиция k) вычислить индекс состояния; 2) выбрать "
      "действие по правилу (2.16); 3) если выбрано continue и k < γ_max — породить "
      "следующий черновой токен и вернуться к шагу 1, иначе перейти к верификации; "
      "4) проверить блок по правилу (1.4) с порогом T = a_verif, принять согласованный "
      "префикс и пересемплировать первый отвергнутый токен; 5) обновить контекст и начать "
      "новый блок. Признаки наблюдаются попутно, без дополнительных прямых проходов."),
    P("Таким образом, метод JointAdaSpec сводит инженерную задачу подбора гиперпараметров "
      "спекулятивного декодирования к решению компактного MDP и табличному применению его "
      "политики, сохраняя принцип управляемого отклонения от целевого распределения через "
      "порог T. Детали программной реализации описанного пайплайна — сбора трейсов, оценки "
      "MDP, решения и инференса — приведены в разделе 3."),
])

def razdel3(fig3_block):
    return block([
    H1("3 Программная реализация"),
    P("Настоящий раздел описывает программную реализацию метода JointAdaSpec и сопутствующего "
      "исследовательского инструментария. Подраздел 3.1 вводит общую архитектуру комплекса и "
      "поток данных между стадиями обработки; подразделы 3.2–3.4 последовательно "
      "соответствуют трём стадиям метода — сбору трейсов, оценке параметров и решению MDP, "
      "инференсу с выученной политикой — и связывают программные модули с алгоритмами "
      "раздела 2. Подраздел 3.5 описывает реализацию базовых методов, подраздел 3.6 фиксирует "
      "используемые инструменты, средства непрерывной интеграции и механизмы "
      "воспроизводимости."),

    H2("3.1 Архитектура программного комплекса"),
    P("Программный комплекс разделён на два стека с различной зоной ответственности. "
      "Исторический стек sp_samp/ содержит эталонные реализации спекулятивного декодирования, "
      "AutoJudge [6], Top-K и SpecExec [15] и служит для сопоставления с методами из "
      "литературы. Тематический стек jointadaspec/ реализует собственно метод JointAdaSpec и "
      "оформлен в виде набора слабосвязанных подпакетов, перечисленных в таблице 3.1. "
      "Разделение на два стека позволяет развивать тематический код независимо, не нарушая "
      "ранее зафиксированные результаты сравнительных бенчмарков."),
    TBL_CAPTION("Таблица 3.1 — Подпакеты программного комплекса jointadaspec/"),
    TBL([["Подпакет", "Назначение"],
         ["core/", "вычисление признаков состояния и правило нечёткой верификации"],
         ["mdp/", "дискретизация пространства, оценка переходов, итерация по ценности"],
         ["inference/", "онлайн-декодер с выученной табличной политикой"],
         ["baselines/", "одномерные и каскадные базовые декодеры"],
         ["metrics/", "расчёт скорости, качества и границы Парето"],
         ["analysis/", "проверка эмпирических условий C1–C4 и N1–N2"],
         ["utils/", "загрузка моделей, работа с распределениями, манифесты"]],
        [2400, 6954], align=["left", "left"]),
    P("Управление экспериментом построено на конфигурационном фреймворке Hydra: пара "
      "моделей, набор данных, размер сетки состояний и гиперпараметры награды задаются "
      "декларативно в YAML-файлах каталога configs/, что исключает разнесённость параметров "
      "по исходному коду. Центральная структура параметров MDP приведена в листинге 3.1; её "
      "поля дословно соответствуют величинам раздела 2 (H_max, K_max, γ_max, сетка 20 × 20, "
      "набор порогов T, дисконт λ, множитель κ, сглаживание α)."),
    LISTING("Листинг 3.1 — Параметры табличного MDP (jointadaspec/mdp/spaces.py)",
            "@dataclass(frozen=True)\n"
            "class MDPConfig:\n"
            "    H_max: float = 6.0          # верхняя граница энтропии H\n"
            "    K_max: float = 8.0          # верхняя граница дивергенции K\n"
            "    gamma_max: int = 8          # предельная длина черновика\n"
            "    N_H: int = 20               # число бинов по оси H\n"
            "    N_K: int = 20               # число бинов по оси K\n"
            "    T_levels: tuple = (1.0, 1.22, 1.49, 1.82,\n"
            "                       2.22, 2.71, 3.3, 4.0)   # сетка порогов\n"
            "    lambda_discount: float = 0.99   # дисконт λ\n"
            "    kappa: float = 1.0          # множитель Лагранжа κ\n"
            "    alpha_smooth: float = 1.0   # сглаживание Лапласа\n"
            "    nu_min: int = 5             # порог прямой оценки переходов\n"
            "    quality_risk_form: str = \"multiplicative\"   # 'additive': теорема B"),
    P("Поток данных организован как линейный конвейер из четырёх стадий, изображённый на "
      "рисунке 3.1. Первая стадия (scripts/01_collect_traces.py) собирает трейсы "
      "спекулятивного декодирования на отложенной выборке промптов и сохраняет их в формате "
      "Parquet. Вторая (scripts/02_solve_mdp.py) оценивает функцию переходов и функцию "
      "награды, решает MDP методом итерации по ценности и сохраняет совместную политику "
      "вместе с двумя каскадными. Третья (scripts/03_benchmark.py) выполняет многосидовый "
      "бенчмарк против базовых методов и записывает результаты в JSONL. Четвёртая "
      "(scripts/04_verify_conditions.py) проверяет эмпирические условия монотонности C1–C4 и "
      "невырожденности N1–N2 и строит диагностические графики; шаблоны из reports/templates "
      "строят по сохранённым артефактам Парето-диаграммы, пороговые поверхности и графики "
      "ablation."),
    fig3_block,
    P("Каждая стадия завершается записью артефакта на диск и не зависит от состояния процесса "
      "предыдущей стадии. Отсюда два практических следствия: решение MDP можно повторять "
      "многократно при разных значениях множителя Лагранжа κ без пересбора трейсов, а "
      "длительный бенчмарк допускает докатку после сбоя. Такая организация прямо отвечает "
      "цели совместимости с существующими инференс-системами без переобучения базовых "
      "моделей."),

    H2("3.2 Модуль сбора трейсов"),
    P("Модуль jointadaspec/mdp/traces.py реализует сбор одношаговых переходов марковского "
      "процесса. Для каждого посещённого состояния процесс генерации вычисляет энтропию "
      "чернового распределения H, дивергенцию Кульбака — Лейблера K между черновым и целевым "
      "распределениями, дискретный индекс состояния и множество допустимых в нём действий."),
    P("Вычисление признаков вынесено в модуль core/features.py. Энтропия H и дивергенция K "
      "вычисляются в натах функциями entropy и kl_divergence, принимающими как логиты, так и "
      "уже нормированные распределения. Функция quantize отображает непрерывную тройку "
      "(H, K, k) в линейный индекс состояния: значения H и K обрезаются до диапазонов "
      "[0, H_max] и [0, K_max], переводятся в номера бинов по равномерной сетке 20 × 20 и "
      "сворачиваются с позицией k в единый индекс из [0, |S|). Обратная функция dequantize "
      "восстанавливает центры бинов и применяется при интерпретации выученной политики."),
    P("Реализация отличается от схематичного описания подраздела 2.1.3 одной существенной "
      "деталью. В посещённом состоянии оцениваются все допустимые действия, и для каждого из "
      "них записывается отдельная строка трейса с наблюдённой наградой и следующим "
      "состоянием; лишь после этого случайно выбирается одно действие для продолжения "
      "rollout. Такая схема повышает плотность покрытия пар «состояние — действие» при том же "
      "числе промптов и уменьшает долю пар, для которых эмпирическая оценка переходов "
      "опирается на сглаживание Лапласа."),
    P("Трейсы сохраняются в Parquet вместе с JSON-метаданными — датой сбора, хешем коммита "
      "Git, числом собранных переходов и полным набором параметров MDP. Привязка трейса к "
      "коммиту и параметрам нужна не только для воспроизводимости: она позволяет повторно "
      "решать MDP при изменённых коэффициентах награды, не запуская заново дорогостоящую "
      "стадию сбора, которая требует вызовов целевой модели."),

    H2("3.3 Модуль оценки MDP и решения value iteration"),
    P("Оценка параметров MDP и его решение разнесены по двум модулям. Модуль "
      "jointadaspec/mdp/estimation.py по множеству собранных переходов строит разреженную "
      "матрицу переходов формы (|S| · |A|) × |S|, таблицу ожидаемых наград и счётчик "
      "посещений. Когда число наблюдений пары «состояние — действие» не меньше порога "
      "nu_min, используется прямая эмпирическая оценка распределения переходов; при меньшем "
      "числе наблюдений к оценке добавляется аддитивное сглаживание Лапласа с коэффициентом "
      "alpha_smooth по формуле (2.14). Разреженное представление оправдано структурой задачи: "
      "из любого состояния достижимо лишь небольшое число соседних, поэтому матрица переходов "
      "содержит порядка одного процента ненулевых элементов."),
    P("Модуль jointadaspec/mdp/value_iteration.py решает MDP разреженной реализацией итерации "
      "по ценности (формула 2.15), ядро которой приведено в листинге 3.2. Множество "
      "допустимых действий маскируется по компоненте k: при достижении предельной длины "
      "черновика k = γ_max все действия continue исключаются, и состояние становится "
      "терминальным по оси длины. Итерации продолжаются до выполнения критерия остановки по "
      "норме невязки; на табличном MDP из 3600 состояний сходимость достигается за время "
      "порядка единиц секунд на одном ядре процессора."),
    LISTING("Листинг 3.2 — Ядро разреженной итерации по ценности "
            "(jointadaspec/mdp/value_iteration.py)",
            "def solve_mdp(transitions, rewards, config, action_mask=None):\n"
            "    S, A = rewards.shape\n"
            "    V = np.zeros(S)\n"
            "    invalid = ~action_mask if action_mask is not None else None\n"
            "    for _ in range(config.max_vi_iterations):\n"
            "        expected = transitions.dot(V).reshape(S, A)   # разреженно\n"
            "        Q = rewards + config.lambda_discount * expected\n"
            "        if invalid is not None:\n"
            "            Q[invalid] = -np.inf\n"
            "        new_V = np.max(Q, axis=1)\n"
            "        if np.max(np.abs(new_V - V)) < config.epsilon_convergence:\n"
            "            V = new_V; break\n"
            "        V = new_V\n"
            "    pi_star = np.argmax(Q, axis=1)\n"
            "    return V, pi_star"),
    P("Стадия 02_solve_mdp.py решает MDP не для одного, а для набора значений множителя κ из "
      "конфигурации (поле kappa_values). Для каждого κ базовая конфигурация копируется с "
      "заменой поля kappa, после чего общим решателем находятся совместная и обе каскадные "
      "политики, сохраняемые с суффиксом значения κ. Перебор κ строит, согласно теореме 2.4, "
      "набор точек на границе Парето «скорость — качество» — и всё это без повторного сбора "
      "трейсов, благодаря независимости стадий конвейера."),
    P("Каскадные базовые политики реализованы в том же модуле как решения MDP с "
      "ограничениями: одна ось управления фиксируется по результату первого одномерного "
      "решения, после чего оптимизируется вторая. Использование общего решателя для "
      "совместной и каскадной политик устраняет систематическое расхождение, которое "
      "возникло бы при сравнении разнородных реализаций, и делает измеренный разрыв качества "
      "свойством самих политик, а не различием их программной реализации. Найденная политика "
      "сохраняется в формате .npz — массив выбранных действий вместе с конфигурацией, при "
      "которой он получен; объём файла составляет порядка двух килобайт, что на много "
      "порядков меньше размера моделей и согласуется с оценкой накладных расходов из "
      "подраздела 2.4."),

    H2("3.4 Модуль инференса с политикой JointAdaSpec"),
    P("Инференс с выученной политикой сосредоточен в модулях inference/policy.py и "
      "inference/jointadaspec.py. Первый загружает сохранённую таблицу политики и "
      "сопутствующую конфигурацию; второй реализует онлайн-декодер JointAdaSpecDecoder. На "
      "каждом шаге генерации черновика декодер получает распределения черновой и целевой "
      "моделей, вычисляет признаки H и K, дискретизирует их, извлекает из таблицы действие — "
      "решение о продолжении черновика и порог нечёткой верификации — и применяет это "
      "действие по правилу (1.4)."),
    P("Само правило приёма реализовано в модуле core/verification.py. Точная верификация "
      "(функция modified_rejection_sampling) принимает черновой токен с вероятностью "
      "min(1, p / q) и при отклонении пересемплирует токен из остаточного распределения, "
      "воспроизводя формулу (1.2); нечёткий вариант ослабляет условие сравнения множителем T "
      "согласно (1.4). Декодер JointAdaSpecDecoder вызывает это правило с порогом, который "
      "вернула политика, и накапливает принятый префикс блока, после чего обновляет "
      "состояние и переходит к следующему блоку."),
    P("Для согласованности с теорией состояние использует задержанное значение дивергенции: "
      "компонента K_prev инициализируется значением K_init = 0 и обновляется только после "
      "фазы верификации, когда становится доступным распределение целевой модели. Тем самым "
      "онлайн-состояние всегда опирается на уже наблюдённые величины и не требует обращения к "
      "целевой модели на стадии генерации черновика, что и заложено в инференс-алгоритме "
      "подраздела 2.4."),
    P("Quality-aware вариант метода реализует теорему B: штраф за расхождение распределений "
      "усиливается в состояниях с высокой дивергенцией K и большой позицией k. Выбор между "
      "аддитивной формой штрафа (корректной по теореме B и сохраняющей λ-сжимаемость) и "
      "мультипликативной задаётся флагом quality_risk_form в структуре MDPConfig; по "
      "умолчанию сохранена мультипликативная форма ради обратной совместимости с ранее "
      "выученными политиками. Существенно, что эта версия не меняет интерфейс инференса — "
      "модификация затрагивает лишь формирование награды на стадии оценки MDP, поэтому "
      "существующий бенчмарк-раннер использует новую политику без правок формата выходного "
      "JSONL."),

    H2("3.5 Реализация baseline-методов"),
    P("Корректное сравнение метода требует единообразной реализации базовых декодеров. "
      "Подпакет jointadaspec/baselines содержит шесть из них, перечисленных в таблице 3.2. "
      "Все они написаны поверх общего интерфейса спекулятивного декодера из core/, что "
      "исключает расхождения реализации как источник систематической погрешности."),
    TBL_CAPTION("Таблица 3.2 — Базовые декодеры подпакета jointadaspec/baselines"),
    TBL([["Декодер", "Описание"],
         ["vanilla_ar", "авторегрессионная генерация только целевой моделью"],
         ["fixed_sd", "спекулятивное декодирование с фиксированной длиной черновика"],
         ["fuzzy_sd", "нечёткое спекулятивное декодирование с фиксированным порогом T [7]"],
         ["specdecpp", "адаптивный выбор длины черновика в духе SpecDec++ [8]"],
         ["cascade_length_then_verif", "каскадная оптимизация «длина → порог»"],
         ["cascade_verif_then_length", "каскадная оптимизация «порог → длина»"]],
        [3500, 5854], align=["left", "left"]),
    P("Наличие именно каскадных базовых методов принципиально для замысла работы. Каскад "
      "воспроизводит наиболее естественную альтернативу совместной оптимизации — "
      "последовательный подбор сначала одного, затем другого параметра управления. Поэтому "
      "измеренный разрыв между JointAdaSpec и лучшим из каскадов служит прямой эмпирической "
      "проверкой слабого доминирования (теорема 2.3): проверяется, превосходит ли совместная "
      "политика лучшую каскадную. Как показано в разделе 4, этот разрыв статистически "
      "незначим — совместная и каскадная политики неразличимы, что согласуется с теоремой D."),
    P("Все декодеры пишут результаты в единую схему JSONL версии 2. Для GSM8K дополнительно "
      "фиксируется точное совпадение финального числового ответа на уровне отдельной задачи, "
      "а сводные записи содержат медианную скорость в токенах в секунду, долю принятий "
      "(acceptance rate), агрегированные оценки качества и бутстреп-доверительные интервалы. "
      "Единая схема позволяет обрабатывать результаты всех методов одним валидатором и одними "
      "шаблонами отчётов."),
    P("Стадия 03_benchmark.py выполняет сравнение в многосидовом режиме и устойчива к сбоям: "
      "результаты дописываются в results.jsonl построчно, а уже посчитанные комбинации "
      "(метод, сид, промпт) пропускаются при повторном запуске по ключу resume_key. Наряду с "
      "основным файлом ведётся совместимый со старыми читателями журнал run.jsonl. Такая "
      "организация позволяет докатывать длительный прогон после прерывания, не теряя ранее "
      "вычисленных записей."),

    H2("3.6 Инструменты, тестирование и воспроизводимость"),
    P("Программный комплекс написан на языке Python версии 3.12. Работа с моделями опирается "
      "на библиотеки PyTorch и Transformers, оценка и решение MDP — на разреженные матрицы "
      "scipy.sparse и пакет NumPy, конфигурирование экспериментов — на Hydra и OmegaConf, "
      "построение отчётов — на matplotlib, модульное тестирование — на pytest. Эталонный "
      "прогон, описанный в разделе 4, выполнен со связкой PyTorch 2.9.1 для CUDA 12.8 и "
      "Transformers 4.57.3 на графическом ускорителе NVIDIA RTX 5090."),
    P("Качество кода поддерживается непрерывной интеграцией. При каждом изменении система "
      "GitHub Actions запускает синтаксическую проверку и валидацию конфигураций, полный "
      "набор из 72 модульных тестов (15 файлов в каталоге tests/), дымовой прогон SpecExec и "
      "строгую проверку схемы выходного JSONL. Тяжёлые локальные каталоги — веса моделей, "
      "наборы данных, журналы и виртуальное окружение — исключены из системы контроля "
      "версий, что удерживает репозиторий компактным."),
    P("Набор тестов покрывает ключевые инварианты метода, а не только синтаксис. Тесты "
      "модуля MDP проверяют сходимость итерации по ценности, маскирование действий continue "
      "при k = γ_max и форму извлечённой политики; тесты признаков — корректность энтропии, "
      "дивергенции и обратимость дискретизации; тесты верификации — сохранение распределения "
      "точным правилом и долю принятий при нечётком; тесты каскада — совпадение совместной и "
      "каскадной политик на сепарабельной награде (вырожденный случай теоремы 2.3); сквозной "
      "тест прогоняет весь конвейер на игрушечных моделях. Тем самым проверяется "
      "математическая корректность ядра, а не только отсутствие ошибок компиляции."),
    P("Конфигурации и выходные данные дополнительно защищены валидаторами. Отдельный скрипт "
      "сверяет согласованность пресетов моделей, методов и экспериментов и совместимость "
      "токенизаторов внутри пары; строгий валидатор схемы проверяет каждую запись выходного "
      "JSONL перед построением отчётов. Локальные цели make check и make test запускают эти "
      "проверки вместе со всем набором тестов одной командой, благодаря чему регрессии "
      "обнаруживаются немедленно, а конфигурация остаётся единственным источником истины о "
      "параметрах прогона."),
    P("Воспроизводимость обеспечивается двумя механизмами. Детерминизм прогонов достигается "
      "раздельным посевом генераторов случайных чисел Python, NumPy и PyTorch для каждого "
      "начального значения из набора [42, 43, 44], установкой переменной окружения "
      "CUBLAS_WORKSPACE_CONFIG и включением детерминированного режима вычислений. Каждый "
      "прогон сопровождается манифестом в каталоге reports/manifests, который фиксирует хеш "
      "коммита Git, признак незакоммиченных изменений, разрешённую конфигурацию, список сидов "
      "и контрольные суммы SHA256 артефактов трейсов и политики. Отдельные вендорные ядра "
      "CUDA при этом сохраняют предупреждение о неполной воспроизводимости; в таких случаях "
      "первичной записью считаются именно манифест и список сидов, а малый числовой дрейф "
      "признаётся допустимым. Детали описанного комплекса использованы в экспериментальном "
      "исследовании раздела 4."),
    ])

def razdel4(figs):
    return block([
    H1("4 Экспериментальное исследование"),
    P("Настоящий раздел представляет экспериментальную оценку метода JointAdaSpec. Подраздел "
      "4.1 фиксирует методику и метрики, 4.2 — исследуемые пары моделей и данные. Подраздел "
      "4.3 содержит основной результат на паре с высоким отношением мощностей, 4.4 — "
      "честный нуль-результат и его триангуляцию на паре с низким отношением. Подразделы "
      "4.5–4.6 проверяют адаптивность против фиксированного порога (теорема E) и поведение "
      "вдоль параметра компромисса κ вместе с нелинейностью качества (теорема G). Подраздел "
      "4.7 "
      "сопоставляет метод с AutoJudge и обобщает результаты. Все числовые значения взяты из "
      "зафиксированных артефактов прогонов."),

    H2("4.1 Методика экспериментов и метрики"),
    P("Эксперименты выполнены на одном графическом ускорителе NVIDIA RTX 5090. Основным "
      "набором служит GSM8K — школьные математические задачи, требующие многошагового "
      "рассуждения [5], — в режиме zero-shot CoT с ограничением 256 токенов на ответ; "
      "качество измеряется точным совпадением (exact match) извлечённого числового ответа с "
      "эталоном. Скорость выражена пропускной способностью в токенах в секунду и ускорением "
      "относительно других методов; дополнительно фиксируется доля принятых черновых токенов "
      "(acceptance rate)."),
    P("Статистическая надёжность обеспечена многосидовой схемой. Каждая конфигурация "
      "прогоняется на трёх начальных значениях из детерминированной последовательности "
      "[42, 43, 44], что при выборке в 100–500 промптов даёт от 300 до 1500 парных "
      "наблюдений. Разность качества оценивается на парных данных: для каждого промпта "
      "сравниваются ответы двух методов, значимость проверяется парным критерием Макнемара, "
      "а доверительные интервалы вычисляются бутстрепом. Трейсы для оценки MDP собираются на "
      "обучающей части GSM8K, а бенчмарк выполняется на тестовой — обе выборки законно "
      "отложены."),
    P("Прогоны разделены на три уровня надёжности, смешивать которые при интерпретации "
      "недопустимо. Содержательный прогон имеет достаточный объём трейсов и промптов для "
      "статистических выводов. Дымовой прогон на одном-пяти промптах проверяет "
      "работоспособность схемы, обработку сидов и манифесты, но его метрики результатами "
      "метода не считаются. Частичный прогон сохранил артефакты лишь части стадий. Ниже "
      "содержательными признаются только прогоны первого уровня."),

    H2("4.2 Исследуемые пары моделей и наборы данных"),
    P("Исследованы две пары моделей семейства Qwen2.5 [13], различающиеся отношением "
      "мощностей целевой и черновой моделей (таблица 4.1). Высокое отношение делает пару "
      "показательной для режима, в котором шаг целевой модели дорог, а черновой дёшев; "
      "именно при таком отношении формула ускорения (1.3) обещает наибольший выигрыш. "
      "Низкое отношение соответствует менее благоприятному режиму и служит проверкой "
      "устойчивости."),
    TBL_CAPTION("Таблица 4.1 — Исследуемые пары моделей"),
    TBL([["Пара", "Целевая модель", "Черновая модель", "Отношение", "Роль"],
         ["Основная", "Qwen2.5-14B-Instruct", "Qwen2.5-0.5B-Instruct", "≈28×",
          "благоприятный режим"],
         ["Дополнительная", "Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct", "4,7×",
          "проверка устойчивости"]],
        [1500, 2600, 2600, 950, 1704],
        align=["left", "left", "left", "center", "left"]),
    P("Обе пары выбраны из соображений совместимости токенизаторов: черновая и целевая "
      "модели внутри пары используют идентичную таблицу словаря, что является обязательным "
      "условием корректности спекулятивной верификации (подраздел 1.2). Веса всех моделей "
      "размещены локально, что исключает зависимость прогона от внешних репозиториев."),

    H2("4.3 Основной результат: пара 14B/0.5B"),
    P("На основной паре сравнивались четыре метода: прямая генерация целевой моделью "
      "(target_only), ванильное спекулятивное декодирование (speculative), лучшая каскадная "
      "политика и совместная политика JointAdaSpec. Результаты при n = 1500 приведены в "
      "таблице 4.2."),
    TBL_CAPTION("Таблица 4.2 — Сравнение методов на паре 14B/0.5B (GSM8K, n = 1500)"),
    TBL([["Метод", "EM, %", "ток/с", "к spec", "Δ EM к target, п.п.", "p"],
         ["target_only", "52,93", "14,45", "2,96×", "—", "—"],
         ["speculative", "53,27", "4,88", "1,00×", "+0,33", "0,877"],
         ["cascade_verif_then_length", "57,13", "10,29", "2,11×", "+4,20", "0,015"],
         ["jointadaspec", "57,00", "10,84", "2,22×", "+4,07", "0,020"]],
        [3050, 1000, 1000, 1000, 1950, 1354],
        align=["left", "center", "center", "center", "center", "center"]),
    P("Совместная политика повышает точность на +4,07 п.п. относительно прямой генерации "
      "(p = 0,0205, 95 %-й доверительный интервал [+0,73; +7,40]) при пропускной "
      "способности 10,84 ток/с — в 2,22 раза выше ванильного спекулятивного декодирования. "
      "Однако совместная и каскадная политики статистически неразличимы: их разность "
      "составляет −0,13 п.п. (p = 0,96). Это не дефект настройки, а прямое следствие теоремы "
      "D: преимущество каскада на множестве нарушения условия C4 близко к нулю, поэтому "
      "совместная политика практически сводится к каскадной. Защищаемый вывод формулируется "
      "не как «совместная лучше всех», а как «семейство адаптивного управления (совместная и "
      "каскадная политики) значимо превосходит прямую генерацию по точности при "
      "восстановленной скорости»."),
    P("Парный характер критерия Макнемара здесь существен. Значимость определяют не валовые "
      "доли правильных ответов, а дискордантные пары — промпты, на которых два метода "
      "расходятся: совместная политика чаще исправляет ошибку прямой генерации, чем портит "
      "верный ответ, и именно этот перевес исправлений над порчей даёт прирост. Доверительный "
      "интервал [+0,73; +7,40], не пересекающий нуля, подтверждает значимость на уровне "
      "p < 0,05."),
    figs["pareto"],
    P("Существенна честная интерпретация скорости. Самым быстрым методом является именно "
      "прямая авторегрессионная генерация целевой моделью (14,45 ток/с), что видно на "
      "рисунке 4.1: точка target_only расположена правее всех. Ванильное спекулятивное "
      "декодирование на этой паре теряет скорость (4,88 ток/с), а адаптивное управление эту "
      "потерю восстанавливает до 10,84 ток/с. Поэтому приведённое ускорение «2,22×» — это "
      "ускорение относительно ванильного спекулятивного декодирования, а не относительно "
      "прямой генерации; корректная формулировка вклада по скорости — «восстановление "
      "пропускной способности, теряемой ванильным SD»."),

    H2("4.4 Нуль-результат и триангуляция: пара 7B/1.5B"),
    P("На паре с низким отношением мощностей (4,7×) картина качественно иная (таблица 4.3). "
      "Ни совместная, ни каскадная политика не превосходят прямую генерацию по точности, а "
      "ванильное спекулятивное декодирование здесь медленнее прямой генерации — прямое "
      "следствие формулы ускорения (1.3) при малом разрыве в стоимости моделей."),
    TBL_CAPTION("Таблица 4.3 — Сравнение методов на паре 7B/1.5B (GSM8K, n = 1500)"),
    TBL([["Метод", "EM, %", "ток/с", "к spec", "Δ EM к target, п.п.", "p"],
         ["target_only", "60,20", "24,93", "1,90×", "—", "—"],
         ["speculative", "60,60", "13,13", "1,00×", "+0,40", "0,830"],
         ["cascade_verif_then_length", "57,93", "15,76", "1,20×", "−2,27", "0,144"],
         ["jointadaspec", "58,73", "15,77", "1,20×", "−1,47", "0,369"]],
        [3050, 1000, 1000, 1000, 1950, 1354],
        align=["left", "center", "center", "center", "center", "center"]),
    P("Ранний прогон на меньшей выборке давал положительную оценку (+3,50 п.п.), однако она "
      "не устояла при увеличении мощности. Чтобы исключить зависимость вывода от конкретного "
      "среза данных, прирост совместной политики относительно прямой генерации измерен на "
      "трёх независимых отложенных окнах (таблица 4.4, рисунок 4.2)."),
    TBL_CAPTION("Таблица 4.4 — Триангуляция нуль-результата на паре 7B/1.5B"),
    TBL([["Окно (start)", "n", "joint − target, п.п.", "p"],
         ["1100 (исходное)", "200", "+3,50", "0,146"],
         ["100 (lock)", "1500", "−1,47", "0,369"],
         ["600 (триангуляция)", "1500", "−2,53", "0,094"]],
        [3000, 1100, 3100, 2154],
        align=["left", "center", "center", "center"]),
    figs["win3"],
    P("На двух независимых хорошо обеспеченных окнах (n = 1500) прирост качества "
      "отрицателен, а ранний положительный результат при n = 200 объясняется дисперсией "
      "малой выборки. Вывод устойчив к выбору среза: на низком отношении мощностей метод не "
      "даёт выигрыша в качестве. Это очерчивает границу применимости — совместное "
      "адаптивное управление полезно при достаточно большом разрыве в стоимости моделей."),
    P("Расхождение раннего и поздних окон иллюстрирует роль статистической мощности. При "
      "n = 200 ширина доверительного интервала для разности долей превышает несколько "
      "процентных пунктов, поэтому единичное окно не разделяет малый эффект и его отсутствие. "
      "Увеличение до n = 1500 сужает интервал, и оценка стабилизируется около нуля или ниже. "
      "Ранний положительный результат поэтому интерпретируется как артефакт малой выборки, а "
      "не как эффект, утраченный при росте мощности."),

    H2("4.5 Адаптивность против фиксированного порога (теорема E)"),
    P("Естественный вопрос — нужна ли адаптивность вообще, если порог T можно подобрать "
      "фиксированным. Ответ даёт сравнение совместной политики с нечётким спекулятивным "
      "декодированием при фиксированных значениях T на основной паре (таблица 4.5, "
      "рисунок 4.3)."),
    TBL_CAPTION("Таблица 4.5 — JointAdaSpec против фиксированного fuzzy_sd (14B/0.5B, n = 300)"),
    TBL([["Метод", "EM, %", "ток/с", "доля принятий", "парно к joint"],
         ["fuzzy_sd, T = 1,0", "53,67", "2,85", "0,152", "joint +4,33 (p = 0,275)"],
         ["fuzzy_sd, T = 1,25", "50,33", "2,95", "0,163", "joint +7,67 (p = 0,051)"],
         ["fuzzy_sd, T = 1,5", "51,67", "3,00", "0,167", "joint +6,33 (p = 0,115)"],
         ["fuzzy_sd, T = 2,0", "53,33", "3,06", "0,174", "joint +4,67 (p = 0,243)"],
         ["jointadaspec", "58,00", "11,12", "0,551", "—"]],
        [2350, 950, 950, 1700, 3404],
        align=["left", "center", "center", "center", "left"]),
    figs["E"],
    P("Совместная политика доминирует каждый фиксированный порог сразу по обеим осям: она "
      "точнее на +4…+8 п.п. exact match и при этом быстрее примерно в 3,7 раза (11,12 против "
      "≈3,0 ток/с). Это и есть содержание теоремы E: адаптивное управление эмпирически "
      "нетривиально по сравнению с естественным неадаптивным базисом. Тем самым то "
      "единственное, чему совместная политика лишь не уступает, — это другая адаптивная "
      "MDP-политика (каскад), что в точности предсказано теоремой D."),

    H2("4.6 Анализ компромисса по κ и нелинейности качества (теорема G)"),
    P("Множитель κ из функции награды (2.8) задаёт компромисс «скорость — качество». На "
      "шести значениях κ ∈ {0, 1, 5, 20, 50, 100} совместная и каскадная политики "
      "пересчитывались и заново прогонялись на фиксированном срезе пары 7B/1.5B "
      "(рисунок 4.4)."),
    figs["kappa"],
    P("Пропускная способность вдоль κ почти постоянна (16,28–16,55 ток/с), а точность "
      "меняется немонотонно (совместная 55,7–60,3 %, каскадная 55,0–61,0 %); при этом "
      "совместная и каскадная политики близки при каждом κ (расхождение не более трёх "
      "пунктов). Эквивалентность joint ≈ cascade устойчива ко всему диапазону "
      "трейд-офф-параметра, а не только к его значению по умолчанию. Следует честно "
      "оговорить: строгая выпуклость достижимого множества (теорема 2.4) при объёме n = 300 "
      "на значение κ чисто не демонстрируется — этот результат носит характер согласованности, "
      "а не строгого подтверждения."),
    P("В основных прогонах использовано значение κ = 1 по умолчанию. Развёртка показывает, "
      "что выбор κ слабо влияет на скорость и умеренно — на качество, поэтому метод не "
      "требует тонкой настройки этого параметра под каждую пару моделей. Набор решений при "
      "разных κ образует, в духе теоремы 2.4, дискретную аппроксимацию границы Парето, из "
      "которой при необходимости выбирается рабочая точка под заданное ограничение качества."),
    P("Природа прироста качества раскрывается анализом по доле принятий (рисунок 4.5). Доля "
      "«перевёрнутых» ответов (churn) держится около 45 % и почти не зависит от уровня "
      "принятий; однако чистое изменение точности — разность долей «было неверно, стало "
      "верно» и «было верно, стало неверно» — немонотонно: оно составляет +7,4 п.п. при "
      "низкой и умеренной доле принятий и −2,6 п.п. при высокой. Содержательно, чрезмерное "
      "доверие дешёвой черновой модели на «высокоприёмном» режиме портит качество. "
      "Заявленный прирост +4 п.п. — это смесь по диапазону, а «рабочая точка» контроллера — "
      "умеренная доля принятий."),

    H2("4.7 Сравнение с AutoJudge и обсуждение результатов"),
    P("Для контекста метод сопоставлен с AutoJudge — приближённым методом с обучаемым "
      "верификатором (раздел 1.4) — на паре 7B/1.5B (таблица 4.6). AutoJudge с откалиброванным "
      "порогом достигает на этой паре более высокой точности (61,7 % exact match), что "
      "ожидаемо: на низком отношении мощностей обучаемый отбор значимых токенов выигрывает у "
      "управления гиперпараметрами. Сравнение приведено как ориентир из литературы; setup "
      "AutoJudge (k = 4, выборка n = 100 × 3) отличается от прогонов JointAdaSpec и не "
      "является парным."),
    TBL_CAPTION("Таблица 4.6 — Контекст: AutoJudge и базовые методы на паре 7B/1.5B (k = 4)"),
    TBL([["Метод", "EM, %", "ток/с", "к spec"],
         ["target_only (7B)", "58,1", "78,6", "1,67×"],
         ["speculative", "56,9", "47,2", "1,00×"],
         ["AutoJudge, t = 0,09", "61,7", "55,9", "1,18×"],
         ["AutoJudge, t = 1,0", "52,3", "63,4", "1,34×"],
         ["Top-K, rank = 4", "54,3", "71,5", "1,52×"]],
        [3100, 1500, 1500, 3254],
        align=["left", "center", "center", "center"]),
    P("Существенно, что JointAdaSpec и методы вроде AutoJudge не исключают друг друга. "
      "Совместное управление парой (длина, порог) применимо поверх любой черновой модели и "
      "любого правила приёма, поэтому в принципе сочетается с обучаемым верификатором; "
      "экспериментальная проверка такой комбинации обозначена как направление дальнейшей "
      "работы. К наиболее перспективным улучшениям относятся увеличение окна черновика, "
      "распределительные признаки состояния и перенос табличной политики на ускоритель для "
      "устранения остаточных накладных расходов инференса."),
    P("Результаты следует сопроводить обсуждением угроз валидности. Измерения скорости "
      "получены на одном графическом ускорителе; на другой аппаратуре или при пакетировании "
      "запросов абсолютные значения изменятся, хотя относительные сравнения методов "
      "устойчивее. Отдельные вендорные ядра CUDA не полностью детерминированы, поэтому "
      "первичной записью воспроизводимости служат манифест и список начальных значений, а "
      "малый числовой дрейф признаётся допустимым (приложение Г). Качество оценивается "
      "только по GSM8K: бенчмарки кода и диалога в текущем раннере не оцениваются по точному "
      "совпадению, поэтому выводы о качестве ограничены областью математических рассуждений."),
    P("Отдельного внимания заслуживает устойчивость к смещению распределения. Политика "
      "обучается при длине черновика k = 8; её перенос на k = 16 без переобучения ожидаемо "
      "приводит к регрессии, поскольку расширенное пространство состояний не покрыто "
      "трейсами — это методологически ожидаемое ограничение области определения, а не дефект "
      "метода. На объединённой отложенной выборке пары 7B/1.5B (n = 3000) совместная "
      "политика теряет −2,00 п.п. (p = 0,067), причём каскадная теряет больше; такое "
      "поведение характеризуется как грациозная деградация при сдвиге распределения. Эти "
      "ограничения очерчивают область надёжного применения метода."),
    P("Совокупность результатов складывается в следующий честный вывод. Вклад работы — это "
      "единый MDP-фреймворк совместного управления длиной черновика и порогом верификации "
      "вместе с теоретическим аппаратом (теоремы A, B, C, D, 2.3, 2.4), а не утверждение "
      "«совместная политика побеждает все методы». Эмпирически адаптивное управление значимо "
      "превосходит любой неадаптивный базис — прямую генерацию, ванильное спекулятивное "
      "декодирование и фиксированный нечёткий порог (теорема E), — и при этом совпадает с "
      "другой адаптивной MDP-политикой, каскадной, ровно как предсказывает теорема D. "
      "Границы применимости очерчены экспериментально: метод полезен при высоком отношении "
      "мощностей моделей и не даёт выигрыша при низком, а его выигрыш по скорости следует "
      "понимать как восстановление пропускной способности относительно ванильного "
      "спекулятивного декодирования, а не как превосходство над прямой генерацией. Эти "
      "наблюдения и направления их преодоления обобщены в заключении."),
    ])

ZAKL = block([
    H1("Заключение"),
    P("В работе решена задача разработки и исследования метода адаптивного спекулятивного "
      "декодирования, совместно управляющего длиной черновика и порогом нечёткой верификации. "
      "Все четыре поставленные во введении задачи выполнены: проведён анализ современных "
      "методов и выявлен исследовательский пробел — отсутствие совместной оптимизации двух "
      "осей управления; построена и теоретически исследована MDP-модель совместного "
      "управления; реализован воспроизводимый программный комплекс из стадий сбора трейсов, "
      "оценки MDP, решения value iteration и инференса с выученной политикой; проведена "
      "многосидовая экспериментальная оценка на современных открытых моделях."),
    P("Предложенный метод JointAdaSpec формализует выбор пары (длина черновика, порог "
      "верификации) на каждом цикле декодирования как действие в дискретном марковском "
      "процессе принятия решений с малым пространством состояний (3600 элементов), решаемом "
      "точно методом итерации по ценности без обучения нейросетевых модулей. На паре "
      "Qwen2.5-14B → 0.5B с высоким отношением мощностей метод повышает точность на наборе "
      "GSM8K на +4,07 п.п. exact match относительно прямой генерации (p = 0,0205 по парному "
      "критерию Макнемара при n = 1500 парных наблюдениях) при восстановлении пропускной "
      "способности до 2,22-кратной относительно ванильного спекулятивного декодирования."),
    P("Центральный теоретический и эмпирический результат — точная характеризация "
      "соотношения совместной и каскадной политик. Они статистически неразличимы "
      "(−0,13 п.п., p = 0,96), и это совпадение не случайно: доказана теорема D, дающая "
      "точное выражение разрыва ценности через преимущество каскада на множестве нарушения "
      "условия супермодулярности; эмпирически это преимущество пренебрежимо мало, поэтому "
      "нарушение условия повсеместно, но доброкачественно. Получен и сопутствующий "
      "теоретический аппарат: оценка выборочной сложности оценщика переходов (теорема A), "
      "Беллман-инвариантность аддитивного штрафа качества (теорема B), линейная по мере "
      "нарушения оценка субоптимальности каскада (теорема C), слабое доминирование "
      "совместной оптимизации (теорема 2.3) и скаляризация Парето-фронта (теорема 2.4)."),
    P("Эмпирически адаптивное управление значимо превосходит любой неадаптивный базис — "
      "прямую генерацию, ванильное спекулятивное декодирование и фиксированный нечёткий "
      "порог: на основной паре совместная политика точнее каждого фиксированного порога на "
      "+4…+8 п.п. и быстрее примерно в 3,7 раза (теорема E). Научная новизна работы состоит "
      "в первой формализации совместного управления длиной черновика и порогом верификации "
      "как единого марковского процесса принятия решений с доказуемыми свойствами; ранее эти "
      "две оси рассматривались в литературе порознь."),
    P("Полученные результаты честно очерчивают и границы метода. Самым быстрым методом "
      "остаётся прямая авторегрессионная генерация, а выигрыш по скорости следует понимать "
      "как восстановление пропускной способности, теряемой ванильным спекулятивным "
      "декодированием, а не как превосходство над прямой генерацией. На паре с низким "
      "отношением мощностей (7B/1.5B, 4,7×) прирост качества не подтверждён на трёх "
      "независимых отложенных окнах. Прирост качества немонотонен по доле принятий — выигрыш "
      "на умеренном и проигрыш на высоком уровне, — а вклад совместности ограничен тем, что "
      "она сводится к каскадной политике. Защищаемый тезис — это единый фреймворк и его "
      "честная характеризация, а не безусловное превосходство совместной политики."),
    P("Дальнейшее развитие метода видится в нескольких направлениях. Обобщение управления с "
      "линейной длины черновика на форму дерева кандидатов расширило бы пространство "
      "действий и потенциальный выигрыш. Включение распределительных признаков состояния "
      "повысило бы точность выученной политики. Увеличение окна черновика и перенос "
      "табличной политики на ускоритель устранили бы остаточные накладные расходы инференса. "
      "Наконец, настройка награды под конкретные классы задач — в частности, генерацию "
      "кода — позволила бы перенести результат за пределы математических рассуждений."),
])

# --------------------- Список источников ---------------------
def REF(num, text):
    return ('<w:p><w:pPr><w:ind w:left="426" w:hanging="426"/>'
            '<w:spacing w:after="80"/><w:jc w:val="both"/>'
            '<w:rPr><w:sz w:val="28"/></w:rPr></w:pPr>'
            + _runs(f"{num}. {text}") + '</w:p>')

SPISOK = block([
    H1("Список использованных источников"),
    REF(1, "Azar M. G. Minimax PAC bounds on the sample complexity of reinforcement "
           "learning with a generative model / M. G. Azar, R. Munos, H. J. Kappen // "
           "Machine Learning. — 2013. — Vol. 91, № 3. — P. 325–349."),
    REF(2, "Bellman R. Dynamic Programming / R. Bellman. — Princeton : Princeton "
           "University Press, 1957. — 342 p."),
    REF(3, "Cai T. Medusa: Simple LLM Inference Acceleration Framework with Multiple "
           "Decoding Heads [Электронный ресурс] / T. Cai, Y. Li, Z. Geng [et al.]. — "
           "Режим доступа: https://arxiv.org/abs/2401.10774 (дата обращения: 03.06.2026)."),
    REF(4, "Chen C. Accelerating Large Language Model Decoding with Speculative Sampling "
           "[Электронный ресурс] / C. Chen, S. Borgeaud, G. Irving [et al.]. — Режим "
           "доступа: https://arxiv.org/abs/2302.01318 (дата обращения: 03.06.2026)."),
    REF(5, "Cobbe K. Training Verifiers to Solve Math Word Problems [Электронный ресурс] "
           "/ K. Cobbe, V. Kosaraju, M. Bavarian [et al.]. — Режим доступа: "
           "https://arxiv.org/abs/2110.14168 (дата обращения: 03.06.2026)."),
    REF(6, "Garipov R. AutoJudge: Judge Decoding Without Manual Annotation [Электронный "
           "ресурс] / R. Garipov, F. Velikonivtsev, R. Svirschevski [et al.]. — Режим "
           "доступа: https://arxiv.org/abs/2504.20039 (дата обращения: 03.06.2026)."),
    REF(7, "Holsman M. Fuzzy Speculative Decoding for a Tunable Accuracy–Runtime "
           "Tradeoff [Электронный ресурс] / M. Holsman, Y. Huang, B. Dhingra. — Режим "
           "доступа: https://arxiv.org/abs/2502.20704 (дата обращения: 03.06.2026)."),
    REF(8, "Huang K. SpecDec++: Boosting Speculative Decoding via Adaptive Candidate "
           "Lengths [Электронный ресурс] / K. Huang, X. Guo, M. Wang. — Режим доступа: "
           "https://arxiv.org/abs/2405.19715 (дата обращения: 03.06.2026)."),
    REF(9, "Kakade S. Approximately Optimal Approximate Reinforcement Learning / "
           "S. Kakade, J. Langford // Proceedings of the 19th International Conference "
           "on Machine Learning (ICML). — 2002. — P. 267–274."),
    REF(10, "Leviathan Y. Fast Inference from Transformers via Speculative Decoding / "
            "Y. Leviathan, M. Kalman, Y. Matias // Proceedings of the 40th International "
            "Conference on Machine Learning (ICML). — Honolulu, 2023. — P. 19274–19286."),
    REF(11, "Li Y. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty "
            "[Электронный ресурс] / Y. Li, F. Wei, C. Zhang, H. Zhang. — Режим доступа: "
            "https://arxiv.org/abs/2401.15077 (дата обращения: 03.06.2026)."),
    REF(12, "Puterman M. L. Markov Decision Processes: Discrete Stochastic Dynamic "
            "Programming / M. L. Puterman. — New York : John Wiley & Sons, 1994. — 649 p."),
    REF(13, "Qwen Team. Qwen2.5 Technical Report [Электронный ресурс] / Qwen Team. — "
            "Режим доступа: https://arxiv.org/abs/2412.15115 (дата обращения: 03.06.2026)."),
    REF(14, "Sutton R. S. Reinforcement Learning: An Introduction / R. S. Sutton, "
            "A. G. Barto. — 2nd ed. — Cambridge : MIT Press, 2018. — 552 p."),
    REF(15, "Svirschevski R. SpecExec: Massively Parallel Speculative Decoding for "
            "Interactive LLM Inference on Consumer Devices / R. Svirschevski, A. May, "
            "Z. Chen [et al.] // Advances in Neural Information Processing Systems "
            "(NeurIPS). — 2024."),
    REF(16, "Topkis D. M. Supermodularity and Complementarity / D. M. Topkis. — "
            "Princeton : Princeton University Press, 1998. — 272 p."),
    REF(17, "Vaswani A. Attention Is All You Need / A. Vaswani, N. Shazeer, N. Parmar "
            "[et al.] // Advances in Neural Information Processing Systems (NeurIPS). — "
            "2017. — P. 5998–6008."),
    REF(18, "Yin M. A Theoretical Perspective for Speculative Decoding Algorithm / "
            "M. Yin, M. Chen, K. Huang, M. Wang [Электронный ресурс]. — Режим доступа: "
            "https://arxiv.org/abs/2411.00841 (дата обращения: 03.06.2026)."),
    REF(19, "Zheng L. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena "
            "[Электронный ресурс] / L. Zheng, W.-L. Chiang, Y. Sheng [et al.]. — Режим "
            "доступа: https://arxiv.org/abs/2306.05685 (дата обращения: 03.06.2026)."),
])

# --------------------- Приложения ---------------------
APP_A = block([
    H1("Приложение А. Доказательства теорем"),
    P("В приложении собраны доказательства утверждений, идеи которых приведены в разделах 1 "
      "и 2. Нумерация теорем сохранена; формулы приложения нумеруются (А.номер)."),

    H2("А.1 Свойство точности спекулятивного декодирования"),
    P("Утверждение. Последовательное применение правила (1.2) к γ черновым токенам, "
      "порождённым моделью q, даёт совместное распределение принятого префикса и "
      "замещающего токена, в точности совпадающее с распределением прямой авторегрессионной "
      "генерации из целевой модели p."),
    P("Доказательство ведётся индукцией по длине черновика γ. База (γ = 1): один черновой "
      "токен принимается с вероятностью min(1, p/q), а при отклонении пересемплируется из "
      "нормированного остатка (p − q)_+; прямая проверка показывает, что итоговое "
      "распределение токена равно p. Шаг индукции: пусть утверждение верно для длины γ − 1. "
      "Для первого токена результат правила (1.2) распределён по p; при его принятии "
      "контекст обновляется, и к оставшимся γ − 1 кандидатам применяется предположение "
      "индукции, дающее условное распределение целевой модели; при отклонении итерация "
      "завершается одним токеном из остатка. Совместное распределение записывается суммой "
      "по числу N подряд принятых токенов,"),
    EQ(mrow("P", mdelim(mrow(msub("x", "1"), op(", … , "), msub("x", mrow("N", op("+"), "1")))),
            op(" = "),
            mnary("∑", "k", None,
                  mrow(op("Pr"), mdelim(mrow("N", op(" = "), "k"), "[", "]"), " ",
                       "p", mdelim(mrow(msub("x", "1"), op(", … , "),
                                        msub("x", mrow("k", op("+"), "1"))))))), "А.1"),
    P("где каждое слагаемое по теореме о произведении условных вероятностей и предположению "
      "индукции воспроизводит распределение p на соответствующей подпоследовательности. "
      "Марковость генерации обеспечивает независимость шага от контекста, что завершает "
      "доказательство."),

    H2("А.2 Теорема A (выборочная сложность оценщика переходов)"),
    P("Доказывается оценка (2.10). По неравенству треугольника ошибка функции ценности "
      "разбивается на статистическую и смещения. Статистическая часть: для каждой пары "
      "(s, a) эмпирическое распределение переходов есть нормированный счётчик, к которому "
      "покоординатно применяется неравенство Хёфдинга; объединение оценок (union bound) по "
      "всем |S| · |A| парам даёт логарифмический множитель ln(2|S| · |A| / δ). Перенос "
      "ошибки переходов на ошибку ценности использует λ-сжатие оператора Беллмана и вносит "
      "множитель 1 / (1 − λ)²: одна степень отвечает за горизонт, вторая — за распространение "
      "ошибки через неподвижную точку (стандартный приём, см. [1])."),
    P("Часть смещения: сглаживание Лапласа с параметром α при n наблюдениях отклоняет оценку "
      "от несглаженной не более чем на α / (n + α) покоординатно; суммирование по носителю и "
      "по парам (s, a) и распространение через сжатие дают аддитивный член "
      "α |S| R / ((1 − λ)(n_min + α)). Сложение двух частей и даёт правую часть (2.10) с "
      "вероятностью не менее 1 − δ. Бэунд не туг по абсолютной величине, но устанавливает "
      "скорость сходимости порядка 1 / √(n_min) с убывающим по n_min смещением, чего "
      "достаточно для обоснования табличного пайплайна."),

    H2("А.3 Теорема B (Беллман-инвариантность аддитивного штрафа)"),
    P("Пусть к награде добавлен штраф −g(s), зависящий только от состояния. Уравнение "
      "оптимальности для изменённого MDP принимает вид"),
    EQ(mrow(msubsup("V", "g", "*"), mdelim("s"), op(" = "), op("−"), "g", mdelim("s"),
            op(" + "), msub(op("max"), mrow("a", op(" ∈ "), "A")),
            mdelim(mrow("r", mdelim(mrow("s", op(", "), "a")), op(" + "), "λ",
                        mnary("∑", "s′", None,
                              mrow("P", mdelim(mrow("s′", op(" | "), "s", op(", "), "a")),
                                   " ", msubsup("V", "g", "*"), mdelim("s′")))), "[", "]")),
       "А.2"),
    P("Слагаемое −g(s) не зависит от действия и выносится за знак максимума, поэтому "
      "argmax по a — а с ним и оптимальная политика — совпадает с политикой исходного MDP; "
      "функция ценности отличается лишь на дисконтированную сумму штрафов вдоль траектории. "
      "Следовательно, штраф качества способна изменить только связанная с действием форма, "
      "что и обосновывает конструкцию награды (2.8). Аддитивность сохраняет λ-сжатие "
      "оператора, тогда как мультипликативная форма его разрушала."),

    H2("А.4 Теорема C (оценка субоптимальности каскада)"),
    P("Доказательство опирается на лемму о разности производительностей [9], выражающую "
      "разрыв ценности двух политик через стационарное среднее преимущества:"),
    EQ(mrow(msub("V", "joint"), op(" − "), msub("V", "cascade"), op(" = "),
            mfrac("1", mrow("1", op(" − "), "λ")), " ",
            msub(op("E"), mrow("s", op(" ~ "), msubsup("μ", "J", "*"))),
            mdelim(mrow(msup("A", msub("π", "C")),
                        mdelim(mrow("s", op(", "), msub("π", "J"), mdelim("s")))), "[", "]")),
       "А.3"),
    P("Вне множества B нарушения условия C4 каскадная политика локально оптимальна, и "
      "преимущество неположительно; на B оно по модулю не превосходит 2R_max / (1 − λ). "
      "Разбивая ожидание по индикатору B и оценивая часть на B сверху, получаем оценку "
      "(2.11), линейную по стационарной массе нарушения μ*_J(B). Честная оговорка: "
      "эмпирически μ*_J(B) ≈ 0,90 (см. приложение В), поэтому правая часть (2.11) "
      "оценивается величиной порядка нескольких сотен и как абсолютная оценка вакуумна; "
      "содержательный результат даёт теорема D."),

    H2("А.5 Теорема D (точный разрыв ценности)"),
    P("Равенство (А.3) выполнено точно, а не только как оценка сверху. Поэтому величина "
      "разрыва ценности определяется не размером множества B, а средним преимуществом "
      "каскадной политики на нём. Эмпирическое усреднение этого преимущества пренебрежимо "
      "мало (порядка 6·10⁻⁵ на паре 14B/0.5B и 10⁻³ на паре 7B/1.5B), поэтому, несмотря на "
      "повсеместное нарушение C4, точный стационарно-взвешенный разрыв мал (+0,005 и +0,10 "
      "соответственно). Это и объясняет наблюдаемое практическое совпадение совместной и "
      "каскадной политик: нарушение C4 повсеместно, но доброкачественно."),

    H2("А.6 Теоремы 2.3 и 2.4"),
    P("Теорема 2.3 (слабое доминирование). Любая каскадная политика реализуема и в "
      "совместном пространстве, то есть Π_cascade ⊂ Π_joint. Оптимум по большему множеству "
      "политик не меньше оптимума по меньшему, поэтому V_joint(s) ≥ V_cascade(s) во всех "
      "состояниях s. Строгого доминирования это рассуждение не даёт, и эмпирически "
      "(раздел 4) разрыв статистически неотличим от нуля."),
    P("Теорема 2.4 (скаляризация Парето). Множество достижимых пар (η, D) по всем "
      "стохастическим стационарным политикам выпукло: рандомизация политик с весами w_i "
      "даёт пару"),
    EQ(mrow(mdelim(mrow("η", op(", "), "D")), op(" = "),
            mnary("∑", "i", None,
                  mrow(msub("w", "i"), " ",
                       mdelim(mrow(msub("η", "i"), op(", "), msub("D", "i")))))), "А.4"),
    P("то есть выпуклую комбинацию операционных характеристик. Линейный функционал η − κD "
      "достигает максимума на выпуклом множестве в опорной точке его верхней границы — "
      "Парето-фронта — с наклоном опорной прямой κ. Перебор κ ≥ 0 обходит фронт, что и "
      "утверждает теорема."),
])

APP_B = block([
    H1("Приложение Б. Архитектура репозитория и листинги"),
    P("Программный комплекс образован двумя стеками — историческим sp_samp/ с эталонными "
      "реализациями и тематическим jointadaspec/ (подпакеты core, mdp, inference, baselines, "
      "metrics, analysis, utils; таблица 3.1). Поток данных линеен: стадия 01 собирает "
      "трейсы, 02 оценивает MDP и решает его, 03 выполняет бенчмарк, 04 проверяет условия "
      "(рисунок 3.1). Ниже приведены ключевые фрагменты реализации."),
    LISTING("Листинг Б.1 — Поля одношаговой трейс-записи (jointadaspec/mdp/traces.py)",
            "record = {\n"
            "    \"state_index\": state_idx,    # дискретный индекс s = (i_H, i_K, k)\n"
            "    \"action_index\": action_idx,  # индекс действия (a_length, T)\n"
            "    \"k\": k,                       # позиция в черновике\n"
            "    \"accepted\": int(accepted),    # принят ли токен\n"
            "    \"proposed\": int(proposed),    # предложен ли токен черновиком\n"
            "    \"reward\": reward,             # мгновенная награда r(s, a)\n"
            "    \"next_state_index\": next_idx, # следующее состояние s'\n"
            "}"),
    LISTING("Листинг Б.2 — Оценка параметров MDP со сглаживанием (jointadaspec/mdp/estimation.py)",
            "@dataclass\n"
            "class EstimatedMDP:\n"
            "    transitions: sparse.csr_matrix   # (|S|*|A|) x |S|, ~1% ненулевых\n"
            "    rewards: np.ndarray              # (|S|, |A|)\n"
            "    visit_counts: np.ndarray         # (|S|, |A|)\n"
            "\n"
            "def estimate_mdp_parameters(traces_path, config):\n"
            "    # накопить N(s,a,s') и суммы наград из Parquet-трейсов;\n"
            "    # при visits >= nu_min — прямая оценка, иначе — сглаживание\n"
            "    # Лапласа с alpha_smooth на локальном носителе соседних состояний;\n"
            "    # rewards[s,a] = reward_sums[s,a] / visits\n"
            "    return EstimatedMDP(transitions, rewards, visit_counts)"),
    LISTING("Листинг Б.3 — Шаг онлайн-декодера (jointadaspec/inference/jointadaspec.py)",
            "H = entropy(q_probs); K = self.K_prev          # признаки состояния\n"
            "action_length, threshold = self.policy.get_action(H=H, K=K, k=k)\n"
            "if action_length == \"stop\":\n"
            "    self._verify_block(...)                     # нечёткая верификация, T\n"
            "else:\n"
            "    k = min(k + 1, self.policy.config.gamma_max) # продолжить черновик\n"
            "self.K_prev = kl_divergence(q_probs, p_probs)   # задержанное обновление K"),
])


def app_v(figs):
    return block([
    H1("Приложение В. Дополнительные экспериментальные данные"),
    P("Приложение содержит вспомогательные данные, дополняющие раздел 4: интерпретацию "
      "выученной политики, проверку теоретических условий и точные величины теорем C и D."),
    figs["advB"],
    figs["policy"],
    figs["surface"],
    P("Точные величины, лежащие в основе теорем C и D, вычислены по сошедшимся политикам "
      "обеих пар моделей (скрипт scripts/analyze_theorem_c_gap.py) и сведены в таблице В.1."),
    TBL_CAPTION("Таблица В.1 — Точные величины теорем C и D по обеим парам моделей"),
    TBL([["Величина", "14B/0.5B", "7B/1.5B"],
         ["Доля состояний с нарушением C4", "0,89", "0,89"],
         ["Стационарная масса μ*_J(B)", "0,90", "0,93"],
         ["Оценка теоремы C (правая часть)", "≈360", "≈374"],
         ["Точный разрыв ценности V_joint − V_cascade", "+0,005", "+0,10"],
         ["Доля состояний со слабым доминированием", "100 %", "100 %"]],
        [4400, 2477, 2477],
        align=["left", "center", "center"]),
    P("Таблица показывает ключевое: при почти повсеместном нарушении условия C4 (масса ≈ 0,9) "
      "оценка теоремы C вакуумна, однако точный разрыв ценности мал, а слабое доминирование "
      "(теорема 2.3) выполнено на всех состояниях. Развёртка по множителю κ (рисунок 4.4) "
      "подтверждает устойчивость равенства совместной и каскадной политик ко всему диапазону "
      "трейд-офф-параметра."),
    ])


APP_G = block([
    H1("Приложение Г. Манифесты воспроизводимости"),
    P("Каждый содержательный прогон сопровождается машинно-генерируемым манифестом в каталоге "
      "reports/manifests. Манифест фиксирует окружение (версии библиотек, ускоритель), "
      "состояние репозитория (хеш коммита Git и признак незакоммиченных изменений), полную "
      "разрешённую конфигурацию Hydra, список начальных значений генераторов и контрольные "
      "суммы SHA256 артефактов трейсов и политики. Этого достаточно, чтобы однозначно "
      "восстановить условия прогона. Сокращённый пример приведён в листинге Г.1."),
    LISTING("Листинг Г.1 — Фрагмент манифеста прогона 14B/0.5B (lock, 2026-05-14)",
            "{\n"
            "  \"generated_at\": \"2026-05-13T19:40:46Z\",\n"
            "  \"git_sha\": \"e1b32ea7…\", \"git_branch\": \"main\", \"git_dirty\": true,\n"
            "  \"nvidia_smi\": \"NVIDIA GeForce RTX 5090, 32607 MiB\",\n"
            "  \"python\": \"3.12.3\",\n"
            "  \"torch_version\": \"2.9.1+cu128\", \"transformers_version\": \"4.57.3\",\n"
            "  \"scipy_version\": \"1.17.0\", \"numpy_version\": \"2.4.2\",\n"
            "  \"seed_list\": [42, 43, 44],\n"
            "  \"policy_path\": \"outputs/…/02_solve/policy.npz\",\n"
            "  \"policy_npz_sha256\": \"f0f1e09c0b41cfc9…\",\n"
            "  \"hydra_config\": {\"H_max\": 6.0, \"K_max\": 8.0, \"gamma_max\": 8,\n"
            "                    \"N_H\": 20, \"N_K\": 20, \"lambda_discount\": 0.99,\n"
            "                    \"kappa\": 1.0, \"alpha_smooth\": 1.0, \"nu_min\": 5}\n"
            "}"),
    P("Манифест сохраняется автоматически стадией scripts/05_write_manifest.py и привязывает "
      "результаты раздела 4 к конкретному состоянию кода и данных. Отдельные вендорные ядра "
      "CUDA сохраняют предупреждение о неполной воспроизводимости; в этом случае первичной "
      "записью считаются именно манифест и список начальных значений, а малый числовой "
      "дрейф признаётся допустимым."),
])


# ==========================================================================
# СБОРКА ПАКЕТА
# ==========================================================================
def build():
    work = os.path.join(HERE, "_newera_build")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    with zipfile.ZipFile(BASE) as z:
        names = z.namelist()
        z.extractall(work)

    doc_path = os.path.join(work, "word", "document.xml")
    base_xml = open(doc_path, encoding="utf-8").read()

    m_open = re.search(r"<w:document\b[^>]*>", base_xml)
    doc_open = m_open.group(0)
    m_sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", base_xml, re.S)
    sect_pr = m_sect.group(0)

    media_dir = os.path.join(work, "word", "media")
    os.makedirs(media_dir, exist_ok=True)
    rels_path = os.path.join(work, "word", "_rels", "document.xml.rels")
    rel_tpl = ('<Relationship Id="{rid}" '
               'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
               'Target="media/{name}"/>')
    embedded = []                       # (src_path, media_name) для дозаписи в zip
    state = {"rels": open(rels_path, encoding="utf-8").read(), "rid": 100}

    def _png_size(path):
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")

    def register_png(src_path, caption, cx_emu=5_760_000):
        """Встроить готовый PNG: вернуть блок «рисунок + подпись» либо подпись-фоллбек."""
        size = _png_size(src_path) if (src_path and os.path.exists(src_path)) else None
        if not size:
            return FIG_CAPTION(caption)
        pw, ph = size
        cy = int(cx_emu * ph / pw)
        n = state["rid"]; state["rid"] += 1
        rid, name = f"rId{n}", f"image_newera{n}.png"
        dst = os.path.join(media_dir, name)
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copyfile(src_path, dst)
        state["rels"] = state["rels"].replace(
            "</Relationships>", rel_tpl.format(rid=rid, name=name) + "</Relationships>")
        embedded.append((dst, name))
        return block([FIG_IMAGE(cx_emu, cy, rid, n, caption.split(" —")[0]),
                      FIG_CAPTION(caption)])

    # Рисунки 1.1 и 3.1 — генерируются matplotlib во временный путь, затем регистрируются
    f1 = os.path.join(media_dir, "_tmp_fig1.png")
    fig_block = (register_png(f1,
                 "Рисунок 1.1 — Цикл спекулятивного декодирования: черновик предлагает блок "
                 "токенов, целевая модель проверяет их одним проходом, принятый префикс "
                 "дополняется одним ресемплом", 5_400_000)
                 if make_figure(f1) else FIG_CAPTION(
                 "Рисунок 1.1 — Цикл спекулятивного декодирования (черновик → проверка → "
                 "принятый префикс + ресемпл)"))
    f3 = os.path.join(media_dir, "_tmp_fig3.png")
    fig3_block = (register_png(f3,
                  "Рисунок 3.1 — Архитектура программного комплекса JointAdaSpec и поток "
                  "данных между стадиями конвейера", 5_760_000)
                  if make_figure_pipeline(f3) else FIG_CAPTION(
                  "Рисунок 3.1 — Архитектура программного комплекса JointAdaSpec и поток "
                  "данных между стадиями конвейера"))

    # Рисунки Раздела 4 — готовые PNG из reports/thesis_figs/png_slides
    FIGDIR = os.path.join(HERE, "..", "reports", "thesis_figs", "png_slides")
    def _r4(fname, caption):
        return register_png(os.path.join(FIGDIR, fname), caption, 5_760_000)
    figs4 = {
        "pareto": _r4("fig1_pareto.png",
            "Рисунок 4.1 — Парето-фронт «скорость — качество» для обеих пар моделей: совместная "
            "и каскадная политики совпадают по качеству, тогда как прямая генерация целевой "
            "моделью остаётся самой быстрой"),
        "win3": _r4("fig_3win_robustness.png",
            "Рисунок 4.2 — Прирост точности JointAdaSpec относительно прямой генерации на трёх "
            "независимых отложенных окнах пары 7B/1.5B"),
        "E": _r4("fig_E_adaptivity_ablation.png",
            "Рисунок 4.3 — Сравнение JointAdaSpec с фиксированными порогами fuzzy_sd по качеству "
            "и скорости (пара 14B/0.5B, n = 300)"),
        "kappa": _r4("fig_bonus_kappa_sweep.png",
            "Рисунок 4.4 — Развёртка по множителю κ: пропускная способность почти постоянна, "
            "совместная и каскадная политики близки при каждом κ (пара 7B/1.5B)"),
        "G": _r4("fig_G_acceptance_em.png",
            "Рисунок 4.5 — Чистое изменение точности по терцилям доли принятий: выигрыш при "
            "умеренной и проигрыш при высокой доле принятий"),
    }

    # Рисунки Приложения В — готовые PNG из reports/thesis_figs/png_slides
    figsV = {
        "advB": _r4("fig_D_advantage_on_B.png",
            "Рисунок В.1 — Распределение преимущества каскадной политики на множестве "
            "нарушения условия C4: масса сосредоточена около нуля (теорема D)"),
        "policy": _r4("fig_policy_interpretability.png",
            "Рисунок В.2 — Интерпретация выученной политики: вероятность продолжения "
            "черновика и средний выбранный порог в зависимости от признаков состояния"),
        "surface": _r4("fig3_threshold_surface.png",
            "Рисунок В.3 — Пороговая поверхность выученной политики в координатах "
            "(энтропия H, дивергенция K)"),
    }

    open(rels_path, "w", encoding="utf-8").write(state["rels"])

    body = (TITLE + REFERAT + SODER + VVEDENIE + razdel1(fig_block)
            + RAZDEL2 + razdel3(fig3_block) + razdel4(figs4) + ZAKL + SPISOK
            + APP_A + APP_B + app_v(figsV) + APP_G)

    new_xml = f"{doc_open}<w:body>{body}{sect_pr}</w:body></w:document>"
    open(doc_path, "w", encoding="utf-8").write(new_xml)

    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(os.path.join(work, name), name)
        for p, nm in embedded:
            z.write(p, f"word/media/{nm}")

    shutil.rmtree(work)
    n_p = new_xml.count("<w:p>") + new_xml.count("<w:p ")
    n_tbl = new_xml.count("<w:tbl>")
    print(f"Готово: {OUT}")
    print(f"Абзацев: ~{n_p}, таблиц: {n_tbl}, рисунков встроено: {len(embedded)}")
    print(f"Размер: {os.path.getsize(OUT)} байт")


if __name__ == "__main__":
    build()
