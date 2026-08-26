# -*- coding: utf-8 -*-
"""Convert report/*.md chapters into XePersian LaTeX chapters."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # report/
MD_DIR = ROOT
OUT_CH = Path(__file__).resolve().parents[1] / "chapters"
OUT_FM = Path(__file__).resolve().parents[1] / "frontmatter"

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

CHAPTER_TITLES = {
    1: "مقدمه و کلیات پژوهش",
    2: "مبانی نظری، ادبیات موضوع و نقد ساختاری مدل مرجع",
    3: "داده‌ها، توپولوژی گراف‌های زیستی و پروتکل ارزیابی دوگانه",
    4: "فرمولاسیون ریاضی و معماری روش‌های پیشنهادی",
    5: "پروتکل آموزش، آزمایش‌های تجربی و تحلیل جامع نتایج",
    6: "بحث، تحلیل پاشنه‌های آشیل، نتیجه‌گیری و افق‌های آینده",
}

# footnote n -> ('note', tex) | ('cite', key) | ('both', key, note)
FOOTNOTES = {
    1: {
        1: ("note", r"زیست‌مولکولی (\lr{Biomolecular})"),
        2: ("note", r"زیست‌شناسی سامانه‌ها (\lr{Systems Biology})"),
        3: ("note", r"کشف و بازکاربردپذیری دارو (\lr{Drug Discovery \& Repurposing})"),
        4: ("note", r"غربال‌گری با توان عملیاتی بالا (\lr{High-Throughput Screening})"),
        5: ("note", r"هزینه‌های بالای مراحل پیش‌بالینی و نرخ شکست در فازهای بالینی."),
        6: ("note", r"غربال‌گری مجازی (\lr{Virtual Screening})"),
        7: (
            "note",
            r"فارماکودینامیک (\lr{Pharmacodynamics}): مطالعه اثرات بیوشیمیایی و فیزیولوژیکی دارو بر بدن.",
        ),
        8: (
            "note",
            r"فارماکوکینتیک (\lr{Pharmacokinetics}): مطالعه جذب، توزیع، متابولیسم و دفع دارو.",
        ),
        9: (
            "note",
            r"پلی‌فارماسی (\lr{Polypharmacy}): مصرف همزمان چندین دارو توسط یک بیمار.",
        ),
        10: (
            "note",
            r"اینتراکتوم (\lr{Interactome}): مجموعه کامل برهم‌کنش‌های مولکولی در یک سلول یا ارگانیسم.",
        ),
        11: (
            "note",
            r"مطالعات هم‌خوانی سراسر ژنوم (\lr{Genome-Wide Association Studies}).",
        ),
        12: ("note", r"یادگیری بازنمایی روی گراف (\lr{Graph Representation Learning})."),
        13: ("note", r"پیش‌بینی پیوند (\lr{Link Prediction})."),
        14: ("cite", "barabasi1999"),
        15: ("note", r"شبکه‌های عصبی گراف (\lr{Graph Neural Networks})."),
        16: (
            "both",
            "kipf2017gcn",
            r"شبکه‌های پیچشی گراف (\lr{Graph Convolutional Networks}).",
        ),
        17: ("cite", "huang2020skipgnn"),
        18: ("cite", "li2023evaluating"),
        19: ("cite", "zhou2009predicting"),
        20: (
            "both",
            "ioffe2015batch",
            r"نرمال‌سازی دسته‌ای (\lr{Batch Normalization}).",
        ),
        21: (
            "note",
            r"فروپاشی بازنمایی (\lr{Representation Collapse}): پدیده‌ای که در آن امبدینگ‌های گره‌های مختلف به یک بردار یکسان همگرا می‌شوند.",
        ),
        22: (
            "both",
            "wang2020uniformity",
            r"یکنواختی در فضای بازنمایی (\lr{Uniformity}).",
        ),
        23: ("cite", "brody2022gatv2"),
        24: (
            "note",
            r"جلوگیری از نشت اطلاعات در انتخاب آستانه (\lr{Threshold Leakage Prevention}).",
        ),
    },
    2: {
        1: (
            "both",
            "gilmer2017mpnn",
            r"شبکه‌های عصبی پیام‌رسان (\lr{Message Passing Neural Networks}).",
        ),
        2: ("cite", "kipf2017gcn"),
        3: ("both", "li2018deeper", r"بیش‌هموارسازی (\lr{Over-smoothing})."),
        4: ("both", "watts1998smallworld", r"خاصیت جهان‌کوچک (\lr{Small-World Property})."),
        5: ("cite", "zhou2009predicting"),
        6: ("cite", "huang2020skipgnn"),
        7: ("cite", "barabasi1999"),
        8: ("cite", "li2023evaluating"),
        9: (
            "both",
            "huang2020skipgnncode",
            r"مخزن کد مرجع: \pcode{github.com/kexinhuang12345/SkipGNN}",
        ),
    },
}

BRACKET_CITES = {
    1: {
        14: "barabasi1999",
        17: "huang2020skipgnn",
        18: "li2023evaluating",
        19: "zhou2009predicting",
        23: "brody2022gatv2",
    },
    3: {
        1: "huang2020skipgnn",
        2: "zhou2009predicting",
        3: "li2023evaluating",
        15: "bionsnap",
        16: "luck2020huri",
        17: "pinero2020disgenet",
    },
    4: {24: "brody2022gatv2"},
    5: {
        3: "li2023evaluating",
        22: "wang2020uniformity",
        25: "maaten2008tsne",
    },
    6: {
        1: "landrum2016rdkit",
        2: "chithrananda2020chemberta",
        3: "lin2023esm2",
        4: "elnaggar2022prottrans",
        5: "schlichtkrull2018rgcn",
        6: "wang2020uniformity",
    },
}

FIGURES = {
    "fig1_uniform_vs_hard_auprc.png": (
        "fig:uniform-hard-auprc",
        "مقایسه همزمان AUPRC در بانک یکنواخت (چپ) و بانک سخت (راست) برای هفت مدل. میله‌ها میانگین و خطوط خطا انحراف معیار در میان سه سید هستند.",
        "مقایسه AUPRC یکنواخت و سخت",
    ),
    "fig2_ablation_ladder.png": (
        "fig:ablation-ladder",
        "نردبان ابلیشن چهارمرحله‌ای مدل AMS روی دیتاست DTI در معیار Hard AUPRC.",
        "نردبان ابلیشن AMS در DTI",
    ),
    "fig3_missing_edge_robustness.png": (
        "fig:robustness",
        "مقاومت مدل‌ها در برابر حذف تصادفی یال‌های آموزشی در دیتاست DTI.",
        "مقاومت در برابر حذف یال",
    ),
    "fig4_precision_recall_curves.png": (
        "fig:pr-curves",
        "منحنی‌های دقت-بازخوانی در آزمون یکنواخت، سید ۴۲. در نسخۀ فعلی مخزن، آرایه‌های ذخیره‌شده مربوط به مدل AMS هستند؛ رتبه‌بندی هفت مدل در شکل~\\ref{fig:uniform-hard-auprc} آمده است.",
        "منحنی‌های دقت-بازخوانی",
    ),
    "fig5_hard_auprc_heatmap.png": (
        "fig:heatmap",
        "نقشۀ حرارتی میانگین Hard AUPRC (دیتاست $\\times$ مدل) در سه سید.",
        "نقشۀ حرارتی Hard AUPRC",
    ),
    "fig6_delta_hard_auprc.png": (
        "fig:delta-ams",
        "دلتای Hard AUPRC هر مدل نسبت به AMS ($\\Delta = \\text{Model} - \\text{AMS}$).",
        "دلتای Hard AUPRC نسبت به AMS",
    ),
    "fig7_hard_auprc_seeds.png": (
        "fig:seed-box",
        "توزیع بین‌سیدی Hard AUPRC برای تمام مدل‌ها (سیدهای ۴۲، ۱۲۳ و ۷).",
        "توزیع بین‌سیدی Hard AUPRC",
    ),
    "fig8_uniform_vs_hard_auroc.png": (
        "fig:auroc",
        "مقایسه AUROC در بانک یکنواخت و بانک سخت برای هفت مدل.",
        "مقایسه AUROC یکنواخت و سخت",
    ),
    "fig9_learning_curves.png": (
        "fig:learning-curves",
        "منحنی‌های یادگیری: روند Val AUPRC در طول اپاک‌های آموزش (میانگین $\\pm$ انحراف معیار).",
        "منحنی‌های یادگیری",
    ),
    "fig10_f1_at_tau.png": (
        "fig:f1-tau",
        "شاخص F1 در آستانۀ منجمد اعتبارسنجی ($\\tau^{\\ast}$). هوریستیک به‌دلیل نداشتن آستانۀ احتمالی حذف شده است.",
        "شاخص F1 در آستانۀ بهینه",
    ),
}

TSNE_PANEL = r"""
\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{fig_tsne_DTI_gcn_seed42.png}\\[2pt]
{\small \lr{GCN}}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{fig_tsne_DTI_skipgnn_seed42.png}\\[2pt]
{\small \lr{SkipGNN}}
\end{minipage}\\[8pt]
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{fig_tsne_DTI_ams_seed42.png}\\[2pt]
{\small \lr{AMS-SkipGNN}}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}\centering
\includegraphics[width=\linewidth]{fig_tsne_DTI_gat_seed42.png}\\[2pt]
{\small \lr{SkipGATv2}}
\end{minipage}
\caption[تصویرسازی \lr{t-SNE} لایۀ آخر]{تصویرسازی \lr{t-SNE} از امبدینگ لایۀ آخر روی دیتاست \lr{DTI} (سید ۴۲). رنگ‌ها نوع موجودیت را در گراف دوقطبی نشان می‌دهند.}
\label{fig:tsne-dti}
\end{figure}
"""


def strip_tail(text: str) -> str:
    cut_markers = [
        "\n---\n\n[^",
        "\n---\n\n**مراجع",
        "\n**مراجع فصل",
        "\n# فهرست مراجع",
        "\n[^1]:",
    ]
    positions = []
    for m in cut_markers:
        i = text.find(m)
        if i != -1:
            positions.append(i)
    if positions:
        text = text[: min(positions)].rstrip() + "\n"
    return text


def extract_abstract(ch1: str) -> tuple[str, str, str]:
    m = re.match(r"# چکیده\n\n(.+?)\n\n---\n+", ch1, flags=re.S)
    if not m:
        raise SystemExit("Could not find abstract in 1.md")
    body = ch1[m.end() :]
    abs_md = m.group(1).strip()
    kw = ""
    if "**واژگان کلیدی:**" in abs_md:
        abs_md, kw = abs_md.split("**واژگان کلیدی:**", 1)
        abs_md = abs_md.strip()
        kw = kw.strip()
    return abs_md, kw, body


def split_math(text: str):
    pattern = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$]+\$)")
    parts = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(("text", text[last : m.start()]))
        kind = "display" if m.group().startswith("$$") else "inline"
        parts.append((kind, m.group()))
        last = m.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def escape_text(s: str) -> str:
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)#", r"\\#", s)
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    return s


def wrap_code(s: str) -> str:
    def repl(m):
        inner = m.group(1).replace("\\", r"\\").replace("_", r"\_")
        return r"\pcode{" + inner + "}"

    return re.sub(r"`([^`]+)`", repl, s)


def apply_cites(s: str, ch: int) -> str:
    fn = FOOTNOTES.get(ch, {})

    def fn_repl(m):
        n = int(m.group(1))
        spec = fn.get(n)
        if spec is None:
            return m.group(0)
        if spec[0] == "note":
            return r"\footnote{" + spec[1] + "}"
        if spec[0] == "cite":
            return r"\cite{" + spec[1] + "}"
        return r"\footnote{" + spec[2] + r"}\cite{" + spec[1] + "}"

    s = re.sub(r"\[\^(\d+)\]", fn_repl, s)

    br = BRACKET_CITES.get(ch, {})

    def br_repl(m):
        n = int(m.group(1))
        key = br.get(n)
        if key is None:
            return m.group(0)
        return r"\cite{" + key + "}"

    s = re.sub(r"\[(\d+)\]", br_repl, s)
    return s


def wrap_latin(s: str) -> str:
    """Put Latin tokens in \\lr{...} so Calibri is used (B Nazanin has almost no Latin glyphs)."""
    pieces = re.split(
        r"(\\(?:pcode|lr|ref|label|cite|eqref|texttt|path)\{[^}]*\})",
        s,
    )
    out = []
    word = re.compile(
        r"(?<![A-Za-z\\])[A-Za-z0-9+\-]*[A-Za-z][A-Za-z0-9+\-_]*"
        r"(?:\s+[A-Za-z0-9+\-]*[A-Za-z][A-Za-z0-9+\-_]*)*"
    )
    for p in pieces:
        if p.startswith("\\"):
            out.append(p)
        else:
            out.append(word.sub(lambda m: r"\lr{" + m.group(0) + "}", p))
    return "".join(out)


def inline_format(s: str, ch: int) -> str:
    s = wrap_code(s)
    s = wrap_latin(s)
    s = apply_cites(s, ch)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    s = escape_text(s)
    return s


def latinize_text_cmd(s: str) -> str:
    return re.sub(
        r"\\text\{(\s*)([A-Za-z][A-Za-z0-9 +\-]*)\}",
        r"\\text{\1\\lr{\2}}",
        s,
    )


def convert_display(block: str) -> str:
    inner = block.strip()
    inner = inner[2:-2].strip()
    label = None
    tm = re.match(r"\\tag\{([^}]+)\}\s*", inner)
    if tm:
        label = tm.group(1)
        inner = inner[tm.end() :]
    inner = inner.replace("★", r"\star")
    inner = latinize_text_cmd(inner)
    if label:
        return (
            "\n\\begin{equation}\n"
            f"\\label{{eq:{label}}}\n"
            f"{inner}\n"
            "\\end{equation}\n"
        )
    return "\n\\[\n" + inner + "\n\\]\n"


def split_md_table_row(ln: str) -> list[str]:
    """Split a markdown table row without breaking math that contains |."""
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"@@CELLMATH{len(held) - 1}@@"

    protected = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$]+\$", hold, ln)
    cells = [c.strip() for c in protected.strip().strip("|").split("|")]
    out = []
    for c in cells:
        for i, blk in enumerate(held):
            c = c.replace(f"@@CELLMATH{i}@@", blk)
        out.append(c)
    return out


def _cell_plain_len(c: str) -> int:
    t = re.sub(r"\$[^$]*\$", "", c)
    t = re.sub(r"`[^`]+`", "x", t)
    return len(t)


def convert_table(md: str, ch: int, caption: str | None, label: str | None) -> str:
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return md
    header = split_md_table_row(lines[0])
    rows = [split_md_table_row(ln) for ln in lines[2:]]
    ncols = len(header)
    all_cells = header + [c for row in rows for c in row]
    max_len = max((_cell_plain_len(c) for c in all_cells), default=0)
    wrap = max_len > 48 or (ncols <= 3 and max_len > 28)
    out = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize"]
    if caption:
        out.append("\\caption{" + caption + "}")
    if label:
        out.append("\\label{" + label + "}")
    if wrap:
        if ncols == 3:
            spec = r">{\hspace{0pt}}p{0.22\textwidth}>{\hspace{0pt}}X>{\hspace{0pt}}X"
        elif ncols == 2:
            spec = r">{\hspace{0pt}}X>{\hspace{0pt}}X"
        else:
            spec = r">{\hspace{0pt}}X" * ncols
        out.append("\\begin{tabularx}{\\textwidth}{" + spec + "}")
    else:
        if ncols >= 6:
            out.append("\\begin{adjustbox}{max width=\\textwidth}")
        out.append("\\begin{tabular}{" + ("l" * ncols) + "}")
    out.append("\\toprule")
    out.append(" & ".join(convert_inline_line(c, ch) for c in header) + r" \\")
    out.append("\\midrule")
    for row in rows:
        row = (row + [""] * ncols)[:ncols]
        cells = []
        for c in row:
            c = c.replace("★", r"$\star$")
            cells.append(convert_inline_line(c, ch))
        out.append(" & ".join(cells) + r" \\")
    out.append("\\bottomrule")
    if wrap:
        out.append("\\end{tabularx}")
    else:
        out.append("\\end{tabular}")
        if ncols >= 6:
            out.append("\\end{adjustbox}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


HEADING_RE = re.compile(
    r"^(#{2,4})\s+([۰-۹0-9\-]+)\.?\s*(.*)$"
)
FIG_FILE_RE = re.compile(r"`(fig\d+_[a-z0-9_]+.png)`")
TABLE_CAPTION_RE = re.compile(r"^\*\*جدول\s+([۰-۹0-9\-]+):\*\*\s*(.*)$")


def heading_to_tex(hashes: str, num: str, title: str, ch: int) -> str:
    level = len(hashes)
    title = FIG_FILE_RE.sub("", title)
    title = re.sub(r"\(\s*\)", "", title).strip(" :")
    title = inline_format(title.strip(), ch)
    latin_num = num.translate(PERSIAN_DIGITS)
    cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
    return f"\\{cmd}{{{title}}}\\label{{sec:{latin_num}}}\n"


def convert_paragraphs(text: str, ch: int) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    pending_caption = None
    pending_label = None
    in_list = False
    inserted_tsne = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("\\end{itemize}")
            in_list = False

    while i < n:
        raw = lines[i]
        ln = raw.rstrip()
        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue

        if ln.strip() == "---":
            close_list()
            i += 1
            continue

        # skip leftover H1
        if ln.startswith("# "):
            i += 1
            continue

        tm = TABLE_CAPTION_RE.match(ln.strip())
        if tm:
            close_list()
            pending_caption = convert_inline_line(tm.group(2).strip(), ch)
            pending_label = "tab:" + tm.group(1).translate(PERSIAN_DIGITS)
            i += 1
            continue

        if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*-+", lines[i + 1]):
            close_list()
            block = [ln]
            i += 1
            while i < n and lines[i].lstrip().startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            out.append(
                convert_table("\n".join(block), ch, pending_caption, pending_label)
            )
            pending_caption = None
            pending_label = None
            continue

        hm = HEADING_RE.match(ln)
        if hm:
            close_list()
            out.append(heading_to_tex(hm.group(1), hm.group(2), hm.group(3), ch))
            # maybe insert figure after heading that names a png
            fm = FIG_FILE_RE.search(ln)
            if fm:
                fname = fm.group(1)
                if fname in FIGURES:
                    lab, cap, *rest = FIGURES[fname]
                    cap = convert_inline_line(cap, ch)
                    cap_cmd = f"\\caption{{{cap}}}"
                    if rest:
                        short = convert_inline_line(rest[0], ch)
                        cap_cmd = f"\\caption[{short}]{{{cap}}}"
                    out.append(
                        "\\begin{figure}[htbp]\n"
                        "\\centering\n"
                        f"\\includegraphics[width=0.98\\textwidth]{{{fname}}}\n"
                        f"{cap_cmd}\n"
                        f"\\label{{{lab}}}\n"
                        "\\end{figure}\n"
                    )
            if (not inserted_tsne) and "t-SNE" in ln and ch == 5:
                inserted_tsne = True
                out.append(TSNE_PANEL)
            i += 1
            continue

        if ln.lstrip().startswith("- "):
            if not in_list:
                out.append("\\begin{itemize}")
                in_list = True
            item = ln.lstrip()[2:]
            out.append("\\item " + convert_inline_line(item, ch))
            i += 1
            continue

        close_list()
        # accumulate paragraph
        para = [ln]
        i += 1
        while i < n:
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if nxt.startswith("#") or nxt.lstrip().startswith("|") or nxt.lstrip().startswith("- ") or nxt.strip() == "---":
                break
            if TABLE_CAPTION_RE.match(nxt.strip()):
                break
            if nxt.startswith("$$") or (para and para[-1].strip().endswith("$$")):
                para.append(nxt)
                i += 1
                continue
            if nxt.startswith("$$") or nxt.strip().startswith("$$"):
                break
            para.append(nxt)
            i += 1
        out.append(convert_inline_mixed("\n".join(para), ch))

    close_list()
    return "\n".join(out)


def convert_inline_mixed(text: str, ch: int) -> str:
    return convert_inline_line(text, ch)


def normalize_persian(s: str) -> str:
    """B Nazanin lacks several Arabic punctuation glyphs and U+0654."""
    s = s.replace("هٔ", "ۀ")
    s = s.replace("ه\u0654", "ۀ")
    s = s.replace("\u0654", "")
    s = s.replace("٬", ",")
    s = s.replace("٫", ".")
    s = s.replace("٪", "%")
    return s


def convert_inline_line(text: str, ch: int) -> str:
    displays: list[str] = []
    inlines: list[str] = []

    def disp_repl(m: re.Match) -> str:
        displays.append(convert_display(m.group(0)))
        return f"\uE012{len(displays) - 1}\uE013"

    def inl_repl(m: re.Match) -> str:
        inner = m.group(0)[1:-1].strip()
        # Lone \to between Persian words is a prose arrow, not LTR math.
        if inner in (r"\to", r"\rightarrow", r"\longrightarrow"):
            inlines.append(r"\faarrow{}")
        else:
            inlines.append(latinize_text_cmd(m.group(0)))
        return f"\uE010{len(inlines) - 1}\uE011"

    text = normalize_persian(text)
    text = re.sub(r"\$\$[\s\S]*?\$\$", disp_repl, text)
    text = re.sub(r"\$[^$]+\$", inl_repl, text)
    text = inline_format(text, ch)
    text = text.replace("±", r"$\pm$")
    text = text.replace("→", r"\faarrow{}")
    text = text.replace("×", r"$\times$")
    text = text.replace("•", r"\textbullet{}")
    # ASCII hyphen: B Nazanin has no U+2013 from TeX -- ligatures.
    text = text.replace("—", " - ")
    text = text.replace("–", "-")
    for i in range(len(displays) - 1, -1, -1):
        text = text.replace(f"\uE012{i}\uE013", displays[i])
    for i in range(len(inlines) - 1, -1, -1):
        text = text.replace(f"\uE010{i}\uE011", inlines[i])
    return text


def write_abstract(abs_md: str, kw: str) -> None:
    paragraphs = [p.strip() for p in abs_md.split("\n\n") if p.strip()]
    body = "\n\n".join(convert_inline_mixed(p, 1) for p in paragraphs)
    keywords = inline_format(kw, 1)
    tex = rf"""\chapter*{{چکیده}}
\addcontentsline{{toc}}{{chapter}}{{چکیده}}
\markboth{{چکیده}}{{چکیده}}

{body}

\vspace{{1em}}
\noindent\textbf{{واژگان کلیدی:}} {keywords}
"""
    (OUT_FM / "abstract.tex").write_text(tex, encoding="utf-8")


EN_ABSTRACT = r"""
\clearpage
\begin{latin}
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{\lr{Abstract}}
\markboth{Abstract}{Abstract}

\setstretch{1.15}
\noindent
\textbf{Title:} Redesigning Skip-Graph Neural Networks for Predicting Biomolecular Interactions

\vspace{0.8em}
Predicting biomolecular interactions---including drug--target, drug--drug, protein--protein, and gene--disease links---is a core problem in systems biology and computational drug discovery. SkipGNN (Huang et al., 2020) injects a 2-hop skip graph into a graph neural network and reports strong results on standard benchmarks. A theoretical and empirical audit in this work identifies four structural limits of that model: a geometric dead-end of the $A^2$ operator on bipartite graphs, loss of density information under binarization, a linear decoder bottleneck, and degree bias in a uniform evaluation protocol.

Four models are implemented in one leak-free pipeline: AMS-SkipGNN (resource-allocation skip, adaptive gating, four-way tensor decoder), SkipGATv2 (sparse dynamic edge attention), ThreeHopSkipGNN (1/2/3-hop multi-scale paths), and ContrastiveSkipGNN (symmetric InfoNCE). Evaluation uses a dual-bank protocol (uniform negatives and degree-aware hard negatives) on DTI, DDI, PPI, and GDI with three random seeds.

On the hard bank the proposed models improve over SkipGNN: on DDI, AMS reaches $0.912$ versus $0.724$; on PPI, SkipGATv2 reaches $0.778$ versus $0.622$; on DTI and GDI the gains are about $+0.10$ to $+0.12$ percentage points. On DTI, AMS, SkipGATv2, and 3-hop are statistically on par (overlapping standard-deviation intervals). On GDI, the analytical resource-allocation heuristic attains $0.842$ and outperforms every neural model---an important limit of identity-feature GNNs on large sparse graphs.

\vspace{0.8em}
\noindent\textbf{Keywords:} link prediction, graph neural networks, SkipGNN, bipartite graphs, resource allocation, dual-bank evaluation
\end{latin}
"""


def main() -> None:
    OUT_CH.mkdir(parents=True, exist_ok=True)
    OUT_FM.mkdir(parents=True, exist_ok=True)

    ch1_raw = normalize_persian(unicodedata.normalize("NFC", (MD_DIR / "1.md").read_text(encoding="utf-8")))
    abs_md, kw, ch1_body = extract_abstract(ch1_raw)
    write_abstract(abs_md, kw)
    (OUT_FM / "abstract-en.tex").write_text(EN_ABSTRACT.strip() + "\n", encoding="utf-8")

    for n in range(1, 7):
        raw = normalize_persian(unicodedata.normalize("NFC", (MD_DIR / f"{n}.md").read_text(encoding="utf-8")))
        if n == 1:
            raw = ch1_body
        raw = strip_tail(raw)
        # drop the markdown H1 chapter title; LaTeX \chapter in main.tex
        raw = re.sub(r"^# فصل.*\n+", "", raw)
        body = convert_paragraphs(raw, n)
        if n == 2:
            body = body.replace("جدول ۵-۱", "جدول~\\ref{tab:5-3}")
            body = body.replace(
                "امبدینگ‌های Node2Vec",
                r"امبدینگ‌های \lr{Node2Vec}\cite{grover2016node2vec}",
            )
        if n == 4:
            body = body.replace(
                "برخلاف GAT نسخه اول",
                r"برخلاف GAT نسخه اول\cite{velickovic2018gat}",
            )
        header = f"% Auto-generated from report/{n}.md — do not edit by hand.\n"
        (OUT_CH / f"ch0{n}.tex").write_text(header + body + "\n", encoding="utf-8")
        print(f"wrote chapters/ch0{n}.tex ({len(body)} chars)")


if __name__ == "__main__":
    main()
