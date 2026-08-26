# Persian B.Sc. report (XeLaTeX)

Project report for Amirreza Nemati, University of Mazandaran.
Source prose is the Markdown files in `report/1.md`–`report/6.md`.
Citations are unified in `references.bib` (one bibliography at the end).

## Compile

Requires TeX Live with `xelatex`, `biber`, and `extbook`, plus the fonts
**B Nazanin**, **Nazanin Bold**, **B Titr**, and **IranNastaliq**
(already installed for the author under the Windows user font folder).

From this directory:

```powershell
.\compile.ps1
```

or:

```powershell
python tools\convert_md.py
xelatex -interaction=nonstopmode main.tex
biber main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

The PDF is `main.pdf`.

Persian body is 14 pt B Nazanin. Latin/English is Calibri at about 12.5 pt
(`Scale=0.893`). Compile with XeLaTeX only.
