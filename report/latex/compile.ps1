$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python tools\convert_md.py
if ($LASTEXITCODE -ne 0) { throw "Markdown conversion failed" }

xelatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw "xelatex pass 1 failed" }
biber main
if ($LASTEXITCODE -ne 0) { throw "biber failed" }
xelatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw "xelatex pass 2 failed" }
xelatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw "xelatex pass 3 failed" }

Write-Host "Built $(Join-Path $PSScriptRoot 'main.pdf')"
