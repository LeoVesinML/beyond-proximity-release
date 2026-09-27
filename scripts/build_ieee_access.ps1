$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$paper = Join-Path $repo "papers\ieee-access"
$miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue) -and (Test-Path $miktex)) {
    $env:Path = "$miktex;$env:Path"
}

python -B (Join-Path $repo "scripts\sync_ieee_access_paper.py")
if ($LASTEXITCODE -ne 0) { throw "IEEE Access synchronization failed" }

Push-Location $paper
try {
    if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        pdflatex -halt-on-error -interaction=nonstopmode main.tex
        if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed" }
        bibtex main
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed" }
        pdflatex -halt-on-error -interaction=nonstopmode main.tex
        if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed" }
        pdflatex -halt-on-error -interaction=nonstopmode main.tex
        if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed" }
    } else {
        $tectonic = Get-ChildItem -Path (Join-Path $env:USERPROFILE ".codex\plugins\cache\openai-bundled\latex") `
            -Recurse -Filter tectonic.exe -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if (-not $tectonic) { throw "Neither pdflatex nor bundled Tectonic is available" }
        & $tectonic.FullName -X compile --outdir . --outfmt pdf --untrusted main.tex
        if ($LASTEXITCODE -ne 0) { throw "Tectonic build failed" }
    }
} finally {
    Pop-Location
}

python -B (Join-Path $repo "scripts\check_ieee_access_paper.py")
if ($LASTEXITCODE -ne 0) { throw "IEEE Access compliance check failed" }
Write-Host "IEEE Access PDF: $paper\main.pdf"
