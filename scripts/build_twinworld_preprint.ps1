$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$paperDir = Join-Path $repo "papers\twinworld"
$miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue) -and (Test-Path $miktex)) {
    $env:Path = "$miktex;$env:Path"
}

function Find-Tectonic {
    $command = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $bundledRoot = Join-Path $env:USERPROFILE ".codex\plugins\cache\openai-bundled\latex"
    if (Test-Path $bundledRoot) {
        $bundled = Get-ChildItem -Path $bundledRoot -Recurse -Filter tectonic.exe -File `
            -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($bundled) {
            return $bundled.FullName
        }
    }

    return $null
}

$tectonic = Find-Tectonic
Push-Location $paperDir
try {
    if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        pdflatex -halt-on-error -interaction=nonstopmode main_preprint.tex
        if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed" }
        bibtex main_preprint
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed" }
        pdflatex -halt-on-error -interaction=nonstopmode main_preprint.tex
        if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed" }
        pdflatex -halt-on-error -interaction=nonstopmode main_preprint.tex
        if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed" }
    } elseif ($tectonic) {
        & $tectonic -X compile --outdir . --outfmt pdf --untrusted main_preprint.tex
        if ($LASTEXITCODE -ne 0) { throw "Tectonic build failed" }
    } else {
        throw "Neither pdflatex nor tectonic is available"
    }
    Copy-Item -Force main_preprint.pdf main.pdf
} finally {
    Pop-Location
}

Write-Host "ID-free pre-submission PDFs: $paperDir\main_preprint.pdf and $paperDir\main.pdf"
