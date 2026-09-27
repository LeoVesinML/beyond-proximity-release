param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9]+$')]
    [string]$PaperId
)

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
$idPath = Join-Path $paperDir "review_id.tex"
$idSource = "\def\TwinWorldPaperID{$PaperId}`n"
[System.IO.File]::WriteAllText($idPath, $idSource, [System.Text.UTF8Encoding]::new($false))

Push-Location $paperDir
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
    } elseif ($tectonic) {
        & $tectonic -X compile --outdir . --outfmt pdf --untrusted main.tex
        if ($LASTEXITCODE -ne 0) { throw "Tectonic build failed" }
    } else {
        throw "Neither pdflatex nor tectonic is available"
    }
} finally {
    Pop-Location
}

Write-Host "TwinWorld anonymous review PDF: $paperDir\main.pdf (Paper ID $PaperId)"
