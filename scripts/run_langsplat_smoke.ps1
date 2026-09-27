param(
    [string]$Image = "semanticsplat-langsplat:d70edb8",
    [string]$AssetRoot = "C:\GitProjects\baseline-deps\LangSplat-assets",
    [string]$ModelCache = "C:\GitProjects\baseline-deps\model-cache"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$assetRoot = (Resolve-Path $AssetRoot).Path
$out = Join-Path $repo "outputs\baselines\langsplat_smoke_v1"
$revision = "d70edb86df0fcbda19dc0d9739a3e5140a5e65fc"
$renderedFeatureCandidates = @(
    (Join-Path $assetRoot "output\sofa_1\train\ours_30000\renders_npy\00000.npy"),
    (Join-Path $assetRoot "output\sofa_1\train\ours_None\renders_npy\00000.npy")
)
$renderedFeature = $renderedFeatureCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $renderedFeature) {
    $renderedFeature = $renderedFeatureCandidates[0]
}
$aeCheckpoint = Join-Path $assetRoot "ckpt\sofa\best_ckpt.pth"
$modelCheckpoint = Join-Path $assetRoot "output\sofa_1\chkpnt30000.pth"

foreach ($required in @($aeCheckpoint, $modelCheckpoint, (Join-Path $assetRoot "data\sofa"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing LangSplat asset: $required" }
}
New-Item -ItemType Directory -Force -Path $out,$ModelCache | Out-Null

$repoMount = $repo.Replace("\", "/")
$assetMount = $assetRoot.Replace("\", "/")
$cacheMount = ([System.IO.Path]::GetFullPath($ModelCache)).Replace("\", "/")
$renderCommand = "python render.py -s /assets/data/sofa -m /assets/output/sofa_1 --feature_level 1 --include_feature --skip_test"

if (-not (Test-Path -LiteralPath $renderedFeature)) {
    $renderArgs = @(
        "run", "--rm", "--gpus", "all",
        "-v", "${assetMount}:/assets",
        "-v", "${cacheMount}:/models",
        $Image,
        "python", "render.py", "-s", "/assets/data/sofa", "-m", "/assets/output/sofa_1",
        "--feature_level", "1", "--include_feature", "--skip_test"
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @renderArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "native_render.log")
    $renderExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($renderExitCode -ne 0) { throw "LangSplat native render failed" }
    $renderedFeature = $renderedFeatureCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not (Test-Path -LiteralPath $renderedFeature)) {
    throw "LangSplat render completed without an expected feature map: $($renderedFeatureCandidates -join ', ')"
}

$featureRecord = $renderedFeature.Replace("\", "/")
$aeRecord = $aeCheckpoint.Replace("\", "/")
$modelRecord = $modelCheckpoint.Replace("\", "/")
$queryArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", "${repoMount}:/workspace",
    "-v", "${assetMount}:/assets",
    "-v", "${cacheMount}:/models",
    $Image,
    "python", "/workspace/baselines/langsplat/native_query.py",
    "--feature-map", ($featureRecord -replace [regex]::Escape($assetMount), "/assets"),
    "--ae-checkpoint", "/assets/ckpt/sofa/best_ckpt.pth",
    "--out", "/workspace/outputs/baselines/langsplat_smoke_v1/native_results.json",
    "--scene-id", "langsplat-sofa-official", "--query-id", "ls001",
    "--query", "Find the sofa", "--selected-view-id", "sofa_train_00000",
    "--revision", $revision, "--native-command", "$renderCommand; native_query.py ls001",
    "--artifact-record", $featureRecord,
    "--artifact-record", $aeRecord,
    "--artifact-record", $modelRecord
)
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker @queryArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "native_query.log")
$queryExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($queryExitCode -ne 0) { throw "LangSplat native query failed" }

python -B (Join-Path $repo "scripts\adapt_langsplat_output.py") `
    --native (Join-Path $out "native_results.json") `
    --benchmark (Join-Path $repo "docs\benchmarks\baseline_smoke_queries_v1.json") `
    --scene-id "langsplat-sofa-official" `
    --benchmark-scene-id "langsplat-sofa" `
    --mode live `
    --out $out
if ($LASTEXITCODE -ne 0) { throw "LangSplat canonical adaptation failed" }

Write-Host "LangSplat smoke complete: $out"
