param(
    [string]$Image = "semanticsplat-conceptgraphs:72f5962",
    [string]$ModelCache = "C:\GitProjects\baseline-deps\model-cache",
    [string]$Benchmark = "docs\benchmarks\benchmark_queries_v2.json",
    [string]$Out = "outputs\baselines\conceptgraphs_full_v1",
    [int]$FrameLimit = 0,
    [switch]$ForceQuery
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $repo $Out
$nativeData = Join-Path $outDir "native_data"
$benchmarkPath = Join-Path $repo $Benchmark
$revision = "72f5962822b5e8678a446f367a06df1a977d2a4d"
$detectionsSuffix = "semanticsplat_full_detections_v1"
$mappingSuffix = "semanticsplat_full_mapping_v1"
$scenePairs = @(
    @{ Capture = "ConferenceHall-capture-pilot"; Benchmark = "ConferenceHall" },
    @{ Capture = "Museume-capture"; Benchmark = "Museume" },
    @{ Capture = "Theater-capture"; Benchmark = "Theater" },
    @{ Capture = "outdoor-street-capture"; Benchmark = "outdoor-street" },
    @{ Capture = "outdoor-drone-capture"; Benchmark = "outdoor-drone" }
)

New-Item -ItemType Directory -Force -Path $outDir,$nativeData,$ModelCache | Out-Null

$repoMount = $repo.Replace("\", "/")
$cacheMount = ([System.IO.Path]::GetFullPath($ModelCache)).Replace("\", "/")
$outMount = $Out.Replace("\", "/")
$dataRoot = "/workspace/$outMount/native_data"
$benchmarkContainer = "/workspace/$($Benchmark.Replace('\', '/'))"
$classes = "/opt/conceptgraphs/conceptgraph/scannet200_classes.txt"

foreach ($pair in $scenePairs) {
    $sceneId = $pair.Capture
    $benchmarkSceneId = $pair.Benchmark
    $scenePath = Join-Path $repo "backend\data\scenes\$sceneId"
    $transforms = Get-Content -Raw -LiteralPath (Join-Path $scenePath "transforms.json") | ConvertFrom-Json
    $sceneFrameCount = @($transforms.frames).Count
    $limit = if ($FrameLimit -gt 0) { [Math]::Min($FrameLimit, $sceneFrameCount) } else { $sceneFrameCount }
    if ($limit -lt 1) { throw "No frames available for $sceneId" }

    Write-Host "=== ConceptGraphs full scene: $sceneId ($limit frame(s)) ==="

    python -B (Join-Path $repo "scripts\prepare_conceptgraphs_scene.py") `
        --scene $scenePath `
        --out $nativeData `
        --limit $limit
    if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs scene conversion failed for $sceneId" }

    $sceneOut = Join-Path $outDir $sceneId
    New-Item -ItemType Directory -Force -Path $sceneOut | Out-Null
    $config = "$dataRoot/captured_scene.yaml"
    $sceneData = "$dataRoot/$sceneId"

    $detectionsDir = Join-Path $nativeData "$sceneId\exps\$detectionsSuffix\detections"
    $existingDetectionCount = 0
    if (Test-Path -LiteralPath $detectionsDir) {
        $existingDetectionCount = @(Get-ChildItem -LiteralPath $detectionsDir -Filter "*.pkl.gz").Count
    }
    if ($existingDetectionCount -ge $limit) {
        Write-Host "Reusing $existingDetectionCount existing detection file(s) for $sceneId"
    } else {
        $detectionArgs = @(
            "run", "--rm", "--gpus", "all",
            "-e", "HF_HOME=/models/huggingface",
            "-v", "${repoMount}:/workspace",
            "-v", "${cacheMount}:/models",
            $Image,
            "bash", "-lc",
            "ln -sf /models/yolov8l-world.pt yolov8l-world.pt; ln -sf /models/mobile_sam.pt mobile_sam.pt; cp /workspace/baselines/conceptgraphs/streamlined_detections_empty_safe.py scripts/streamlined_detections_empty_safe.py; python scripts/streamlined_detections_empty_safe.py `"`$@`"", "--",
            "dataset_root=$dataRoot", "dataset_config=$config", "scene_id=$sceneId",
            "start=0", "end=$limit", "stride=1", "desired_height=320", "desired_width=640",
            "classes_file=$classes", "device=cuda", "save_video=false",
            "exp_suffix=$detectionsSuffix"
        )
        $ErrorActionPreference = "Continue"
        & docker @detectionArgs 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_detection.log")
        $dockerExit = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($dockerExit -ne 0) { throw "ConceptGraphs native detection failed for $sceneId" }
    }

    python -B (Join-Path $repo "scripts\augment_conceptgraphs_detections.py") `
        --detections $detectionsDir
    if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs detection augmentation failed for $sceneId" }

    $mapRelative = "native_data/$sceneId/exps/$mappingSuffix/pcd_$mappingSuffix.pkl.gz"
    $mapHost = Join-Path $outDir ($mapRelative.Replace("/", "\"))
    if (Test-Path -LiteralPath $mapHost) {
        Write-Host "Reusing existing map for ${sceneId}: $mapRelative"
    } else {
        $mappingArgs = @(
            "run", "--rm", "--gpus", "all",
            "-e", "HF_HOME=/models/huggingface",
            "-v", "${repoMount}:/workspace",
            "-v", "${cacheMount}:/models",
            $Image,
            "python", "slam/streamlined_mapping.py",
            "dataset_root=$dataRoot", "dataset_config=$config", "scene_id=$sceneId",
            "start=0", "end=$limit", "stride=1", "image_height=320", "image_width=640",
            "detections_exp_suffix=$detectionsSuffix", "exp_suffix=$mappingSuffix",
            "spatial_sim_type=iou",
            "save_video=false", "save_objects_all_frames=false", "vis_render=false",
            "use_rerun=false", "dbscan_remove_noise=false", "run_denoise_final_frame=false",
            "run_filter_final_frame=false", "run_merge_final_frame=false", "obj_min_detections=1"
        )
        $ErrorActionPreference = "Continue"
        & docker @mappingArgs 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_mapping.log")
        $dockerExit = $LASTEXITCODE
        $ErrorActionPreference = "Stop"

        if ($dockerExit -ne 0) {
            if (-not (Test-Path -LiteralPath $mapHost)) {
                throw "ConceptGraphs native mapping failed for $sceneId"
            }
            Write-Warning "ConceptGraphs mapping exited non-zero after writing $mapRelative for $sceneId; continuing with native query."
        }
    }

    $nativeOutRelative = "$Out/native_results_$sceneId.json".Replace("\", "/")
    $nativeOutHost = Join-Path $outDir "native_results_$sceneId.json"
    $manifestRelative = "native_data/$sceneId/conversion_manifest.json"
    $nativeCommand = "streamlined_detections.py end=$limit; streamlined_mapping.py end=$limit; native_batch_query.py $benchmarkSceneId"
    if ((Test-Path -LiteralPath $nativeOutHost) -and -not $ForceQuery) {
        Write-Host "Reusing existing native query output for ${sceneId}: $nativeOutRelative"
    } else {
        $queryArgs = @(
            "run", "--rm", "--gpus", "all",
            "-e", "HF_HOME=/models/huggingface",
            "-v", "${repoMount}:/workspace",
            "-v", "${cacheMount}:/models",
            $Image,
            "python", "/workspace/baselines/conceptgraphs/native_batch_query.py",
            "--map", "/workspace/$outMount/$mapRelative",
            "--benchmark", $benchmarkContainer,
            "--benchmark-scene-id", $benchmarkSceneId,
            "--manifest", "/workspace/$outMount/$manifestRelative",
            "--out", "/workspace/$nativeOutRelative",
            "--scene-id", $sceneId,
            "--revision", $revision,
            "--native-command", $nativeCommand,
            "--artifact-record", $mapRelative,
            "--artifact-record", $manifestRelative
        )
        $ErrorActionPreference = "Continue"
        & docker @queryArgs 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_query.log")
        $dockerExit = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($dockerExit -ne 0) { throw "ConceptGraphs native batch query failed for $sceneId" }
    }
}

python -B (Join-Path $repo "scripts\adapt_conceptgraphs_full_run.py") `
    --native-dir $out `
    --benchmark $benchmarkPath `
    --run-id "conceptgraphs_full_v1" `
    --mode live `
    --out $outDir
if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs full canonical adaptation failed" }

python -B (Join-Path $repo "scripts\evaluate_results.py") `
    --benchmark $benchmarkPath `
    --run-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs full evaluation failed" }

Write-Host "ConceptGraphs full run complete: $outDir"
