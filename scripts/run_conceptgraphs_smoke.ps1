param(
    [string]$Image = "semanticsplat-conceptgraphs:72f5962",
    [string]$ModelCache = "C:\GitProjects\baseline-deps\model-cache",
    [int]$FrameLimit = 1
)

$ErrorActionPreference = "Stop"
if ($FrameLimit -lt 1) { throw "FrameLimit must be at least 1" }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$out = Join-Path $repo "outputs\baselines\conceptgraphs_smoke_v1"
$nativeData = Join-Path $out "native_data"
$sceneId = "ConferenceHall-capture-pilot"
$revision = "72f5962822b5e8678a446f367a06df1a977d2a4d"
$detectionsSuffix = "semanticsplat_detections_v1"
$mappingSuffix = "semanticsplat_mapping_v1"
$mapRelative = "native_data/$sceneId/exps/$mappingSuffix/pcd_$mappingSuffix.pkl.gz"

New-Item -ItemType Directory -Force -Path $out,$ModelCache | Out-Null

python -B (Join-Path $repo "scripts\prepare_conceptgraphs_scene.py") `
    --scene (Join-Path $repo "backend\data\scenes\$sceneId") `
    --out $nativeData `
    --limit $FrameLimit
if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs scene conversion failed" }

$repoMount = $repo.Replace("\", "/")
$cacheMount = ([System.IO.Path]::GetFullPath($ModelCache)).Replace("\", "/")
$dataRoot = "/workspace/outputs/baselines/conceptgraphs_smoke_v1/native_data"
$config = "$dataRoot/captured_scene.yaml"
$classes = "/opt/conceptgraphs/conceptgraph/scannet200_classes.txt"

$detectionArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", "${repoMount}:/workspace",
    "-v", "${cacheMount}:/models",
    $Image,
    "bash", "-lc",
    "ln -sf /models/yolov8l-world.pt yolov8l-world.pt; ln -sf /models/mobile_sam.pt mobile_sam.pt; python scripts/streamlined_detections.py `"`$@`"", "--",
    "dataset_root=$dataRoot", "dataset_config=$config", "scene_id=$sceneId",
    "start=0", "end=$FrameLimit", "stride=1", "desired_height=320", "desired_width=640",
    "classes_file=$classes", "device=cuda", "save_video=false",
    "exp_suffix=$detectionsSuffix"
)
$ErrorActionPreference = "Continue"
& docker @detectionArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "native_detection.log")
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($dockerExit -ne 0) { throw "ConceptGraphs native detection failed" }

$detectionsDir = Join-Path $nativeData "$sceneId\exps\$detectionsSuffix\detections"
python -B (Join-Path $repo "scripts\augment_conceptgraphs_detections.py") `
    --detections $detectionsDir
if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs detection augmentation failed" }

$mappingArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", "${repoMount}:/workspace",
    "-v", "${cacheMount}:/models",
    $Image,
    "python", "slam/streamlined_mapping.py",
    "dataset_root=$dataRoot", "dataset_config=$config", "scene_id=$sceneId",
    "start=0", "end=$FrameLimit", "stride=1", "image_height=320", "image_width=640",
    "detections_exp_suffix=$detectionsSuffix", "exp_suffix=$mappingSuffix",
    "save_video=false", "save_objects_all_frames=false", "vis_render=false",
    "use_rerun=false", "dbscan_remove_noise=false", "run_denoise_final_frame=false",
    "run_filter_final_frame=false", "run_merge_final_frame=false", "obj_min_detections=1"
)
$ErrorActionPreference = "Continue"
& docker @mappingArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "native_mapping.log")
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($dockerExit -ne 0) {
    $mapArtifact = Join-Path $out ($mapRelative.Replace("/", "\"))
    if (-not (Test-Path -LiteralPath $mapArtifact)) {
        throw "ConceptGraphs native mapping failed"
    }
    Write-Warning "ConceptGraphs mapping exited non-zero after writing $mapRelative; continuing with native query."
}

$detectionRelative = "native_data/$sceneId/exps/$detectionsSuffix/detections/v001.pkl.gz"
$nativeCommand = "streamlined_detections.py end=$FrameLimit; streamlined_mapping.py end=$FrameLimit; native_query.py q051"
$queryArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", "${repoMount}:/workspace",
    "-v", "${cacheMount}:/models",
    $Image,
    "python", "/workspace/baselines/conceptgraphs/native_query.py",
    "--map", "/workspace/outputs/baselines/conceptgraphs_smoke_v1/$mapRelative",
    "--out", "/workspace/outputs/baselines/conceptgraphs_smoke_v1/native_results.json",
    "--scene-id", $sceneId, "--query-id", "q051", "--query", "Find a chair",
    "--selected-view-id", "v001", "--revision", $revision,
    "--native-command", $nativeCommand,
    "--artifact-record", $mapRelative,
    "--artifact-record", $detectionRelative
)
$ErrorActionPreference = "Continue"
& docker @queryArgs 2>&1 | Tee-Object -FilePath (Join-Path $out "native_query.log")
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($dockerExit -ne 0) { throw "ConceptGraphs native query failed" }

python -B (Join-Path $repo "scripts\adapt_conceptgraphs_output.py") `
    --native (Join-Path $out "native_results.json") `
    --benchmark (Join-Path $repo "docs\benchmarks\benchmark_queries_v1.json") `
    --scene-id $sceneId `
    --benchmark-scene-id "ConferenceHall" `
    --mode live `
    --out $out
if ($LASTEXITCODE -ne 0) { throw "ConceptGraphs canonical adaptation failed" }

Write-Host "ConceptGraphs smoke complete: $out"
