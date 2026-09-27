param(
    [ValidateSet("scannet", "replica")]
    [string]$Dataset = "scannet",
    [string]$Image = "semanticsplat-conceptgraphs:72f5962",
    [string]$ModelCache = "C:\GitProjects\baseline-deps\model-cache",
    [string]$ReplicaRoot = "data\replica_nice_slam\Replica",
    [int]$ReplicaFramesPerScene = 5,
    [int]$FrameLimit = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$revision = "72f5962822b5e8678a446f367a06df1a977d2a4d"
$detectionsSuffix = "semanticsplat_public_detections_v1"
$mappingSuffix = "semanticsplat_public_mapping_v1"

if ($Dataset -eq "scannet") {
    $sceneIds = @(
        "scannet_0011_00", "scannet_0030_00", "scannet_0046_00", "scannet_0086_00",
        "scannet_0222_00", "scannet_0378_00", "scannet_0389_00", "scannet_0435_00"
    )
    $benchmark = "docs\benchmarks\scannet_bbq_grounding_pilot_v1.json"
} else {
    $sceneIds = @(
        "replica_room0", "replica_room1", "replica_room2", "replica_office0",
        "replica_office1", "replica_office2", "replica_office3", "replica_office4"
    )
    $benchmark = "docs\benchmarks\replica_scannet_pilot_queries.json"
    $replicaSourceIds = @{
        "replica_room0" = "room0"; "replica_room1" = "room1"; "replica_room2" = "room2"
        "replica_office0" = "office0"; "replica_office1" = "office1"; "replica_office2" = "office2"
        "replica_office3" = "office3"; "replica_office4" = "office4"
    }
}

$out = "outputs\baselines\conceptgraphs_${Dataset}_full_v1"
$outDir = Join-Path $repo $out
$nativeData = Join-Path $outDir "native_data"
$benchmarkPath = Join-Path $repo $benchmark
$openaiClipCache = Join-Path $ModelCache "openai_clip"
New-Item -ItemType Directory -Force -Path $outDir,$nativeData,$ModelCache,$openaiClipCache | Out-Null

$weights = @{
    "yolov8l-world.pt" = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt"
    "mobile_sam.pt" = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
}
foreach ($name in $weights.Keys) {
    $target = Join-Path $ModelCache $name
    if (-not (Test-Path -LiteralPath $target)) {
        Write-Host "Downloading $name"
        Invoke-WebRequest -UseBasicParsing -Uri $weights[$name] -OutFile $target
    }
}
$clipCheckpoint = Join-Path $ModelCache "open_clip_pytorch_model.bin"
if (-not (Test-Path -LiteralPath $clipCheckpoint) -or (Get-Item $clipCheckpoint).Length -ne 3944692325) {
    throw "Missing complete OpenCLIP checkpoint: $clipCheckpoint (expected 3944692325 bytes)"
}
$openaiClipCheckpoint = Join-Path $openaiClipCache "ViT-B-32.pt"
if (-not (Test-Path -LiteralPath $openaiClipCheckpoint) -or (Get-Item $openaiClipCheckpoint).Length -ne 353976522) {
    throw "Missing complete OpenAI CLIP checkpoint: $openaiClipCheckpoint (expected 353976522 bytes)"
}

$repoMount = $repo.Replace("\", "/")
$cacheMount = ([System.IO.Path]::GetFullPath($ModelCache)).Replace("\", "/")
$openaiClipMount = ([System.IO.Path]::GetFullPath($openaiClipCache)).Replace("\", "/")
$outMount = $out.Replace("\", "/")
$dataRoot = "/workspace/$outMount/native_data"
$benchmarkContainer = "/workspace/$($benchmark.Replace('\', '/'))"
$classes = "/opt/conceptgraphs/conceptgraph/scannet200_classes.txt"

foreach ($sceneId in $sceneIds) {
    if ($Dataset -eq "replica") {
        $replicaRootPath = [System.IO.Path]::GetFullPath((Join-Path $repo $ReplicaRoot))
        $sourceScene = Join-Path $replicaRootPath $replicaSourceIds[$sceneId]
        if (-not (Test-Path -LiteralPath (Join-Path $sourceScene "traj.txt"))) {
            throw "Missing official Replica RGB-D trajectory: $sourceScene"
        }
        $limit = if ($FrameLimit -gt 0) { $FrameLimit } else { $ReplicaFramesPerScene }
        python -B (Join-Path $repo "scripts\prepare_conceptgraphs_replica_scene.py") `
            --source $sourceScene --scene-id $sceneId --out $nativeData --frame-count $limit
        if ($LASTEXITCODE -ne 0) { throw "Replica scene conversion failed for $sceneId" }
        $config = "/opt/conceptgraphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
        $imageHeight = 340
        $imageWidth = 600
    } else {
        $scenePath = Join-Path $repo "backend\data\scenes\$sceneId"
        $transformsPath = Join-Path $scenePath "transforms.json"
        if (-not (Test-Path -LiteralPath $transformsPath)) { throw "Missing scene: $sceneId" }
        $transforms = Get-Content -Raw -LiteralPath $transformsPath | ConvertFrom-Json
        $sceneFrameCount = @($transforms.frames).Count
        $limit = if ($FrameLimit -gt 0) { [Math]::Min($FrameLimit, $sceneFrameCount) } else { $sceneFrameCount }
        if ($limit -lt 1) { throw "No frames available for $sceneId" }
        python -B (Join-Path $repo "scripts\prepare_conceptgraphs_scene.py") --scene $scenePath --out $nativeData --limit $limit
        if ($LASTEXITCODE -ne 0) { throw "Scene conversion failed for $sceneId" }
        $config = "$dataRoot/captured_scene.yaml"
        $imageHeight = 320
        $imageWidth = 640
    }

    Write-Host "=== ConceptGraphs ${Dataset}: $sceneId ($limit frame(s)) ==="

    $sceneOut = Join-Path $outDir $sceneId
    New-Item -ItemType Directory -Force -Path $sceneOut | Out-Null
    $detectionsDir = Join-Path $nativeData "$sceneId\exps\$detectionsSuffix\detections"
    $detectionCount = if (Test-Path $detectionsDir) { @(Get-ChildItem $detectionsDir -Filter "*.pkl.gz").Count } else { 0 }
    if ($Force -or $detectionCount -lt $limit) {
        $cmd = "ln -sf /models/yolov8l-world.pt yolov8l-world.pt; ln -sf /models/mobile_sam.pt mobile_sam.pt; cp /workspace/baselines/conceptgraphs/streamlined_detections_empty_safe.py scripts/streamlined_detections_empty_safe.py; python scripts/streamlined_detections_empty_safe.py `"`$@`""
        $args = @(
            "run", "--rm", "--gpus", "all", "-e", "HF_HOME=/models/huggingface",
            "-e", "SEMANTICSPLAT_OPENCLIP_CHECKPOINT=/models/open_clip_pytorch_model.bin",
            "-v", "${repoMount}:/workspace", "-v", "${cacheMount}:/models",
            "-v", "${openaiClipMount}:/root/.cache/clip", $Image,
            "bash", "-lc", $cmd, "--", "dataset_root=$dataRoot", "dataset_config=$config",
            "scene_id=$sceneId", "start=0", "end=$limit", "stride=1", "desired_height=$imageHeight",
            "desired_width=$imageWidth", "classes_file=$classes", "device=cuda", "save_video=false",
            "exp_suffix=$detectionsSuffix"
        )
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & docker @args 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_detection.log")
        $exitCode = $LASTEXITCODE; $ErrorActionPreference = $old
        if ($exitCode -ne 0) { throw "Native detection failed for $sceneId" }
    } else { Write-Host "Reusing $detectionCount detections" }

    python -B (Join-Path $repo "scripts\augment_conceptgraphs_detections.py") --detections $detectionsDir
    if ($LASTEXITCODE -ne 0) { throw "Detection augmentation failed for $sceneId" }

    $mapRelative = "native_data/$sceneId/exps/$mappingSuffix/pcd_$mappingSuffix.pkl.gz"
    $mapHost = Join-Path $outDir ($mapRelative.Replace("/", "\"))
    if ($Force -or -not (Test-Path -LiteralPath $mapHost)) {
        $cmd = "cp slam/streamlined_mapping.py slam/streamlined_mapping_absolute.py; sed -i '/dtype=torch.float,/a\            relative_pose=False,' slam/streamlined_mapping_absolute.py; python slam/streamlined_mapping_absolute.py `"`$@`""
        $args = @(
            "run", "--rm", "--gpus", "all", "-e", "HF_HOME=/models/huggingface",
            "-e", "SEMANTICSPLAT_OPENCLIP_CHECKPOINT=/models/open_clip_pytorch_model.bin",
            "-v", "${repoMount}:/workspace", "-v", "${cacheMount}:/models", $Image,
            "bash", "-lc", $cmd, "--", "dataset_root=$dataRoot", "dataset_config=$config",
            "scene_id=$sceneId", "start=0", "end=$limit", "stride=1", "image_height=$imageHeight",
            "image_width=$imageWidth", "detections_exp_suffix=$detectionsSuffix", "exp_suffix=$mappingSuffix",
            "spatial_sim_type=iou", "save_video=false", "save_objects_all_frames=false",
            "vis_render=false", "use_rerun=false", "dbscan_remove_noise=false",
            "run_denoise_final_frame=false", "run_filter_final_frame=false",
            "run_merge_final_frame=false", "obj_min_detections=1"
        )
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & docker @args 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_mapping.log")
        $exitCode = $LASTEXITCODE; $ErrorActionPreference = $old
        if ($exitCode -ne 0 -and -not (Test-Path -LiteralPath $mapHost)) { throw "Native mapping failed for $sceneId" }
    } else { Write-Host "Reusing native map" }

    $nativeRelative = "$out/native_results_$sceneId.json".Replace("\", "/")
    $nativeHost = Join-Path $outDir "native_results_$sceneId.json"
    $manifestRelative = "native_data/$sceneId/conversion_manifest.json"
    if ($Force -or -not (Test-Path -LiteralPath $nativeHost)) {
        $args = @(
            "run", "--rm", "--gpus", "all", "-e", "HF_HOME=/models/huggingface",
            "-e", "SEMANTICSPLAT_OPENCLIP_CHECKPOINT=/models/open_clip_pytorch_model.bin",
            "-v", "${repoMount}:/workspace", "-v", "${cacheMount}:/models", $Image,
            "python", "/workspace/baselines/conceptgraphs/native_batch_query.py",
            "--map", "/workspace/$outMount/$mapRelative", "--benchmark", $benchmarkContainer,
            "--benchmark-scene-id", $sceneId, "--manifest", "/workspace/$outMount/$manifestRelative",
            "--out", "/workspace/$nativeRelative", "--scene-id", $sceneId, "--revision", $revision,
            "--native-command", "native detection, absolute-pose mapping, CLIP batch query",
            "--image-width", "$imageWidth", "--image-height", "$imageHeight",
            "--artifact-record", $mapRelative, "--artifact-record", $manifestRelative
        )
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & docker @args 2>&1 | Tee-Object -FilePath (Join-Path $sceneOut "native_query.log")
        $exitCode = $LASTEXITCODE; $ErrorActionPreference = $old
        if ($exitCode -ne 0) { throw "Native query failed for $sceneId" }
    } else { Write-Host "Reusing native query result" }
}

python -B (Join-Path $repo "scripts\adapt_conceptgraphs_full_run.py") --native-dir $outDir `
    --benchmark $benchmarkPath --run-id "conceptgraphs_${Dataset}_full_v1" --mode live `
    --dataset-scope $Dataset --out $outDir
if ($LASTEXITCODE -ne 0) { throw "Canonical adaptation failed" }
python -B (Join-Path $repo "scripts\evaluate_results.py") --benchmark $benchmarkPath --run-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
Write-Host "ConceptGraphs public run complete: $outDir"
