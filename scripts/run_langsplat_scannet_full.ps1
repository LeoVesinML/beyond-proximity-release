param(
    [string]$Image = "semanticsplat-langsplat:d70edb8",
    [string]$ModelCache = "C:\GitProjects\baseline-deps\model-cache",
    [string]$SegmentAnything = "C:\GitProjects\baseline-deps\segment-anything-langsplat",
    [string]$SceneId = "scannet_0011_00",
    [int]$Iterations = 3000,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outRelative = "outputs\baselines\langsplat_scannet_full_v1"
$out = Join-Path $repo $outRelative
$scene = Join-Path $out "scene"
$sourceScene = Join-Path $repo "backend\data\scenes\$SceneId"
$benchmark = Join-Path $repo "docs\benchmarks\scannet_bbq_grounding_pilot_v1.json"
$revision = "d70edb86df0fcbda19dc0d9739a3e5140a5e65fc"
$samCheckpoint = Join-Path $ModelCache "sam_vit_b_01ec64.pth"
$clipCheckpoint = Join-Path $ModelCache "open_clip_vit_b16.bin"
$rgbCheckpoint = Join-Path $out "rgb_final_-1\chkpnt$Iterations.pth"
$aeDir = Join-Path $out "ae_checkpoint"
$aeCheckpoint = Join-Path $aeDir "best_ckpt.pth"
$languageModel = Join-Path $out "langsplat_3"
$languageCheckpoint = Join-Path $languageModel "chkpnt$Iterations.pth"

New-Item -ItemType Directory -Force -Path $out,$scene,$aeDir | Out-Null
if (-not (Test-Path $samCheckpoint) -or (Get-Item $samCheckpoint).Length -ne 375042383) {
    throw "Missing complete SAM ViT-B checkpoint: $samCheckpoint"
}
if (-not (Test-Path $clipCheckpoint) -or (Get-Item $clipCheckpoint).Length -ne 598529951) {
    throw "Missing complete OpenCLIP ViT-B-16 checkpoint: $clipCheckpoint"
}
if (-not (Test-Path (Join-Path $SegmentAnything "segment_anything\__init__.py"))) {
    throw "Missing official LangSplat Segment Anything dependency: $SegmentAnything"
}

python -B (Join-Path $repo "scripts\prepare_langsplat_scannet_scene.py") --scene $sourceScene --out $scene
if ($LASTEXITCODE -ne 0) { throw "LangSplat ScanNet conversion failed" }

$repoMount = $repo.Replace("\", "/")
$cacheMount = ([System.IO.Path]::GetFullPath($ModelCache)).Replace("\", "/")
$segmentAnythingMount = ([System.IO.Path]::GetFullPath($SegmentAnything)).Replace("\", "/")
$aeMount = ([System.IO.Path]::GetFullPath($aeDir)).Replace("\", "/")
$containerScene = "/workspace/$($outRelative.Replace('\', '/'))/scene"
$containerOut = "/workspace/$($outRelative.Replace('\', '/'))"

if ($Force -or -not (Test-Path $rgbCheckpoint)) {
    $cmd = "cd /opt/langsplat && sed -i 's/self.include_feature = True/self.include_feature = False/' arguments/__init__.py && python train.py -s $containerScene -m $containerOut/rgb_final --iterations $Iterations --densify_from_iter 100 --densify_until_iter 500 --densification_interval 200 --densify_grad_threshold 0.001 --checkpoint_iterations $Iterations --save_iterations $Iterations --test_iterations 999999 --quiet"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -v "${repoMount}:/workspace" $Image bash -lc $cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "native_rgb_training.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0 -or -not (Test-Path $rgbCheckpoint)) { throw "LangSplat RGB 3DGS training failed" }
} else { Write-Host "Reusing RGB 3DGS checkpoint: $rgbCheckpoint" }

$featureFiles = Join-Path $scene "language_features"
if ($Force -or -not (Test-Path $featureFiles) -or @(Get-ChildItem $featureFiles -Filter "*_f.npy" -ErrorAction SilentlyContinue).Count -lt 4) {
    $cmd = "rm -rf /tmp/segment-anything-langsplat && cp -a /segment-anything-langsplat /tmp/segment-anything-langsplat && python /workspace/scripts/patch_langsplat_sam.py --source /tmp/segment-anything-langsplat/segment_anything/automatic_mask_generator.py && cd /opt/langsplat && python /workspace/scripts/patch_langsplat_preprocess.py --source preprocess.py --out /tmp/preprocess_local.py && PYTHONPATH=/tmp/segment-anything-langsplat SEMANTICSPLAT_OPENCLIP_CHECKPOINT=/models/open_clip_vit_b16.bin SEMANTICSPLAT_SAM_MODEL=vit_b SEMANTICSPLAT_SAM_POINTS_PER_SIDE=32 SEMANTICSPLAT_SAM_POINTS_PER_BATCH=8 SEMANTICSPLAT_SAM_CROP_LAYERS=1 python /tmp/preprocess_local.py --dataset_path $containerScene --resolution 320 --sam_ckpt_path /models/sam_vit_b_01ec64.pth"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -e PYTHONPATH=/segment-anything-langsplat -v "${repoMount}:/workspace" -v "${cacheMount}:/models" -v "${segmentAnythingMount}:/segment-anything-langsplat:ro" $Image bash -lc $cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "native_preprocess.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "LangSplat language-feature preprocessing failed" }
} else { Write-Host "Reusing language features" }

python -B (Join-Path $repo "scripts\resize_langsplat_seg_maps.py") --scene $scene
if ($LASTEXITCODE -ne 0) { throw "LangSplat segmentation-map resizing failed" }

if ($Force -or -not (Test-Path $aeCheckpoint)) {
    $cmd = "cd /opt/langsplat/autoencoder && sed -i '/add_histogram/d' train.py && python train.py --dataset_path $containerScene --num_epochs 100 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name $SceneId"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -v "${repoMount}:/workspace" -v "${aeMount}:/opt/langsplat/autoencoder/ckpt/$SceneId" $Image bash -lc $cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "native_autoencoder_train.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0 -or -not (Test-Path $aeCheckpoint)) { throw "LangSplat autoencoder training failed" }
} else { Write-Host "Reusing autoencoder checkpoint" }

$dim3 = Join-Path $scene "language_features_dim3"
if ($Force -or -not (Test-Path $dim3) -or @(Get-ChildItem $dim3 -Filter "*_f.npy" -ErrorAction SilentlyContinue).Count -lt 4) {
    $cmd = "cd /opt/langsplat/autoencoder && python test.py --dataset_path $containerScene --dataset_name $SceneId"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -v "${repoMount}:/workspace" -v "${aeMount}:/opt/langsplat/autoencoder/ckpt/$SceneId" $Image bash -lc $cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "native_autoencoder_encode.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "LangSplat 3D feature encoding failed" }
}

python -B (Join-Path $repo "scripts\resize_langsplat_seg_maps.py") --scene $scene
if ($LASTEXITCODE -ne 0) { throw "LangSplat encoded segmentation-map resizing failed" }

if ($Force -or -not (Test-Path $languageCheckpoint)) {
    $cmd = "cd /opt/langsplat && python train.py -s $containerScene -m $containerOut/langsplat --start_checkpoint $containerOut/rgb_final_-1/chkpnt$Iterations.pth --feature_level 3 --iterations $Iterations --checkpoint_iterations $Iterations --save_iterations $Iterations --test_iterations 999999 --quiet"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -v "${repoMount}:/workspace" $Image bash -lc $cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "native_language_training.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0 -or -not (Test-Path $languageCheckpoint)) { throw "LangSplat language-field training failed" }
} else { Write-Host "Reusing language-field checkpoint" }

$renderCheckpoint = Join-Path $languageModel "chkpnt30000.pth"
Copy-Item -Force $languageCheckpoint $renderCheckpoint
$renderRoot = Join-Path $languageModel "train"
$renderCandidates = @(Get-ChildItem $renderRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
    @(Get-ChildItem (Join-Path $_.FullName "renders_npy") -Filter "*.npy" -ErrorAction SilentlyContinue).Count -ge 4
})
if ($Force -or $renderCandidates.Count -eq 0) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    docker run --rm --gpus all -v "${repoMount}:/workspace" $Image python render.py -s $containerScene -m $containerOut/langsplat_3 --feature_level 3 --include_feature --skip_test 2>&1 | Tee-Object -FilePath (Join-Path $out "native_render.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "LangSplat native render failed" }
    $renderCandidates = @(Get-ChildItem $renderRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
        @(Get-ChildItem (Join-Path $_.FullName "renders_npy") -Filter "*.npy" -ErrorAction SilentlyContinue).Count -ge 4
    })
}
if ($renderCandidates.Count -ne 1) { throw "Expected one complete LangSplat render directory, found $($renderCandidates.Count)" }
$renderFolderName = $renderCandidates[0].Name

$native = Join-Path $out "native_results.json"
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
docker run --rm --gpus all -v "${repoMount}:/workspace" -v "${cacheMount}:/models" $Image `
    python /workspace/baselines/langsplat/native_batch_query.py `
    --features-dir "$containerOut/langsplat_3/train/$renderFolderName/renders_npy" `
    --ae-checkpoint "$containerOut/ae_checkpoint/best_ckpt.pth" `
    --clip-checkpoint /models/open_clip_vit_b16.bin `
    --benchmark /workspace/docs/benchmarks/scannet_bbq_grounding_pilot_v1.json `
    --manifest "$containerScene/conversion_manifest.json" --scene-root $containerScene `
    --scene-id $SceneId --out "$containerOut/native_results.json" --revision $revision `
    --native-command "ScanNet conversion; 3000-step RGB 3DGS; SAM-B/CLIP-B features; AE; 3000-step LangSplat level 3; batch query" `
    --artifact-record "rgb_final_-1/chkpnt$Iterations.pth" --artifact-record "ae_checkpoint/best_ckpt.pth" `
    --artifact-record "langsplat_3/chkpnt$Iterations.pth" 2>&1 | Tee-Object -FilePath (Join-Path $out "native_query.log")
$code = $LASTEXITCODE; $ErrorActionPreference = $old
if ($code -ne 0 -or -not (Test-Path $native)) { throw "LangSplat native batch query failed" }

python -B (Join-Path $repo "scripts\adapt_langsplat_output.py") --native $native --benchmark $benchmark `
    --scene-id $SceneId --benchmark-scene-id $SceneId --run-id "langsplat_scannet_full_v1" --mode live --out $out
if ($LASTEXITCODE -ne 0) { throw "LangSplat canonical adaptation failed" }
python -B (Join-Path $repo "scripts\evaluate_results.py") --benchmark $benchmark --run-dir $out
if ($LASTEXITCODE -ne 0) { throw "LangSplat canonical evaluation failed" }
Write-Host "LangSplat ScanNet end-to-end pilot complete: $out"
