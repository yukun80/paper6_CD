param(
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $project
$python = Join-Path $project '..\cenv\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = 'python' }

$config = 'configs\zhuozhou_seg_relaxed_lean.json'
$runDir = [System.IO.Path]::GetFullPath((Join-Path $project 'runs\CCDepth\Zhuozhou\seg_relaxed_lean_v1'))
$products = @(
    'CCDepth.tif', 'CCDepth_QA.tif', 'CCDepth_WSE.tif',
    'CCDepth_WSE_gradient_solved.tif', 'CCDepth_depth_signed.tif',
    'CCDepth_depth_solved.tif', 'CCDepth_gradient_directions.tif', 'CCDepth_status.tif'
)

Write-Host '=== Zhuozhou seg: input check ==='
& $python -m ccdepth_local check --config $config
if ($LASTEXITCODE -ne 0) { throw "CCDepth input check failed: $config" }

$hasWorkState = Test-Path -LiteralPath (Join-Path $runDir '.work\state.sqlite') -PathType Leaf
$hasProducts = $false
if (Test-Path -LiteralPath $runDir -PathType Container) {
    $existing = @(Get-ChildItem -LiteralPath $runDir -File | ForEach-Object Name | Sort-Object)
    $expected = @($products | Sort-Object)
    $hasProducts = ($existing.Count -eq $expected.Count -and
        -not (Compare-Object -ReferenceObject $expected -DifferenceObject $existing))
    if (@(Get-ChildItem -LiteralPath $runDir -Force).Count -gt 0 -and -not $hasWorkState -and -not $hasProducts) {
        throw "Output directory contains unrecognized/incomplete data; preserving it: $runDir"
    }
}

Write-Host '=== Zhuozhou seg: run ==='
if ($Resume -and ($hasWorkState -or $hasProducts)) {
    & $python -m ccdepth_local run --config $config --resume
} elseif ($hasWorkState -or $hasProducts) {
    throw "Run state already exists. Re-run this script with -Resume to continue: $runDir"
} else {
    & $python -m ccdepth_local run --config $config
}
if ($LASTEXITCODE -ne 0) { throw "CCDepth run failed: $config" }

Write-Host '=== Zhuozhou seg: final audit ==='
$auditLines = & $python -m ccdepth_local audit --run-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "CCDepth audit failed: $runDir" }
$auditText = ($auditLines -join [Environment]::NewLine).Trim()
if ($auditText -notmatch '^COMPLETE: \d+ components; solved [\d.]+%; audit passed\.') {
    throw "CCDepth audit did not report a completed run: $auditText"
}

$items = @(Get-ChildItem -LiteralPath $runDir -Force)
$files = @($items | Where-Object { -not $_.PSIsContainer } | ForEach-Object Name | Sort-Object)
$expectedProducts = @($products | Sort-Object)
if (@($items | Where-Object { $_.PSIsContainer }).Count -ne 0 -or
    $files.Count -ne $expectedProducts.Count -or
    (Compare-Object -ReferenceObject $expectedProducts -DifferenceObject $files)) {
    throw "Run output is not exactly the expected eight TIFF products: $runDir"
}

$metricsCode = "import json,sys,rasterio; d=rasterio.open(sys.argv[1]); r=json.loads(d.tags(ns='CCDEPTH')['run']); s=r['solve']; p=r['prepare']; n=s['pixels_by_status']; print(json.dumps({'components':s['components'],'support_pixels':p['support_pixels'],'solved_fraction':(n['3']+n['4'])/sum(n.values()),'prepare_seconds':p['seconds'],'solve_seconds':s['seconds'],'peak_rss':s['peak_rss'],'whole_scene':r['whole_scene']}))"
$metricsLine = & $python -c $metricsCode (Join-Path $runDir 'CCDepth.tif')
if ($LASTEXITCODE -ne 0) { throw "Could not read run metrics from TIFF metadata: $runDir" }
$metrics = ($metricsLine -join [Environment]::NewLine) | ConvertFrom-Json
if (-not $metrics.whole_scene) { throw "Expected a whole-scene result: $runDir" }
$recordedSeconds = [double]$metrics.prepare_seconds + [double]$metrics.solve_seconds
$peakGiB = [double]$metrics.peak_rss / 1GB
Write-Host ("Zhuozhou seg: components={0}; support={1}; solved_fraction={2:P4}; recorded_prepare+solve={3:N1}s; peak_RSS={4:N2}GiB" -f `
    $metrics.components, $metrics.support_pixels, $metrics.solved_fraction, $recordedSeconds, $peakGiB)
