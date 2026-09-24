param(
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $project
$python = Join-Path $project '..\cenv\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = 'python' }

$products = @(
    'CCDepth.tif',
    'CCDepth_QA.tif',
    'CCDepth_WSE.tif',
    'CCDepth_WSE_gradient_solved.tif',
    'CCDepth_depth_signed.tif',
    'CCDepth_depth_solved.tif',
    'CCDepth_gradient_directions.tif',
    'CCDepth_status.tif'
)

$runs = @(
    [pscustomobject]@{
        Config = 'configs\guangxi_label_updated_lean.json'
        Output = 'runs\CCDepth\Guangxi\label_relaxed_lean_v2'
        Name = 'Guangxi label'
    },
    [pscustomobject]@{
        Config = 'configs\guangxi_seg_updated_lean.json'
        Output = 'runs\CCDepth\Guangxi\seg_relaxed_lean_v1'
        Name = 'Guangxi seg'
    }
)

function Assert-OnlyProducts([string]$RunDir) {
    $items = @(Get-ChildItem -LiteralPath $RunDir -Force)
    $directories = @($items | Where-Object { $_.PSIsContainer })
    $files = @($items | Where-Object { -not $_.PSIsContainer } | ForEach-Object Name | Sort-Object)
    $expected = @($products | Sort-Object)
    if ($directories.Count -ne 0 -or $files.Count -ne $expected.Count -or
        (Compare-Object -ReferenceObject $expected -DifferenceObject $files)) {
        throw "Run output is not exactly the expected eight TIFF products: $RunDir"
    }
}

foreach ($run in $runs) {
    Write-Host "=== $($run.Name): input check ==="
    & $python -m ccdepth_local check --config $run.Config
    if ($LASTEXITCODE -ne 0) { throw "CCDepth input check failed: $($run.Config)" }

    $runDir = [System.IO.Path]::GetFullPath((Join-Path $project $run.Output))
    $hasWorkState = Test-Path -LiteralPath (Join-Path $runDir '.work\state.sqlite') -PathType Leaf
    $hasProducts = $false
    if (Test-Path -LiteralPath $runDir -PathType Container) {
        $existingNames = @(Get-ChildItem -LiteralPath $runDir -File | ForEach-Object Name | Sort-Object)
        $expectedNames = @($products | Sort-Object)
        $hasProducts = ($existingNames.Count -eq $expectedNames.Count -and
            -not (Compare-Object -ReferenceObject $expectedNames -DifferenceObject $existingNames))
        $hasAny = @(Get-ChildItem -LiteralPath $runDir -Force).Count -gt 0
        if ($hasAny -and -not $hasWorkState -and -not $hasProducts) {
            throw "Output directory contains unrecognized/incomplete data; preserving it: $runDir"
        }
    }

    Write-Host "=== $($run.Name): run ==="
    if ($Resume -and ($hasWorkState -or $hasProducts)) {
        & $python -m ccdepth_local run --config $run.Config --resume
    } elseif ($hasWorkState -or $hasProducts) {
        throw "Run state already exists. Re-run this script with -Resume to continue: $runDir"
    } else {
        & $python -m ccdepth_local run --config $run.Config
    }
    if ($LASTEXITCODE -ne 0) { throw "CCDepth run failed: $($run.Config)" }

    Write-Host "=== $($run.Name): audit and metrics ==="
    $auditLines = & $python -m ccdepth_local audit --run-dir $runDir
    if ($LASTEXITCODE -ne 0) { throw "CCDepth audit failed: $runDir" }
    $auditText = ($auditLines -join [Environment]::NewLine).Trim()
    if ($auditText -notmatch '^COMPLETE: \d+ components; solved [\d.]+%; audit passed\.') {
        throw "CCDepth audit did not report a completed, audited run: $auditText"
    }
    Assert-OnlyProducts $runDir
    $metricsCode = "import json,sys,rasterio; d=rasterio.open(sys.argv[1]); r=json.loads(d.tags(ns='CCDEPTH')['run']); s=r['solve']; p=r['prepare']; n=s['pixels_by_status']; print(json.dumps({'components':s['components'],'support_pixels':p['support_pixels'],'solved_fraction':(n['3']+n['4'])/sum(n.values()),'prepare_seconds':p['seconds'],'solve_seconds':s['seconds'],'peak_rss':s['peak_rss']}))"
    $tifPath = Join-Path $runDir 'CCDepth.tif'
    $metricsLine = & $python -c $metricsCode $tifPath
    if ($LASTEXITCODE -ne 0) { throw "Could not read run metrics from TIFF metadata: $tifPath" }
    $metrics = ($metricsLine -join [Environment]::NewLine) | ConvertFrom-Json
    $recordedSeconds = [double]$metrics.prepare_seconds + [double]$metrics.solve_seconds
    $peakGiB = [double]$metrics.peak_rss / 1GB
    Write-Host ("{0}: components={1}; support={2}; solved_fraction={3:P4}; recorded_prepare+solve={4:N1}s; peak_RSS={5:N2}GiB" -f `
        $run.Name, $metrics.components, $metrics.support_pixels, $metrics.solved_fraction, $recordedSeconds, $peakGiB)
}
