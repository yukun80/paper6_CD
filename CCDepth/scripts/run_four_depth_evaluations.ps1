param(
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$python = Join-Path $project '..\cenv\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }

$configs = @(
    'configs\guangxi_label_relaxed_lean.json',
    'configs\zhengzhou_label_relaxed_lean.json',
    'configs\zhengzhou_seg_relaxed_lean.json',
    'configs\zhuozhou_label_relaxed_lean.json'
)

foreach ($config in $configs) {
    & $python -m ccdepth_local check --config $config
    if ($LASTEXITCODE -ne 0) { throw "CCDepth check failed: $config" }
    if ($Resume) {
        & $python -m ccdepth_local run --config $config --resume
    } else {
        & $python -m ccdepth_local run --config $config
    }
    if ($LASTEXITCODE -ne 0) { throw "CCDepth failed: $config" }
}
