param(
    [Parameter(Mandatory = $true)] [string] $ServerPath,
    [Parameter(Mandatory = $true)] [string] $ModelPath,
    [Parameter(Mandatory = $true)] [string] $DraftPath,
    [int] $Port = 8091,
    [int] $Threads = 20,
    [int] $CpuMoeLayers = 35,
    [int] $ContextSize = 32768
)

$ErrorActionPreference = 'Stop'
foreach ($path in @($ServerPath, $ModelPath, $DraftPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "File not found: $path"
    }
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied; no process was stopped."
}

$env:LLAMA_GLM53_EXACT_KDA_ROWS = '1'
$env:LLAMA_MOE_MROW_VNNI = '1'
$env:LLAMA_MOE_MROW_VNNI_DYNAMIC = '1'
$env:LLAMA_MOE_MROW_VNNI_TILES_PER_THREAD = '4'
$env:LLAMA_MOE_MROW_VNNI_MIN_M = '2'
$env:LLAMA_GLM53_RS_ROLLBACK_CAP = '1'

$serverArgs = @(
    '-m', (Resolve-Path -LiteralPath $ModelPath).Path,
    '-ngl', 'all', '-ncmoe', "$CpuMoeLayers", '--fit', 'off',
    '-c', "$ContextSize", '-b', '512', '-ub', '512',
    '-t', "$Threads", '-tb', "$Threads", '-np', '1', '-fa', 'on',
    '-ctk', 'f16', '-ctv', 'f16',
    '--spec-type', 'draft-dflash', '-md', (Resolve-Path -LiteralPath $DraftPath).Path,
    '--spec-draft-device', 'CUDA0', '--spec-draft-ngl', 'all',
    '--spec-draft-n-max', '1', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
    '--cache-ram', '8192', '--ctx-checkpoints', '32',
    '--host', '127.0.0.1', '--port', "$Port",
    '--alias', 'glm-5.3-flash-iq1m-dflash2', '--slots', '--metrics', '--jinja'
)

& (Resolve-Path -LiteralPath $ServerPath).Path @serverArgs
