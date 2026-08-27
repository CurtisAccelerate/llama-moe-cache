[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$ServerPath,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(512, 262144)][int]$ContextSize = 32768,
    [ValidateRange(1, 256)][int]$Threads = 12,
    [ValidateRange(1, 1024)][int]$CpuMoeLayers = 35,
    [ValidateRange(128, 131072)][int]$CacheMiB = 3600,
    [ValidateRange(1, 8192)][int]$BatchSize = 4096,
    [ValidateRange(1, 8192)][int]$UBatchSize = 4096,
    [string]$StateDirectory,
    [switch]$CacheOff,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$releaseRoot = Split-Path -Parent $PSScriptRoot
$model = Get-Item -LiteralPath $ModelPath
if ($model.PSIsContainer -or $model.Extension -ne '.gguf') { throw 'ModelPath must be a GGUF file.' }
if ($UBatchSize -gt $BatchSize) { throw 'UBatchSize must not exceed BatchSize.' }
if (!$ServerPath) {
    $ServerPath = Join-Path $releaseRoot 'build/bin/Release/llama-server.exe'
    if (!(Test-Path -LiteralPath $ServerPath)) { $ServerPath = Join-Path $releaseRoot 'build/bin/llama-server' }
}
if (!$StateDirectory) {
    $StateDirectory = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'llama-moe-cache'
}
$shards = @($model)
if ($model.Name -match '^(.*)-00001-of-(\d+)\.gguf$') {
    $prefix = $Matches[1]
    $count = [int]$Matches[2]
    $shards = @(for ($i = 1; $i -le $count; $i++) {
        Get-Item -LiteralPath (Join-Path $model.DirectoryName ('{0}-{1:D5}-of-{2:D5}.gguf' -f $prefix, $i, $count))
    })
}
$identity = (($shards | ForEach-Object { '{0}|{1}|{2}' -f $_.FullName, $_.Length, $_.LastWriteTimeUtc.Ticks }) -join ';') + "|cpu=$CpuMoeLayers|cache=$CacheMiB|v3"
$digest = [System.Security.Cryptography.SHA256]::Create()
try { $fingerprint = ([BitConverter]::ToString($digest.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity)))).Replace('-', '').ToLowerInvariant() } finally { $digest.Dispose() }
$hotset = Join-Path $StateDirectory ($fingerprint + '.bin')
$settings = @{
    GGML_CUDA_MOE_CACHE = $(if ($CacheOff) { '0' } else { '1' })
    GGML_CUDA_MOE_CACHE_BUDGET_MB = "$CacheMiB"
    GGML_CUDA_MOE_CACHE_RESERVE_MB = '2048'
    GGML_CUDA_MOE_CACHE_MAX_BLOCK = "$($CpuMoeLayers - 1)"
    GGML_CUDA_MOE_CACHE_MAX_BATCH = '16'
    GGML_CUDA_MOE_CACHE_HOTSET = '1'
    GGML_CUDA_MOE_CACHE_HOTSET_PATH = $hotset
    GGML_CUDA_MOE_CACHE_HOTSET_SAVE_SEC = '15'
    GGML_CUDA_MOE_CACHE_PREFETCH = '1'
    GGML_CUDA_MOE_CACHE_HOT_ONLY = '1'
    GGML_CUDA_MOE_CACHE_STATS = '64'
}
$serverArgs = @('-m', $model.FullName, '-ngl', 'all', '-ncmoe', "$CpuMoeLayers", '--fit', 'off', '-c', "$ContextSize", '-b', "$BatchSize", '-ub', "$UBatchSize", '-t', "$Threads", '-tb', "$Threads", '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '-np', '1', '--spec-type', 'none', '--host', '127.0.0.1', '--port', "$Port", '--jinja', '--cache-prompt')
if ($DryRun) {
    [pscustomobject]@{ Executable = $ServerPath; Arguments = $serverArgs; Environment = $settings } | ConvertTo-Json -Depth 4
    return
}
if (!(Test-Path -LiteralPath $ServerPath -PathType Leaf)) { throw 'Build llama-server first or supply -ServerPath.' }
if ([System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners().Port -contains $Port) {
    throw "TCP port $Port is already listening. No existing server was stopped."
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuInfo = & nvidia-smi --query-gpu=name,power.limit --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'Cannot verify GPU power limit.' }
    foreach ($line in $gpuInfo) {
        $fields = $line -split ','
        if ($fields[0] -match 'RTX 5090') {
            $powerLimit = 0.0
            if (![double]::TryParse($fields[1].Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$powerLimit) -or $powerLimit -gt 450) {
                throw 'Set and verify the RTX 5090 power limit at 450 W or lower before using this preset. No power setting was changed.'
            }
        }
    }
}
if (!$CacheOff) { New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null }
$saved = @{}
Get-ChildItem Env: | Where-Object { $_.Name -like 'GGML_CUDA_MOE_CACHE*' -or $_.Name -like 'LLAMA_SPEC_*' } | ForEach-Object { $saved[$_.Name] = $_.Value }
try {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    foreach ($name in $settings.Keys) { [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process') }
    Write-Host "Single-slot server: http://127.0.0.1:$Port ; cache=$(!$CacheOff) ; CPU MoE layers=$CpuMoeLayers ; speculation=none"
    & $ServerPath @serverArgs
    if ($LASTEXITCODE -ne 0) { throw "llama-server exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $settings.Keys) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
