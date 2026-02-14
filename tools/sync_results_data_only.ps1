param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$RemotePort,

    [Parameter(Mandatory = $true)]
    [string]$RemoteUser,

    [Parameter(Mandatory = $true)]
    [string]$RemotePassword,

    [string]$RemoteRepoRoot = "/root/autodl-tmp/dfrope/hybrid-rope",
    [string]$LocalRepoRoot = ".",
    [string[]]$ResultDirs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$plink = "C:\Users\Admin\.ssh\plink.exe"
$pscp = "C:\Users\Admin\.ssh\pscp.exe"

if (-not (Test-Path $plink)) {
    throw "plink not found: $plink"
}
if (-not (Test-Path $pscp)) {
    throw "pscp not found: $pscp"
}

$localRepo = Resolve-Path $LocalRepoRoot
$tmpDir = Join-Path $localRepo ".tmp_sync"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$remoteTar = "$RemoteRepoRoot/results_data_only_sync.tgz"
$localTar = Join-Path $tmpDir "results_data_only_sync.tgz"

if ($ResultDirs.Count -eq 0) {
    $dirsRemote = "results"
} else {
    $dirsRemote = ($ResultDirs | ForEach-Object { "results/$($_)" }) -join " "
}

$excludeArgs = @(
    "--exclude='*.safetensors'",
    "--exclude='*.pt'",
    "--exclude='*.bin'",
    "--exclude='*.pth'",
    "--exclude='optimizer*'",
    "--exclude='pytorch_model*'"
) -join " "

$remoteCmd = "cd $RemoteRepoRoot && tar -czf $remoteTar --warning=no-file-changed $excludeArgs $dirsRemote && ls -lh $remoteTar"

Write-Host "[1/3] Building remote data-only tar..."
& $plink -batch -ssh -P $RemotePort "$RemoteUser@$RemoteHost" -pw $RemotePassword $remoteCmd

Write-Host "[2/3] Downloading tar..."
& $pscp -batch -P $RemotePort -pw $RemotePassword "$RemoteUser@$RemoteHost`:$remoteTar" $tmpDir

Write-Host "[3/3] Extracting into local repo..."
tar -xzf $localTar -C $localRepo

Write-Host "Done. Data-only sync complete."

