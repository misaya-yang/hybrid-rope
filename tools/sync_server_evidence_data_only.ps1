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
    [string]$LocalTargetRel = "archives/server_artifacts_2026-02-21",
    [switch]$CleanTarget
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

$localTarget = Join-Path $localRepo $LocalTargetRel
if ($CleanTarget -and (Test-Path $localTarget)) {
    Remove-Item -Recurse -Force $localTarget
}
New-Item -ItemType Directory -Force -Path $localTarget | Out-Null

$remoteTar = "$RemoteRepoRoot/server_evidence_data_only_sync.tgz"
$localTar = Join-Path $tmpDir "server_evidence_data_only_sync.tgz"

# Only sync evidence artifacts; never sync model/checkpoint weights.
$syncPaths = @(
    "results",
    "logs",
    "sigmoid_rope_experiments/data",
    "sigmoid_rope_experiments/results",
    "sigmoid_rope_experiments/run_all.log",
    "sigmoid_rope_experiments/run_passkey_debug.log",
    "sigmoid_rope_experiments/run_passkey_debug_final.log",
    "sigmoid_rope_experiments/run_passkey_debug_monkey.log",
    "sigmoid_rope_experiments/run_phase2.log",
    "sigmoid_rope_experiments/run_phase3.log",
    "sigmoid_rope_experiments/run_phase4.log"
) -join " "

$excludeArgs = @(
    "--exclude='*.safetensors'",
    "--exclude='*.pt'",
    "--exclude='*.bin'",
    "--exclude='*.pth'",
    "--exclude='optimizer*'",
    "--exclude='pytorch_model*'",
    "--exclude='training_args.bin'",
    "--exclude='**/checkpoint-*'",
    "--exclude='**/checkpoints/*'",
    "--exclude='**/_weights_quarantine/*'",
    "--exclude='**/final_lora/tokenizer.json'",
    "--exclude='**/final_model/tokenizer.json'",
    "--exclude='**/model/tokenizer.json'"
) -join " "

$remoteCmd = "cd $RemoteRepoRoot && tar -czf $remoteTar --warning=no-file-changed $excludeArgs $syncPaths && ls -lh $remoteTar"

Write-Host "[1/3] Building remote evidence tar..."
& $plink -batch -ssh -P $RemotePort "$RemoteUser@$RemoteHost" -pw $RemotePassword $remoteCmd

Write-Host "[2/3] Downloading tar..."
if (Test-Path $localTar) {
    Remove-Item -Force $localTar
}
& $pscp -batch -P $RemotePort -pw $RemotePassword "$RemoteUser@$RemoteHost`:$remoteTar" $tmpDir

Write-Host "[3/3] Extracting into $localTarget ..."
tar -xzf $localTar -C $localTarget

Write-Host "Done. Evidence sync complete:"
Write-Host "  $localTarget"
