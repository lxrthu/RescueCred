param(
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$Source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = [System.IO.Path]::GetFullPath((Join-Path $Source $OutputDirectory))
if (-not $DistRoot.StartsWith($Source, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must resolve inside the project root"
}
$Stage = Join-Path $DistRoot "RescueCredit"
$Zip = Join-Path $DistRoot "RescueCredit-cloud-ready.zip"

if (Test-Path -LiteralPath $Stage) {
    $resolved = [System.IO.Path]::GetFullPath($Stage)
    if (-not $resolved.StartsWith($DistRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove stage outside dist"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

$excludedDirectories = @(
    ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".git", "dist",
    ".aris", "smoke", "dry_run", "dry_run_final", "runs", "checkpoints", "events"
)
$arguments = @($Source, $Stage, "/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/XD") + $excludedDirectories + @("/XF", "*.pyc", "*.pyo")
& robocopy @arguments | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
Write-Output $Stage
Write-Output $Zip

