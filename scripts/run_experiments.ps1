# =============================================================================
# RxRx1 Experiment Runner
#
# This script runs multiple RxRx1 experiments sequentially.
# Each YAML file represents one experiment configuration.
#
# If one experiment fails, the failure is recorded and the script continues
# running the remaining experiments.
# =============================================================================

$ProjectRoot = "C:\rxrx1"
$Python = "$ProjectRoot\.venv\Scripts\python.exe"
$TrainScript = "$ProjectRoot\scripts\train.py"

# -----------------------------------------------------------------------------
# Add the YAML config files you want to run here.
# Experiments will run sequentially from top to bottom.
# -----------------------------------------------------------------------------
$configs = @(
"configs/baseline_config_test.yaml"
)

Set-Location $ProjectRoot

$failedConfigs = @()

foreach ($config in $configs) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Running experiment: $config"
    Write-Host "========================================"

    & $Python $TrainScript --config $config

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Experiment failed: $config"
        $failedConfigs += $config
        continue
    }

    Write-Host "Finished: $config"
}

Write-Host ""
Write-Host "========================================"
Write-Host "All experiments processed."
Write-Host "========================================"

if ($failedConfigs.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed experiments:"

    foreach ($config in $failedConfigs) {
        Write-Host "  - $config"
    }
} else {
    Write-Host "All experiments completed successfully."
}

