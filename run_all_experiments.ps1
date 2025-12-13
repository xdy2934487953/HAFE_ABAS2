# Set encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Running HAFE-ABSA Comparative Experiments" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Create results directory
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Experiment configurations
$experiments = @(
    @{Dataset="semeval2014"; Model="baseline"; Name="SemEval-2014 Baseline"},
    @{Dataset="semeval2014"; Model="hafe"; Name="SemEval-2014 HAFE"},
    @{Dataset="semeval2016"; Model="baseline"; Name="SemEval-2016 Baseline"},
    @{Dataset="semeval2016"; Model="hafe"; Name="SemEval-2016 HAFE"}
)

$total = $experiments.Count
$current = 0

foreach ($exp in $experiments) {
    $current++
    Write-Host ""
    Write-Host "[$current/$total] $($exp.Name)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    $logFile = "results\$($exp.Dataset)_$($exp.Model).log"
    
    $startTime = Get-Date
    
    # Run experiment
    python train.py --dataset $exp.Dataset --model $exp.Model --epochs 50 2>&1 | Tee-Object -FilePath $logFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Experiment failed!" -ForegroundColor Red
        pause
        exit 1
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host "Completed! Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All experiments completed! Results saved in results\ directory" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to view comparative results..." -ForegroundColor Yellow
pause

# Show comparison results
python compare_results.py

pause