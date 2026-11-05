param(
    [Parameter(Mandatory = $true)]
    [int]$PipelinePid
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'Z:\.venv\Scripts\python.exe'
$aliasRoot = 'Z:\'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Z: alias or project virtual environment is missing.'
}

while (Get-Process -Id $PipelinePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}

Set-Location -LiteralPath $aliasRoot
for ($attempt = 1; $attempt -le 3; $attempt++) {
    Write-Output "Full pipeline check/resume attempt $attempt of 3"
    & $python -m src.data.kaggle_pipeline run
    if ($LASTEXITCODE -eq 0) {
        break
    }
    if ($attempt -eq 3) {
        throw 'Full pipeline did not finish after three resume attempts; RF was not run.'
    }
    Start-Sleep -Seconds 300
}

Write-Output 'All videos processed; running RF-Prior baseline.'
& $python -m src.data.rf_baseline
if ($LASTEXITCODE -ne 0) {
    throw 'RF-Prior baseline failed; inspect the error log.'
}
Write-Output 'W9 incremental pipeline and RF-Prior baseline complete.'
