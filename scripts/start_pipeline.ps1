param(
    [ValidateSet('pilot', 'run')]
    [string]$Mode = 'run',
    [int]$LimitFiles = 3,
    [switch]$ContinuePastQualityHolds
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$aliasRoot = 'Z:\'
if (Test-Path -LiteralPath $aliasRoot) {
    $probeName = '.alias_probe_' + [guid]::NewGuid().ToString('N')
    $probe = Join-Path $projectRoot ('data\' + $probeName)
    New-Item -ItemType File -Path $probe | Out-Null
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $aliasRoot ('data\' + $probeName)))) {
            throw 'Z: is already in use by another location. Choose another free drive letter in this script.'
        }
    } finally {
        Remove-Item -LiteralPath $probe -Force
    }
} else {
    & subst Z: $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create ASCII drive alias for the project.'
    }
}

$python = Join-Path $aliasRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project virtual environment is missing.'
}

$pidFile = Join-Path $projectRoot 'data\pipeline.pid'
if ($Mode -eq 'run' -and (Test-Path -LiteralPath $pidFile)) {
    $oldPid = 0
    if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$oldPid)) {
        $existing = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($existing -and $existing.ProcessName -like 'python*') {
            throw "Full pipeline already running as PID $oldPid."
        }
    }
}

$logDir = Join-Path $projectRoot 'experiments\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "pipeline_${Mode}_${stamp}.out.log"
$stderr = Join-Path $logDir "pipeline_${Mode}_${stamp}.err.log"
$arguments = @('-m', 'src.data.kaggle_pipeline', $Mode)
if ($Mode -eq 'pilot') {
    $arguments += @('--limit-files', $LimitFiles.ToString())
} elseif ($ContinuePastQualityHolds) {
    $arguments += '--continue-past-quality-holds'
}

$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $aliasRoot -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

$watcherProcessId = $null
if ($Mode -eq 'run') {
    Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ASCII
    if (-not $ContinuePastQualityHolds) {
        $watcher = Join-Path $PSScriptRoot 'finish_w9.ps1'
        $watchOut = Join-Path $logDir "finish_w9_${stamp}.out.log"
        $watchErr = Join-Path $logDir "finish_w9_${stamp}.err.log"
        $watchProcess = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $watcher,
                            '-PipelinePid', $process.Id.ToString()) `
            -WorkingDirectory $aliasRoot -RedirectStandardOutput $watchOut `
            -RedirectStandardError $watchErr -WindowStyle Hidden -PassThru
        $watcherProcessId = $watchProcess.Id
    }
}

[pscustomobject]@{
    PID = $process.Id
    Mode = $Mode
    StandardOutput = $stdout
    StandardError = $stderr
    WatcherPID = $watcherProcessId
}
