param(
    [switch]$Once,
    [ValidateRange(1, 3600)]
    [int]$IntervalSeconds = 10,
    [int]$ProcessId = 0
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifestPath = Join-Path $projectRoot 'splits\source_manifest.json'
$statusDir = Join-Path $projectRoot 'data\status'
$stageDir = Join-Path $projectRoot 'data\staging'
$workDir = Join-Path $projectRoot 'data\work'
$pidFile = Join-Path $projectRoot 'data\pipeline.pid'

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw 'Frozen source manifest is missing.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

do {
    $done = @{}
    $holds = @{}
    Get-ChildItem -LiteralPath $statusDir -Filter '*.json' -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            $status = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
            if ($status.source_retained_for_review) {
                $holds[$_.BaseName] = $true
            } else {
                $done[$_.BaseName] = $true
            }
        }
    $completed = @($manifest.files | Where-Object { $done.ContainsKey($_.key) }).Count
    $total = @($manifest.files).Count

    $activePid = $ProcessId
    if (-not $activePid -and (Test-Path -LiteralPath $pidFile)) {
        [void][int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$activePid)
    }
    $process = if ($activePid) { Get-Process -Id $activePid -ErrorAction SilentlyContinue }
    $state = if ($process) { "running PID $activePid" } else { 'not running' }
    $pending = $manifest.files | Where-Object {
        -not $done.ContainsKey($_.key) -and -not $holds.ContainsKey($_.key)
    } | Select-Object -First 1
    $firstHold = $manifest.files | Where-Object { $holds.ContainsKey($_.key) } | Select-Object -First 1
    $next = if ($process -and $pending) { $pending } elseif ($firstHold) { $firstHold } else { $pending }

    $detail = 'queue complete'
    if ($next) {
        $partial = Join-Path $stageDir ($next.key + '.zip.part')
        $source = Join-Path $stageDir ($next.key + [IO.Path]::GetExtension($next.name).ToLowerInvariant())
        $windows = @(Get-ChildItem -LiteralPath (Join-Path $workDir $next.key) -Filter 'w*.npz' -File -ErrorAction SilentlyContinue).Count
        if ($holds.ContainsKey($next.key)) {
            $detail = "quality hold $($next.key): source retained for review"
        } elseif (Test-Path -LiteralPath $partial) {
            $size = (Get-Item -LiteralPath $partial).Length / 1MB
            $detail = "downloading $($next.key): $([math]::Round($size, 1)) MiB received"
        } elseif (Test-Path -LiteralPath $source) {
            $detail = "processing $($next.key): $windows windows"
        } else {
            $detail = "next $($next.key): waiting"
        }
    }
    '{0} | {1}/{2} videos ({3:N1}%) | {4} hold(s) | {5} | {6}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),
        $completed, $total, (100 * $completed / $total), $holds.Count, $state, $detail

    if ($Once -or $completed -eq $total -or -not $process) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
