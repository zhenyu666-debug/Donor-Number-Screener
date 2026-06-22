# push_p37_candidate_api.ps1
# Start (or stop) the p37 candidate SSE scoring API on port 8765.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File push_p37_candidate_api.ps1            # start
#   powershell -ExecutionPolicy Bypass -File push_p37_candidate_api.ps1 -Action stop
#   powershell -ExecutionPolicy Bypass -File push_p37_candidate_api.ps1 -Action status
#
# Side effects:
#   - writes PID to  data/.candidate_api.pid
#   - writes log to  data/.candidate_api.log
#   - probes http://127.0.0.1:<port>/health  and prints the status code

[CmdletBinding()]
param(
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start",
    [int]$Port = 8765,
    [string]$Host = "0.0.0.0"
)

$ErrorActionPreference = 'Stop'

$SrcDir   = 'c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp'
$DataDir  = Join-Path $SrcDir 'data'
$PidFile  = Join-Path $DataDir '.candidate_api.pid'
$LogFile  = Join-Path $DataDir '.candidate_api.log'

if (-not (Test-Path $DataDir)) { New-Item -ItemType Directory -Force -Path $DataDir | Out-Null }

function Get-RunningPid {
    if (Test-Path $PidFile) {
        $pidVal = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($pidVal -and (Get-Process -Id $pidVal -ErrorAction SilentlyContinue)) {
            return [int]$pidVal
        }
    }
    # fallback: netstat lookup
    $conn = netstat -ano | Select-String ":$Port\s" | Select-Object -First 1
    if ($conn) {
        $tokens = ($conn -split '\s+') | Where-Object { $_ }
        return [int]$tokens[-1]
    }
    return $null
}

switch ($Action) {
    'status' {
        $p = Get-RunningPid
        if ($p) {
            Write-Host "[p37_api] running pid=$p port=$Port"
            try {
                $resp = Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
                Write-Host "[p37_api] /health -> $($resp.StatusCode)"
                Write-Host $resp.Content
            } catch {
                Write-Host "[p37_api] /health unreachable: $_"
            }
        } else {
            Write-Host "[p37_api] not running"
        }
        return
    }
    'stop' {
        $p = Get-RunningPid
        if ($p) {
            Write-Host "[p37_api] stopping pid=$p"
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            Remove-Item $PidFile -ErrorAction SilentlyContinue
        } else {
            Write-Host "[p37_api] no running process"
        }
        return
    }
    'start' {
        $existing = Get-RunningPid
        if ($existing) {
            Write-Host "[p37_api] already running pid=$existing port=$Port"
            return
        }
        Push-Location $SrcDir
        try {
            $args = @("p37_candidate_api:app","--host",$Host,"--port","$Port")
            Write-Host "[p37_api] starting: uvicorn $($args -join ' ')"
            $proc = Start-Process -FilePath "uvicorn" `
                                  -ArgumentList $args `
                                  -WorkingDirectory $SrcDir `
                                  -RedirectStandardOutput $LogFile `
                                  -RedirectStandardError  "$LogFile.err" `
                                  -PassThru -NoNewWindow
            Set-Content -Path $PidFile -Value $proc.Id
            Start-Sleep -Seconds 3
            $p = Get-RunningPid
            if ($p) {
                Write-Host "[p37_api] started pid=$p port=$Port"
                try {
                    $resp = Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
                    Write-Host "[p37_api] /health -> $($resp.StatusCode)"
                    Write-Host $resp.Content
                } catch {
                    Write-Host "[p37_api] /health probe failed: $_"
                    Write-Host "[p37_api] tail of $LogFile.err :"
                    Get-Content "$LogFile.err" -Tail 20 -ErrorAction SilentlyContinue
                }
            } else {
                Write-Host "[p37_api] failed to start; see $LogFile.err"
            }
        } finally {
            Pop-Location
        }
    }
}
