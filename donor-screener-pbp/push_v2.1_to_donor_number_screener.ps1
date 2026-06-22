# push_v2.1_to_donor_number_screener.ps1
# Run this in PowerShell from C:\Users\Hasee\.qclaw\workspace\get_jobs

$ErrorActionPreference = 'Stop'

# 1) clone the public Donor-Number-Screener repo into a tmp dir
$tmp = Join-Path $env:TEMP 'dns_clone_' ([guid]::NewGuid().ToString('N').Substring(0,8))
Write-Host "Cloning Donor-Number-Screener to $tmp ..."
git clone https://github.com/zhenyu666-debug/Donor-Number-Screener.git $tmp
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

# 2) make a subfolder donor-screener-pbp inside the clone
$sub = Join-Path $tmp 'donor-screener-pbp'
New-Item -ItemType Directory -Force -Path $sub | Out-Null

# 3) copy the v2.1 files from the workspace
$src = 'c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp'

Copy-Item (Join-Path $src 'src\34_fetch_sse_datasets.py')     (Join-Path $sub '34_fetch_sse_datasets.py')      -Force
Copy-Item (Join-Path $src 'src\35_pareto_best_sse.py')        (Join-Path $sub '35_pareto_best_sse.py')         -Force
Copy-Item (Join-Path $src 'data\paper_sse_extra.yaml')        (Join-Path $sub 'paper_sse_extra.yaml')          -Force
Copy-Item (Join-Path $src 'tests\test_fetch_sse.py')          (Join-Path $sub 'test_fetch_sse.py')             -Force
Copy-Item (Join-Path $src 'tests\test_pareto.py')             (Join-Path $sub 'test_pareto.py')                -Force
Copy-Item (Join-Path $src 'README.md')                        (Join-Path $sub 'README.md')                     -Force
Copy-Item (Join-Path $src 'MANUAL_PUSH.md')                   (Join-Path $sub 'MANUAL_PUSH.md')                -Force

# also copy already-produced CSVs/JSONs if they exist
foreach ($f in @('sse_datasets_combined.csv','sse_datasets_meta.json',
                'pareto_front.csv','pareto_summary.json')) {
    $from = Join-Path $src ('data\' + $f)
    $to   = Join-Path $sub $f
    if (Test-Path $from) {
        Copy-Item $from $to -Force
    } else {
        # write a placeholder so the file is still committed
        "{}" | Set-Content -Path $to
    }
}

# 4) commit + push
Push-Location $tmp
try {
    git add donor-screener-pbp
    git status --short
    git -c user.name="zhenyu666-debug" -c user.email="zhenyu666-debug@users.noreply.github.com" `
        commit -m "feat: v2.1 SSE datasets fetch (OBELiX+COD+CEMP+paper) + Pareto front"
    git push
} finally {
    Pop-Location
}

# 5) cleanup
Remove-Item -Recurse -Force $tmp

Write-Host "Done."
