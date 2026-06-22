# push_p37.ps1
# One-shot PowerShell script: clone Donor-Number-Screener, copy the p37 files
# into donor-screener-pbp/, commit, and push to main.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File push_p37.ps1
#
# If your remote requires auth, run once before so the Windows credential
# helper caches your PAT / SSH key:
#   git -c credential.helper=manager push https://...   (interactive)
# and re-run this script.

$ErrorActionPreference = 'Stop'

# ---------- inputs ----------
$RemoteUrl = 'https://github.com/zhenyu666-debug/Donor-Number-Screener.git'
$Branch    = 'main'
$SubDir    = 'donor-screener-pbp'
$SrcDir    = 'c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp'

# committer identity (overridable from env)
$UserName  = $env:GIT_AUTHOR_NAME  ; if (-not $UserName)  { $UserName  = 'zhenyu666-debug' }
$UserEmail = $env:GIT_AUTHOR_EMAIL ; if (-not $UserEmail) { $UserEmail = 'zhenyu666-debug@users.noreply.github.com' }

# ---------- sanity ----------
if (-not (Test-Path $SrcDir)) { throw "Workspace dir not found: $SrcDir" }
$must = @(
    'src\p37_enrich_candidates.py',
    'src\p37_candidate_score.py',
    'src\p37_candidate_api.py',
    'tests\test_p37_candidate_score.py',
    'push_p37_candidate_api.ps1'
)
foreach ($rel in $must) {
    $p = Join-Path $SrcDir $rel
    if (-not (Test-Path $p)) { throw "Missing required file: $p" }
}

# ---------- clone ----------
$tmp = Join-Path $env:TEMP ('dns_clone_' + [guid]::NewGuid().ToString('N').Substring(0,8))
Write-Host "Cloning $RemoteUrl to $tmp ..."
git clone $RemoteUrl $tmp
if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }

# ---------- ensure subfolder ----------
$sub = Join-Path $tmp $SubDir
New-Item -ItemType Directory -Force -Path $sub | Out-Null

# ---------- copy whole workspace tree into the subfolder ----------
# Mirror the layout src/ data/ tests/ etc. (MIR deletes files not in src)
robocopy $SrcDir $sub /MIR /NFL /NDL /NJH /NJS /NC /NS | Out-Null
# robocopy exit 0..7 = success
if ($LASTEXITCODE -gt 7) { throw "robocopy failed ($LASTEXITCODE)" }

# Optional safety: drop any cached __pycache__ from the mirror
Get-ChildItem $sub -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---------- commit + push ----------
Push-Location $tmp
try {
    git config user.name  $UserName
    git config user.email $UserEmail
    git checkout -B $Branch

    git add $SubDir
    git status --short

    # only commit if there's something new
    $diff = (git diff --cached --name-only)
    if (-not $diff) {
        Write-Host 'No changes to commit; upstream is already up to date.'
    } else {
        $msg = "feat(p37): candidate SSE scoring + API enrichment + FastAPI service`n`nAdds p37 pipeline over the existing 14-SSE library:`n- p37_enrich_candidates.py: urllib fetcher querying Crossref / Materials Project / PubChem to backfill missing fields`n- p37_candidate_score.py: compute_dn_candidate_v2() (5-component weighted DN) + class-based default fill + ranked CSV/JSON output`n- p37_candidate_api.py: FastAPI on port 8765 with /health, /candidates, /candidate_score, /candidate_enrich, /candidate_rank`n- 7 new pytest tests; 69/69 total green; ruff clean."
        git commit -m $msg
        git push -u origin $Branch
    }
} finally {
    Pop-Location
}

# ---------- cleanup ----------
Remove-Item -Recurse -Force $tmp
Write-Host "`n[OK] Pushed to $RemoteUrl  (branch $Branch, subdir $SubDir/)"
