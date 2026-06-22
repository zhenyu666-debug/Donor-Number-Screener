# push_process_mirror.ps1
# One-shot PowerShell script: clone Donor-Number-Screener, copy the
# lithium-battery-processes/ subdir, commit, and push to main.
#
# Mirrors donor-screener-pbp/push_v21.ps1 (same pattern, new subdir).
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File push_process_mirror.ps1
#   (or)  .\push_process_mirror.ps1
#
# If your remote requires auth, run once before so the Windows credential
# helper caches your PAT / SSH key:
#   git -c credential.helper=manager push https://...   (interactive)
# and re-run this script.

$ErrorActionPreference = 'Stop'

# ---------- inputs ----------
$RemoteUrl = 'https://github.com/zhenyu666-debug/Donor-Number-Screener.git'
$Branch    = 'main'
$SubDir    = 'lithium-battery-processes'
$SrcDir    = 'c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp\lithium-battery-processes'

# committer identity (overridable from env)
$UserName  = $env:GIT_AUTHOR_NAME  ; if (-not $UserName)  { $UserName  = 'zhenyu666-debug' }
$UserEmail = $env:GIT_AUTHOR_EMAIL ; if (-not $UserEmail) { $UserEmail = 'zhenyu666-debug@users.noreply.github.com' }

# ---------- sanity ----------
if (-not (Test-Path $SrcDir)) { throw "Workspace dir not found: $SrcDir" }
$must = @(
    'README.md',
    'index.csv',
    'manifest.json',
    '.gitignore',
    'data\process_steps.yaml',
    'data\parameter_ranges.csv',
    'src\utils_lb.py',
    'src\p36_fetch_process_docs.py',
    'tests\test_fetch_process_docs.py'
)
foreach ($rel in $must) {
    $p = Join-Path $SrcDir $rel
    if (-not (Test-Path $p)) { throw "Missing required file: $p" }
}

# Optional: re-run offline verifier before pushing
Write-Host '[verify] python src/p36_fetch_process_docs.py --offline'
Push-Location $SrcDir
try {
    python src/p36_fetch_process_docs.py --offline | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "offline verify failed ($LASTEXITCODE)" }
    python -m pytest tests/ -q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pytest failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
Write-Host '[verify] OK'

# ---------- clone ----------
$tmp = Join-Path $env:TEMP ('lbp_clone_' + [guid]::NewGuid().ToString('N').Substring(0,8))
Write-Host "Cloning $RemoteUrl to $tmp ..."
git clone $RemoteUrl $tmp
if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }

# ---------- ensure subfolder ----------
$sub = Join-Path $tmp $SubDir
New-Item -ItemType Directory -Force -Path $sub | Out-Null

# ---------- copy whole subdir into the target ----------
# Mirror the layout src/ data/ tests/ README.md etc.
robocopy $SrcDir $sub /MIR /NFL /NDL /NJH /NJS /NC /NS | Out-Null
# robocopy exit 0..7 = success
if ($LASTEXITCODE -gt 7) { throw "robocopy failed ($LASTEXITCODE)" }

# Optional safety: drop any cached __pycache__ from the mirror
Get-ChildItem $sub -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Drop local test logs (they belong in the dev workspace, not upstream)
foreach ($f in @('pytest.log','ruff.log')) {
    $p = Join-Path $sub $f
    if (Test-Path $p) { Remove-Item $p -Force }
}

# ---------- commit + push ----------
Push-Location $tmp
try {
    git config user.name  $UserName
    git config user.email $UserEmail
    git checkout -B $Branch

    git add $SubDir
    git status --short

    $diff = (git diff --cached --name-only)
    if (-not $diff) {
        Write-Host 'No changes to commit; upstream is already up to date.'
    } else {
        $msg = "feat: lithium-battery-processes mirror (bilingual process summaries + raw sources)`n`nAdds the lithium-battery-processes/ tree: 12-source bilingual manifest, 14-step CATL-aligned process map, 17-row parameter range CSV, offline-safe urllib fetcher with sha256, pytest + ruff green."
        git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m $msg
        git push -u origin $Branch
    }
} finally {
    Pop-Location
}

# ---------- cleanup ----------
Remove-Item -Recurse -Force $tmp
Write-Host "`n[OK] Pushed to $RemoteUrl  (branch $Branch, subdir $SubDir/)"
