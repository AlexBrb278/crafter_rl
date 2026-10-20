# Film checkpoints of each queued run in turn (see scripts\spectator.py); pairs with queue_ppo.ps1.
#   powershell -ExecutionPolicy Bypass -File scripts\queue_spectator.ps1 ppo-s0 ppo-rarity1-s0
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
foreach ($name in $args) {
    & $py -u scripts/spectator.py $name --every 15 --steps 600
}
