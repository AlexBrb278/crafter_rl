# Run one configuration for several seeds, one after another (finished runs are skipped,
# interrupted ones resume). Run names are <Prefix>-s<seed>.
#   powershell -ExecutionPolicy Bypass -File scripts\queue.ps1 -Config ppo -Bonus 2.0 -Prefix ppo-rarity2 -Seeds 0,1,2
#   powershell -ExecutionPolicy Bypass -File scripts\queue.ps1 -Config ppo_ad -Bonus 1.0 -Prefix ad-rarity1 -Seeds 1
param(
    [string]$Config = "ppo",
    [string]$Bonus = "0",
    [string]$Prefix = "ppo",
    [string]$Seeds = "0"   # comma-separated; a string because `powershell -File` passes "0,1,2" as one argument
)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
foreach ($s in ($Seeds -split ",")) {
    $s = $s.Trim()
    $name = "$Prefix-s$s"
    if (Test-Path (Join-Path $Root "runs\$name\final.json")) { continue }
    & $py -u methods/05_achievement_distillation/train.py --config $Config --seed $s --micro_batch 256 `
        --aux_micro_batch 128 --run_name $name --rarity_bonus $Bonus --resume
}
