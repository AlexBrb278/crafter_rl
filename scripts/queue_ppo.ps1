# Run the PPO baseline and PPO + rarity bonus (paper PPO config) for each given seed, one after another.
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ppo.ps1            # seed 0
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ppo.ps1 1 2        # seeds 1 and 2
# Finished runs (final.json present) are skipped; interrupted ones resume from their checkpoint.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
$seeds = if ($args.Count) { $args } else { @("0") }
foreach ($s in $seeds) {
    foreach ($r in @(@{ name = "ppo-s$s"; bonus = "0" }, @{ name = "ppo-rarity1-s$s"; bonus = "1.0" })) {
        if (Test-Path (Join-Path $Root "runs\$($r.name)\final.json")) { continue }
        & $py -u methods/05_achievement_distillation/train.py --config ppo --seed $s --micro_batch 256 `
            --run_name $r.name --rarity_bonus $r.bonus --resume
    }
}
