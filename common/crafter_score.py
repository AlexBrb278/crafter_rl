"""Crafter score, as defined in Hafner (ICLR 2022).

success_rate_i = % of ALL training episodes (within the step budget) that unlocked
achievement i.  score = exp(mean_i ln(1 + s_i)) - 1, with s_i in [0, 100].
Use this for every method so numbers are comparable across Crafter and Craftax-Classic.
"""
import numpy as np

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]


def success_rates(episode_achievements):
    """episode_achievements: list of dicts {name: count} (one per finished episode)."""
    n = len(episode_achievements)
    return {
        a: 100.0 * sum(ep.get(a, 0) > 0 for ep in episode_achievements) / max(n, 1)
        for a in ACHIEVEMENTS
    }


def crafter_score(rates):
    """rates: dict or array of 22 success rates in percent."""
    s = np.array(list(rates.values()) if isinstance(rates, dict) else rates, dtype=float)
    assert s.shape == (22,), s.shape
    return float(np.exp(np.mean(np.log(1.0 + s))) - 1.0)


if __name__ == "__main__":
    # Sanity check: all-100% -> 100, all-0% -> 0.
    assert abs(crafter_score([100.0] * 22) - 100.0) < 1e-9
    assert crafter_score([0.0] * 22) == 0.0
    print("ok")
