"""Write paper/deck/deck.html from numbers.json and the figures (run make_figures.py first).
The PDF is printed from that page with headless Chromium (to_pdf.py)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json")))
T, E, CFG = N["train"], N["eval"], N["configs"]
OURS, PPO, PPOR, AD = "AD + rarity β=2 (ours)", "PPO", "PPO + rarity β=2", "AD (1 seed)"


def pm(x, d=2):
    m, s = x
    return f"{m:.{d}f} ± {s:.{d}f}" if s else f"{m:.{d}f}"


def rate(cfg, a, d=1):
    return f"{T[cfg]['rates'][a][0]:.{d}f}"


sub = N.get("submitted")
sub_row = ""
if sub:
    sub_row = (f"<tr class='hl2'><td>Ours, re-run with the submitted <code>train.py</code></td><td>{len(sub['runs'])}</td>"
               f"<td>{pm(sub['score'])}</td><td>{pm(sub['reward'])}</td><td>{pm(sub['eval_score'], 1)}</td>"
               f"<td>{pm(sub['eval_reward'], 1)}</td></tr>")
sub_note = (f"Re-running the compact submitted code from scratch gives {pm(sub['score'])} over {len(sub['runs'])} seeds."
            if sub else "The compact submitted code was checked against the reference implementation (see Engineering).")

css = """
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box }
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8985; --line:#e5e4e0; --blue:#2a78d6; --orange:#eb6834;
        --aqua:#1baf7a; --yellow:#eda100; --violet:#6250d6; --tint:#f4f7fc; --surface:#ffffff }
html, body { margin:0; background:var(--surface); color:var(--ink); font-family:'Noto Sans', 'Ubuntu Sans', sans-serif }
.slide { width:1280px; height:720px; padding:46px 60px 40px; position:relative; overflow:hidden; page-break-after:always;
         display:flex; flex-direction:column }
h1 { font-size:31px; margin:0 0 6px; letter-spacing:-0.3px; font-weight:700 }
h2 { font-size:17px; margin:0 0 18px; color:var(--ink2); font-weight:400 }
p, li { font-size:17.5px; line-height:1.45; margin:0 0 9px }
ul { margin:0; padding-left:22px }
li::marker { color:var(--blue) }
b, strong { font-weight:700 }
.foot { position:absolute; bottom:16px; left:60px; right:60px; font-size:11.5px; color:var(--muted); display:flex; justify-content:space-between }
.row { display:flex; gap:34px; flex:1; min-height:0 }
.col { flex:1; min-width:0 }
.fig { display:block; max-width:100%; max-height:100%; object-fit:contain }
.kpi { display:flex; gap:18px; margin:6px 0 16px }
.kpi > div { flex:1; border:1px solid var(--line); border-radius:10px; padding:12px 16px }
.kpi .v { font-size:34px; font-weight:700; letter-spacing:-0.5px }
.kpi .l { font-size:13.5px; color:var(--ink2) }
.box { background:var(--tint); border-radius:10px; padding:14px 18px; margin-top:10px }
.box p { margin:0 0 6px; font-size:16.5px }
table { border-collapse:collapse; width:100%; font-size:15.5px }
th { text-align:left; font-weight:600; color:var(--ink2); border-bottom:1.5px solid var(--ink); padding:7px 8px; font-size:14px }
td { border-bottom:1px solid var(--line); padding:8px 8px }
td:not(:first-child), th:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums }
tr.hl td { background:#eaf2fc; font-weight:700 }
tr.hl2 td { background:#f4f7fc }
.dot { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:7px; vertical-align:baseline }
.small { font-size:14px; color:var(--ink2) }
.tag { display:inline-block; font-size:12px; font-weight:700; letter-spacing:0.6px; color:var(--blue); text-transform:uppercase; margin-bottom:8px }
.eq { font-size:15px; margin:4px 0 10px }
.eq .katex-display { margin:6px 0 }
table.tight td { padding:4px 8px }
.stills { display:flex; gap:16px }
.stills figure { margin:0; flex:1 }
.stills img { width:100%; border-radius:6px; display:block }
.stills figcaption { font-size:13.5px; line-height:1.35; color:var(--ink2); margin-top:7px }
.diagram text { font-family:'Noto Sans', sans-serif }
code { white-space:nowrap; font-family:'Ubuntu Sans Mono', monospace; font-size:0.92em; background:#f1f0ec; padding:1px 5px; border-radius:4px }
.title { justify-content:center; padding-left:90px }
.title h1 { font-size:48px; line-height:1.12; margin-bottom:16px }
.title h2 { font-size:21px; max-width:900px; line-height:1.4 }
"""

FOOT = "Barbu Alexandru · Crafter RL project · November 2026"


def slide(body, n, cls=""):
    return f'<section class="slide {cls}">{body}<div class="foot"><span>{FOOT}</span><span>{n}</span></div></section>'


S = []
S.append(f"""
<div class="tag">Reinforcement learning project · Crafter (Hafner, 2022)</div>
<h1>Chasing the diamond in Crafter</h1>
<h2>PPO with Achievement Distillation, plus a <b>rarity bonus</b> that rewards the achievements
the agent almost never gets. 1M environment steps, 3 seeds per configuration.</h2>
<div class="kpi" style="max-width:980px;margin-top:28px">
  <div><div class="v" style="color:var(--blue)">{pm(T[OURS]['score'])}</div><div class="l">Crafter score, ours (AD + rarity β=2)</div></div>
  <div><div class="v">{pm(T[PPO]['score'])}</div><div class="l">PPO baseline (paper: 15.60 ± 1.66)</div></div>
  <div><div class="v">{N['random']['score']:.2f}</div><div class="l">uniformly random agent</div></div>
</div>
<p class="small" style="margin-top:14px">Barbu Alexandru · November 2026 · code: <code>python train.py</code> trains the best configuration</p>
""")

S.append(f"""
<h1>The task: Crafter, a 2D Minecraft with a 1M-step budget</h1>
<h2>Every episode is a new procedurally generated world. The agent sees 64×64 pixels and has 17 actions.</h2>
<div class="row">
 <div class="col">
  <ul>
   <li><b>22 achievements</b> form a tech tree: wood → table → wood pickaxe → stone → stone pickaxe → coal + furnace → iron → iron pickaxe → <b>diamond</b>. Survival needs food, water, sleep and fighting zombies at night.</li>
   <li><b>Reward:</b> +1 the first time each achievement is unlocked in an episode, ±0.1 per health point gained or lost.</li>
   <li><b>Score:</b> success rate s<sub>i</sub> = % of all training episodes that unlocked achievement i, then a geometric mean:</li>
  </ul>
  <div class="eq" style="margin-left:22px">$$S = \\exp\\Big(\\tfrac{{1}}{{22}}\\textstyle\\sum_{{i=1}}^{{22}} \\ln(1 + s_i)\\Big) - 1$$</div>
  <ul>
   <li>The geometric mean is <b>dominated by the rare achievements</b>: taking one from 0% to 1% helps the score as much as taking another from 10% to 21%.
       This is the idea behind our change.</li>
   <li>Protocol: 1M steps of training, score over <i>all</i> training episodes, mean of 3 seeds.</li>
  </ul>
 </div>
 <div style="width:330px"><img class="fig" src="figs/stills/iron_pickaxe_night.png" style="border-radius:8px">
 <p class="small" style="margin-top:8px">Our agent (seed 0, final weights) mines iron and makes an iron pickaxe at step 178, at night.</p></div>
</div>
""")

S.append(f"""
<h1>Method: PPO with Achievement Distillation (AD)</h1>
<h2>AD (Moon et al., NeurIPS 2023) is the strongest published model-free method on Crafter. We re-implemented it and built on it.</h2>
<svg class="diagram" viewBox="0 0 1160 300" width="1160" height="300">
 <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#52514e"/></marker></defs>
 <g font-size="15" fill="#0b0b0b">
  <rect x="0" y="40" width="130" height="70" rx="8" fill="#f4f7fc" stroke="#cbd9ee"/><text x="65" y="70" text-anchor="middle" font-weight="700">frame o_t</text><text x="65" y="92" text-anchor="middle" font-size="13" fill="#52514e">64×64×3</text>
  <rect x="175" y="25" width="200" height="100" rx="8" fill="#f4f7fc" stroke="#cbd9ee"/><text x="275" y="58" text-anchor="middle" font-weight="700">IMPALA ResNet</text><text x="275" y="80" text-anchor="middle" font-size="13" fill="#52514e">3 stacks 64/128/128 ch,</text><text x="275" y="98" text-anchor="middle" font-size="13" fill="#52514e">GroupNorm, → 256 → 1024</text>
  <rect x="420" y="40" width="120" height="70" rx="8" fill="#f4f7fc" stroke="#cbd9ee"/><text x="480" y="70" text-anchor="middle" font-weight="700">latent z_t</text><text x="480" y="92" text-anchor="middle" font-size="13" fill="#52514e">1024</text>
  <rect x="420" y="150" width="120" height="86" rx="8" fill="#eaf6f1" stroke="#a6dcc6"/><text x="480" y="178" text-anchor="middle" font-weight="700">achievement</text><text x="480" y="197" text-anchor="middle" font-weight="700">state g</text><text x="480" y="219" text-anchor="middle" font-size="12.5" fill="#52514e">last unlock, 1024</text>
  <rect x="600" y="70" width="110" height="80" rx="8" fill="#fff" stroke="#c9c8c2"/><text x="655" y="105" text-anchor="middle" font-weight="700">[z_t, g]</text><text x="655" y="127" text-anchor="middle" font-size="13" fill="#52514e">2048</text>
  <rect x="760" y="35" width="150" height="56" rx="8" fill="#eaf2fc" stroke="#9ec5f4"/><text x="835" y="60" text-anchor="middle" font-weight="700">policy π(a|s)</text><text x="835" y="80" text-anchor="middle" font-size="13" fill="#52514e">17 actions</text>
  <rect x="760" y="125" width="150" height="56" rx="8" fill="#eaf2fc" stroke="#9ec5f4"/><text x="835" y="150" text-anchor="middle" font-weight="700">value V(s)</text><text x="835" y="170" text-anchor="middle" font-size="13" fill="#52514e">normalised targets</text>
  <rect x="960" y="15" width="200" height="120" rx="8" fill="#fdf0ea" stroke="#f6b89a"/><text x="1060" y="42" text-anchor="middle" font-weight="700">reward to learner</text><text x="1060" y="68" text-anchor="middle" font-size="14">game reward r_t</text><text x="1060" y="90" text-anchor="middle" font-size="14">+ β(1 − p_i) on a</text><text x="1060" y="110" text-anchor="middle" font-size="14">first unlock of i (ours)</text>
  <rect x="960" y="160" width="200" height="120" rx="8" fill="#f2f0fc" stroke="#c5bdf2"/><text x="1060" y="187" text-anchor="middle" font-weight="700">AD auxiliary phase</text><text x="1060" y="211" text-anchor="middle" font-size="13.5">every 8 PPO updates:</text><text x="1060" y="231" text-anchor="middle" font-size="13.5">contrastive prediction +</text><text x="1060" y="251" text-anchor="middle" font-size="13.5">OT matching of achievements</text>
 </g>
 <g stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#a)">
  <path d="M130 75H172"/><path d="M375 75H417"/><path d="M540 75Q570 75 597 100"/><path d="M540 193Q570 193 597 128"/>
  <path d="M710 100Q735 63 757 63"/><path d="M710 120Q735 153 757 153"/>
  <path d="M480 112V147" stroke-dasharray="4 3"/>
 </g>
 <text x="0" y="250" font-size="13.5" fill="#52514e">g = normalise(enc(o after unlock) − enc(o before)),</text>
 <text x="0" y="270" font-size="13.5" fill="#52514e">kept until the next unlock, zero at episode start</text>
</svg>
<div class="row" style="margin-top:4px">
 <div class="col"><p><b>Loop.</b> 8 envs × 512 steps = 4,096 steps per update, 245 updates = 1M steps. GAE returns (γ = 0.95, λ = 0.65), then PPO: 3 epochs × 8 minibatches of 512, Adam 3e-4, grad-norm 0.5.</p></div>
 <div class="col"><p><b>Auxiliary phase</b> (every 8 updates, 6 epochs on the last 8 rollouts split into whole episodes): shapes the encoder so that achievements have consistent embeddings, while KL and value terms keep π and V in place.</p></div>
</div>
""")

S.append(r"""
<h1>Objective functions</h1>
<h2>PPO every update; the two AD losses in the auxiliary phase. Each AD loss has its own Adam optimiser.</h2>
<div class="row">
 <div class="col">
  <div class="tag">PPO (clipped surrogate, GAE)</div>
  <div class="eq">$$\begin{aligned}L_{\text{PPO}} = &-\mathbb{E}_t\big[\min(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\,\hat A_t)\big]\\ &+ c_v\,\mathbb{E}_t\big[(V_\theta(s_t) - \hat R_t)^2\big] - c_e\,\mathbb{E}_t\big[\mathcal H(\pi_\theta(\cdot|s_t))\big]\end{aligned}$$</div>
  <p class="small">\(\rho_t = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)\), \(\epsilon = 0.2,\ c_v = 0.5,\ c_e = 0.01\). Value targets are normalised by a running mean / variance.</p>
  <div class="eq">$$\hat A_t = \textstyle\sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l},\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\quad \hat R_t = \hat A_t + V(s_t)$$</div>
  <div class="tag" style="margin-top:8px">Reward the learner sees (ours)</div>
  <div class="eq">$$r'_t = r_t + \beta\,(1 - p_i)$$</div>
  <p class="small">added on the step where achievement \(i\) is first unlocked in the episode; \(p_i\) = share of the last 200 episodes that unlocked \(i\); \(\beta = 2\).</p>
 </div>
 <div class="col">
  <div class="tag">AD: intra-episode prediction</div>
  <div class="eq">$$L_{\text{pred}} = -\log \frac{e^{\langle \hat g(z_t, a_t),\, g^{+}\rangle/\tau}}{e^{\langle \hat g(z_t, a_t),\, g^{+}\rangle/\tau} + e^{\langle \hat g(z_{t'}, a_{t'}),\, g^{+}\rangle/\tau}}$$</div>
  <p class="small">\(g^{+}\) = embedding of the <i>next</i> achievement unlocked after step t; \(t'\) = random step of the same episode; \(\hat g\) = MLP on the FiLM-modulated latent; \(\tau = 0.1\).</p>
  <div class="tag" style="margin-top:6px">AD: cross-episode matching</div>
  <p class="small" style="margin-bottom:4px">Achievements of two episodes are paired by entropic partial optimal transport on \(1 - \cos\) (reg 0.05; pairs with plan weight > 0.5). Same contrastive form: anchor \(g_A\), positive = its match \(g_B\), negative = a random achievement of B.</p>
  <div class="tag" style="margin-top:8px">Both auxiliary losses add</div>
  <div class="eq">$$+\ \beta_\pi\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{before aux}}\big) + \beta_V\,(V_\theta(s) - \hat R)^2,\quad \beta_\pi = \beta_V = 1$$</div>
 </div>
</div>
""")

S.append(f"""
<h1>Our enhancement: a rarity bonus for rare achievements</h1>
<h2>Pay more for what the agent rarely does. The bonus fades by itself as an achievement becomes common.</h2>
<div class="row">
 <div class="col">
  <ul>
   <li>When achievement i is unlocked for the first time in an episode, add <b>β · (1 − p<sub>i</sub>)</b>, where p<sub>i</sub> is the share of the last 200 finished episodes that unlocked i.</li>
   <li>Common achievements stay worth ≈ +1; one never seen before is worth 1 + β. That fits the geometric-mean score.</li>
   <li>An automatic curriculum with no hand-written tech tree. About 25 lines (<code>envs.RarityBonus</code>), and it works with any algorithm.</li>
   <li>Logged rewards and the score always use the <b>game's</b> reward.</li>
  </ul>
  <div class="box">
   <p><b>PPO, 3 seeds per β:</b> β = 0 / 1 / 2 / 4 → {N['beta']['0'][0]:.2f} / {N['beta']['1'][0]:.2f} / <b>{N['beta']['2'][0]:.2f}</b> / {N['beta']['4'][0]:.2f}.</p>
   <p>β = 2 beats PPO on every seed (+8.4, +3.5, +3.7; paired t = 3.26, one-sided p ≈ 0.04), and the spread across seeds shrinks from ± 2.99 to ± 0.87.</p>
   <p>On AD (seed 0): 17.71 → <b>23.08</b>. With 3 seeds: <b>{pm(T[OURS]['score'])}</b>.</p>
  </div>
 </div>
 <div style="width:420px"><img class="fig" src="figs/beta.svg"></div>
</div>
""")

S.append(f"""
<h1>Results: ours beats PPO and the published AD result</h1>
<h2>1M steps. Training score = Crafter protocol (all training episodes). Evaluation = final weights on 200 fresh worlds per seed.</h2>
<table>
 <tr><th>configuration</th><th>seeds</th><th>training score</th><th>training reward (all episodes)</th><th>eval score</th><th>eval reward</th></tr>
 <tr><td><span class="dot" style="background:#8a8985"></span>Random agent</td><td>–</td><td>{N['random']['score']:.2f}</td><td>{N['random']['reward']:.2f}</td><td>{N['random']['score']:.1f}</td><td>{N['random']['reward']:.1f}</td></tr>
 <tr><td><span class="dot" style="background:#eda100"></span>PPO</td><td>3</td><td>{pm(T[PPO]['score'])}</td><td>{pm(T[PPO]['reward_all'])}</td><td>{pm(E[PPO]['score'], 1)}</td><td>{pm(E[PPO]['reward'], 1)}</td></tr>
 <tr><td><span class="dot" style="background:#eb6834"></span>PPO + rarity β=2</td><td>3</td><td>{pm(T[PPOR]['score'])}</td><td>{pm(T[PPOR]['reward_all'])}</td><td>{pm(E[PPOR]['score'], 1)}</td><td>{pm(E[PPOR]['reward'], 1)}</td></tr>
 <tr><td><span class="dot" style="background:#1baf7a"></span>AD (paper config)</td><td>1</td><td>{pm(T[AD]['score'])}</td><td>{pm(T[AD]['reward_all'])}</td><td>{pm(E[AD]['score'], 1)}</td><td>{pm(E[AD]['reward'], 1)}</td></tr>
 <tr class="hl"><td><span class="dot" style="background:#2a78d6"></span>AD + rarity β=2 (ours)</td><td>3</td><td>{pm(T[OURS]['score'])}</td><td>{pm(T[OURS]['reward_all'])}</td><td>{pm(E[OURS]['score'], 1)}</td><td>{pm(E[OURS]['reward'], 1)}</td></tr>
 {sub_row}
</table>
<div class="row" style="margin-top:22px">
 <div class="col"><p><b>Published references</b> (Moon et al., 2023; 1M steps): PPO 15.60 ± 1.66, AD 21.79 ± 1.37 (10 seeds). Our PPO reproduces the paper (15.75); ours is ~1.9 points above published AD.</p></div>
 <div class="col"><p><b>Caveat.</b> Our own plain-AD run has 1 seed and scored low (17.71), so the AD vs AD + bonus gap is measured best on PPO (3 + 3 seeds). {sub_note}</p></div>
</div>
""")

S.append(f"""
<h1>Performance during training: Crafter score</h1>
<h2>Cumulative score over all episodes so far, mean ± sd over seeds. The bonus pulls ahead after ~500k steps.</h2>
<div class="row">
 <div style="width:690px"><img class="fig" src="figs/score_curves.svg"></div>
 <div class="col">
  <ul>
   <li>All methods are identical for the first ~150k steps: easy achievements (wood, table, sapling) come first.</li>
   <li>The gap opens when the stone pickaxe and furnace are discovered, at 400–650k steps.</li>
   <li>The score at 1M is still rising: no method has converged within the budget.</li>
   <li>Ours ends at <b>{T[OURS]['score'][0]:.1f}</b>; the last-100-episode score at the end is ≈ 33.</li>
  </ul>
 </div>
</div>
""")

S.append(f"""
<h1>Average episodic reward</h1>
<h2>Game reward per episode (the bonus is not included), episodes binned per 25.6k steps, mean ± sd over seeds.</h2>
<div class="row">
 <div style="width:690px"><img class="fig" src="figs/reward_curves.svg"></div>
 <div class="col">
  <ul>
   <li>Random agent: <b>{N['random']['reward']:.2f}</b> per episode (1,008 episodes).</li>
   <li>Final reward (last 100 training episodes): PPO {pm(T[PPO]['reward_last100'], 1)}, PPO + rarity {pm(T[PPOR]['reward_last100'], 1)}, ours <b>{pm(T[OURS]['reward_last100'], 1)}</b> (paper AD: 12.6).</li>
   <li>The reward rewards <i>how many</i> achievements an episode gets; the score rewards <i>which</i>. The bonus helps the score more than the reward, because it targets the rare ones.</li>
  </ul>
 </div>
</div>
""")

S.append(f"""
<h1>Success rate on each skill (training, all 1M steps)</h1>
<h2>Log scale, as in the Crafter paper. Achievements ordered by depth in the tech tree.</h2>
<img class="fig" src="figs/success_train.svg" style="width:100%;max-height:430px">
<div class="row" style="margin-top:8px">
 <div class="col"><p>Ours vs PPO: stone pickaxe <b>{rate(OURS,'make_stone_pickaxe')}%</b> vs {rate(PPO,'make_stone_pickaxe')}%, collect iron <b>{rate(OURS,'collect_iron')}%</b> vs {rate(PPO,'collect_iron')}%, furnace {rate(OURS,'place_furnace')}% vs {rate(PPO,'place_furnace')}%.</p></div>
 <div class="col"><p>Never or almost never solved by anyone: eat plant, iron tools, <b>diamond</b>. Cost of the bonus: wake up {rate(OURS,'wake_up')}% vs {rate(AD,'wake_up')}% for plain AD.</p></div>
</div>
""")

S.append(f"""
<h1>Performance during evaluation: the final policies</h1>
<h2>Final weights, stochastic policy as in training, 200 fresh worlds per seed (600 per configuration; plain AD: 200).</h2>
<div class="row">
 <div style="width:640px"><img class="fig" src="figs/eval_bars.svg"></div>
 <div class="col">
  <ul>
   <li>The evaluation score is much higher than the training score, because it measures only the final policy and not the early learning.</li>
   <li>Ours: eval score <b>{pm(E[OURS]['score'], 1)}</b>, reward <b>{pm(E[OURS]['reward'], 1)}</b>, mean length {E[OURS]['length'][0]:.0f} steps.</li>
   <li>The ranking is the same as in training, and the seeds of ours agree closely (sd {E[OURS]['score'][1]:.1f}).</li>
   <li>Every evaluated episode ends in death (none reaches the 10,000-step limit). See "how the agents die".</li>
  </ul>
 </div>
</div>
""")

S.append("""
<h1>Evaluation: skills of the final policies</h1>
<h2>Success rate per achievement over fresh episodes with the final weights (log scale).</h2>
<img class="fig" src="figs/success_eval.svg" style="width:100%;max-height:430px">
<p style="margin-top:10px">Our final agents make a stone pickaxe in most episodes and mine iron in about a third of them, but in 600 fresh episodes they never make an iron tool or find a diamond.
In a larger search (800 episodes per agent) iron tools appear only a handful of times.</p>
""")

S.append(f"""
<h1>Why the bonus works: it breaks the stone-pickaxe wall</h1>
<h2>Success rate over the last 100 training episodes, mean over seeds.</h2>
<img class="fig" src="figs/tech_wall.svg" style="width:100%">
<div class="row" style="margin-top:10px">
 <div class="col"><p>Without the bonus, agents learn the <b>stone sword</b> but stall at the <b>stone pickaxe</b> (1–2% for plain AD), although both need the same wood, stone and table. No pickaxe means no iron.</p></div>
 <div class="col"><p>With the bonus, the first stone pickaxe is worth up to 3× a common achievement, so it is discovered earlier and reinforced: 70% of recent episodes for ours at the end, and iron follows.</p></div>
</div>
""")

S.append(f"""
<h1>Other metrics: losses, entropy and episode length</h1>
<h2>Ours, 3 seeds (one line each). The AD losses are logged once per auxiliary phase.</h2>
<img class="fig" src="figs/losses.svg" style="width:100%;max-height:250px">
<div class="row" style="margin-top:12px">
 <div style="width:420px"><img class="fig" src="figs/length.svg"></div>
 <div class="col">
  <ul>
   <li>Entropy falls from 2.8 (uniform over 17 actions = 2.83) to ≈ 1.0: the policy becomes more decisive but stays stochastic.</li>
   <li>Both contrastive losses keep falling: achievement embeddings become more predictable and more consistent across episodes.</li>
   <li>Episodes get longer as agents learn (~170 → ~250 steps), but most agents still die around the first night (a day lasts 300 steps).</li>
   <li>Wall-clock: ~3.6 h per AD seed on an RTX 5090 laptop GPU, 6.9 h on a 4 GB RTX 3050; PPO alone ~1.2 h.</li>
  </ul>
 </div>
</div>
""")

d = N["deaths"]
S.append(f"""
<h1>Emergent behaviour: how the agents die</h1>
<h2>We hooked the game's damage code to log the cause of every lost health point (final weights, fresh worlds).</h2>
<div class="row">
 <div style="width:660px"><img class="fig" src="figs/deaths.svg">
 <p class="small" style="margin-top:6px">4,800 diagnosed episodes over 24 runs: 100% end in death, median length 220–260 steps.</p></div>
 <div class="col">
  <ul>
   <li><b>They stay awake at night.</b> Zombies cause about half of all deaths, mostly in the dark, but agents spend only ~4% of the night asleep.</li>
   <li><b>They neglect their vitals.</b> At death, water is at 0 in {d['AD + rarity β=2']['drink0_pct']:.0f}% of episodes for ours; with food, water or energy at 0, health stops regenerating.</li>
   <li><b>The bonus makes them sleep less:</b> episodes with any sleep fall from {d['AD']['slept_pct']:.0f}% (AD) to {d['AD + rarity β=2']['slept_pct']:.0f}% (ours). Time goes to the tech tree instead.</li>
   <li>Arrows and lava (≈ 30%) hit them in caves, which is exactly where iron and diamonds are.</li>
  </ul>
 </div>
</div>
""")

S.append("""
<h1>Emergent behaviour: what the agents do (and don't)</h1>
<h2>Frames from evaluation videos (final weights). The bar at the top shows the step, vitals and new achievements.</h2>
<div class="stills">
 <figure><img src="figs/stills/iron_pickaxe_night.png"><figcaption><b>Iron pickaxe at night.</b> Ours, seed 0: coal, furnace, iron and an iron pickaxe by step 178. A skeleton's arrow kills it at step 207.</figcaption></figure>
 <figure><img src="figs/stills/diamond_zombie.png"><figcaption><b>Walks past diamonds.</b> Killed by a zombie at night between two furnaces it built, a diamond one tile away (PPO + rarity, γ = 0.99).</figcaption></figure>
 <figure><img src="figs/stills/three_furnaces.png"><figcaption><b>Builds furnaces side by side.</b> Ours, seed 0: three furnaces, water and energy at 0; it starves beside a lake at step 435.</figcaption></figure>
 <figure><img src="figs/stills/iron_tools_diamond.png"><figcaption><b>Iron tools, 1 health.</b> With GRU memory: both iron tools at step 403, a diamond 4 tiles away, then dies of thirst and fatigue.</figcaption></figure>
</div>
<p style="margin-top:14px">Also seen: a stone sword is learned long before a stone pickaxe (same ingredients); agents carry six stone swords; they press "make iron pickaxe" when the requirements are missing (6% of all actions).</p>
""")

c = CFG
S.append(f"""
<h1>What we tried on top, and what did not help</h1>
<h2>Each idea was motivated by the death analysis and tested on 3 seeds at 1M steps.</h2>
<div class="row">
 <div style="width:620px"><img class="fig" src="figs/all_configs.svg"></div>
 <div class="col">
  <ul style="font-size:16px">
   <li><b>Longer horizon γ = 0.99</b> (nights last ~90 steps, γ = 0.95 sees ~20): {c['PPO + rarity, γ=0.99']['mean']:.1f} vs 21.0, worse on every seed, and survival is unchanged.</li>
   <li><b>GRU memory</b> (remember where water and diamonds are): {c['+ GRU memory']['mean']:.1f}, learns more slowly.</li>
   <li><b>Potential-based survival shaping</b> on food / water / energy: {c['+ survival shaping']['mean']:.1f}. It hides a −(1−γ)Φ cost per step alive; without it ("no drift") {c['+ survival shaping, no drift']['mean']:.1f}, the first change that makes agents <i>live longer</i> (+10% median life).</li>
   <li><b>Own top-down map from pixels</b> (99% accurate odometry): {c['+ own pixel map']['mean']:.1f}. Agents see water in 95% of episodes but still don't go back: memory isn't the bottleneck.</li>
   <li><b>Lesson:</b> within 1M steps, anything that adds learning signal or capacity not aimed at the tech tree slows it down.</li>
  </ul>
 </div>
</div>
""")

S.append(f"""
<h1>Beyond the budget: practising in a learned world model</h1>
<h2>An exploration, <b>not a valid 1M result</b>: the world model was trained on 7k episodes (1.76M steps) recorded from our earlier agents.</h2>
<div class="row">
 <div class="col">
  <ul>
   <li><b>World model</b> (ours, 25.6M parameters): each 7×7 game tile becomes a token (1,024 codes); a causal Transformer over 64 steps predicts the next frame (MaskGIT decoder), inventory, unlocks and death; a <b>top-down map memory</b> built from pixel odometry; a learned <b>craft rule</b> shared across tool tiers.</li>
   <li><b>Dyna-style PPO:</b> each update also trains on 256 dreams × 16 steps, started from real moments near the iron chain. Dreams are sampled (T = 1) because greedy dreams are too safe.</li>
   <li>Score <b>{c['+ dreams (frozen world model)']['mean']:.1f} → {c['+ dreams, start pool']['mean']:.1f}</b> (vs 23.7), iron collected in 15% of episodes (vs 7%), iron pickaxe up to 0.66% (vs 0.01%), and the first diamonds.</li>
  </ul>
 </div>
 <div class="col">
  <div class="box">
   <p><b>What we learned about dreams</b></p>
   <p>• Before the craft rule, the model never dreamed an iron pickaxe: it had seen only 4 in its data (0 of 41 held-out crafts). With the shared rule: 97%.</p>
   <p>• The real bottleneck for iron tools is <i>bringing a table and a furnace together</i> (2% of iron-in-hand moments), not knowing the recipe.</p>
   <p>• Greedy dreams under-count damage ~4× at night; an agent trained on them would learn to ignore the night.</p>
  </div>
 </div>
</div>
""")

S.append("""
<h1>Engineering and reproducibility</h1>
<h2>What it took to run this on a 4 GB laptop GPU, and how the submitted code was checked.</h2>
<div class="row">
 <div class="col">
  <ul>
   <li><b>Faster worlds:</b> Crafter's world generation took ~1 s per reset; with numba it takes 50 ms (identical maps). 8 envs: 180 → 1,620 steps/s.</li>
   <li><b>4 GB VRAM:</b> gradient micro-batching (same gradient, value-normaliser EWMA adjusted to β<sup>1/k</sup>); <code>--micro_batch 256</code>.</li>
   <li><b>Faster AD auxiliary phase:</b> the frozen policy is evaluated once, not every epoch, and each distinct goal is encoded once: 934 s → 544 s, same math.</li>
   <li><b>Results machine-checked:</b> the same configuration on two machines: 20.85 vs 20.90.</li>
  </ul>
 </div>
 <div class="col">
  <div class="box">
   <p><b>Submitted code</b> (<code>train.py</code>, <code>model.py</code>, <code>algo.py</code>, <code>envs.py</code>, <code>evaluate.py</code>; ~1,000 lines with comments, self-contained)</p>
   <p>• <code>python train.py</code> trains the best configuration (AD + rarity β = 2, all paper hyperparameters).</p>
   <p>• Checked against the reference AD implementation on a trained checkpoint: identical forward pass (difference 0.0); PPO and prediction losses and gradients equal to 1e-7, matching loss to 6e-5 (float rounding).</p>
   <p>• Flags reproduce the ablations: <code>--no_ad</code>, <code>--rarity_bonus</code>, <code>--gamma</code>, <code>--vitals</code>.</p>
  </div>
 </div>
</div>
""")

S.append(f"""
<h1>Conclusions</h1>
<h2>&nbsp;</h2>
<div class="row">
 <div class="col">
  <ul>
   <li><b>A rarity bonus is a cheap, effective fix for Crafter's bottleneck.</b> PPO 15.75 → {T[PPOR]['score'][0]:.2f} (3/3 seeds better), and AD + bonus reaches <b>{pm(T[OURS]['score'])}</b>, above the published AD (21.79).</li>
   <li>It works by breaking the <b>stone-pickaxe wall</b>, which opens coal → furnace → iron.</li>
   <li>The best agents now die of <b>neglected vitals and zombies at night</b>, not missing skills. Horizon, memory and survival shaping each cost tech-tree progress within 1M steps.</li>
   <li>Diamonds stay out of reach at 1M steps. Learning in a world model's dreams lifts iron tools ×6–60, but only with data beyond the budget.</li>
  </ul>
 </div>
 <div style="width:360px">
  <div class="box">
   <p><b>Next steps</b></p>
   <p>• A world model learned online from the agent's own 1M steps (honest Dyna).</p>
   <p>• Rewarding "table and furnace together" as a sub-goal.</p>
   <p>• A bonus that is aware of the tech tree (pickaxe before sword).</p>
  </div>
 </div>
</div>
""")

S.append(f"""
<h1>Appendix: hyperparameters (defaults of <code>train.py</code>)</h1>
<h2>All AD hyperparameters are the paper's; the rarity bonus is ours.</h2>
<div class="row">
 <div class="col"><table>
  <tr><th>parameter</th><th>value</th></tr>
  <tr><td>environments × steps per rollout</td><td>8 × 512 = 4,096</td></tr>
  <tr><td>updates (1M steps)</td><td>245</td></tr>
  <tr><td>discount γ / GAE λ</td><td>0.95 / 0.65</td></tr>
  <tr><td>PPO epochs × minibatches</td><td>3 × 8 (512 samples)</td></tr>
  <tr><td>clip ε / value coef / entropy coef</td><td>0.2 / 0.5 / 0.01</td></tr>
  <tr><td>Adam learning rate, grad-norm clip</td><td>3e-4, 0.5</td></tr>
  <tr><td>network</td><td>IMPALA [64,128,128] × 2 blocks</td></tr>
  <tr><td>latent / heads input</td><td>1024 / 2048, 9.23M params</td></tr>
 </table></div>
 <div class="col"><table>
  <tr><th>parameter</th><th>value</th></tr>
  <tr><td>AD phase every / epochs</td><td>8 updates / 6</td></tr>
  <tr><td>contrastive temperature τ</td><td>0.1</td></tr>
  <tr><td>KL and value distillation coefs</td><td>1.0 / 1.0</td></tr>
  <tr><td>OT regularisation, targets per source</td><td>0.05, 16 episodes</td></tr>
  <tr><td><b>rarity bonus β / window</b></td><td><b>2.0 / 200 episodes</b></td></tr>
  <tr><td>vitals shaping α (ablation)</td><td>0 (off)</td></tr>
  <tr><td>seeds</td><td>0, 1, 2</td></tr>
  <tr><td>original runs</td><td>250 updates (1.024M steps)</td></tr>
 </table></div>
</div>
""")

per_seed = "".join(
    f"<tr><td>{k}</td><td>{len(v['seeds'])}</td><td>{' / '.join(f'{x:.2f}' for x in v['seeds'])}</td><td>{v['mean']:.2f}" +
    (f" ± {v['sd']:.2f}" if len(v['seeds']) > 1 else "") + "</td></tr>"
    for k, v in CFG.items())
S.append(f"""
<h1>Appendix: every configuration, per seed</h1>
<h2>Crafter score over all training episodes (1M steps).</h2>
<table class="tight" style="font-size:13px">
<tr><th>configuration</th><th>seeds</th><th>per seed</th><th>mean ± sd</th></tr>
{per_seed}
</table>
""")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Crafter RL deck</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}]}});document.body.dataset.math='ok'"></script>
<style>{css}</style></head><body>
{''.join(slide(b, i + 1, 'title' if i == 0 else '') for i, b in enumerate(S))}
</body></html>"""
open(os.path.join(HERE, "deck.html"), "w").write(html)
print(f"{len(S)} slides -> deck.html")
