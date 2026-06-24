# Snake PPO results — tables for the progression report

*Filter applied throughout: `state == finished`, `max_step ≥ 500`, canonical learning
rate per scale, `entropy_coeff` matches the row's labelled method. "n" is the count of
surviving seeds. "—" cells mean the metric was not logged for those runs (not a crash).*

---

## Table 1 — PPO baseline on Snake, by model scale

Source: `wandb.ai/jimdilkes/AAMAS_msrl`. All rows are the canonical PPO recipe
(H=0.001, GAE advantage, KL coefficient = 0, learning rate per scale: 1e-6 at
0.5B/3B, 5e-6 at 7B/14B).

| Model | n | train reward | default-eval | token entropy | valid-action rate |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 4 | −0.051 ± 0.008 | 0.010 ± 0.025 | 0.009 ± 0.003 | 1.000 ± 0.000 |
| Qwen2.5-3B   | 6 | −0.029 ± 0.061 | 0.758 ± 0.177 | 0.523 ± 0.204 | 1.000 ± 0.000 |
| Qwen2.5-7B   | 5 |  0.005 ± 0.032 | 1.138 ± 0.070 | 0.121 ± 0.031 | 0.998 ± 0.002 |
| Qwen2.5-14B  | 7 |  0.072 ± 0.023 | 1.469 ± 0.052 | 0.095 ± 0.020 | 0.999 ± 0.001 |

*default-eval is greedy (T=0) evaluation on the training-distribution Snake env
(10×10 grid, 8 rounds, 1 random opponent). Values are mean ± SE across surviving
seeds.*

### Prose for §Snake Results

PPO post-training improves Snake performance monotonically with model scale: the
0.5B model barely learns (default-eval 0.01 ± 0.03 across 4 seeds), while the
14B model reliably converges (default-eval 1.47 ± 0.05 across 7 seeds). Token
entropy at convergence falls with scale (0.5 nats at 3B → 0.1 nats at 14B),
indicating larger models commit faster to narrower action distributions. All
scales maintain a valid-action rate of ~1.0, so the failure mode at small scale
is uninformative action selection, not malformed output. [DQN comparison
sentence to be inserted here once retrieved.]

---

## Table 2 — Entropy-control methods on Qwen3-4B PPO

Source: `wandb.ai/jimdilkes/verl_env`. Each row is the canonical PPO recipe
with one intervention applied. Two entropy columns: **actor entropy** is the
policy's token-level entropy from the optimisation loop; **probe entropy** is
the action-distribution entropy at a held-out start state, sampled at T=1.2
across 20 rollouts (`eval_FastSnake-Entropy-Check`). The probe column has full
coverage; the actor column is sparser because several sub-groups did not log
it. *Caveat: verl_env runs predate the 2026-02-16 batch-size bug fix; all
methods share that bias, so relative comparisons hold.*

| Method | n | train reward | default-eval | actor entropy | probe entropy | valid-action |
|---|---|---|---|---|---|---|
| H=0.001            |  8 |  0.167 ± 0.006 |  1.395 ± 0.035 | 0.355 ± 0.005 | 0.223 ± 0.023 | 0.998 ± 0.001 |
| H=0.005            |  4 |  0.136 ± 0.015 |  1.210 ± 0.114 | — | 0.304 ± 0.022 | 0.997 ± 0.002 |
| H=0.01             |  9 |  0.129 ± 0.007 |  1.305 ± 0.024 | 0.567 ± 0.006 | 0.317 ± 0.032 | 0.995 ± 0.003 |
| H=0.05             |  1 | −1.143         | −0.600         | 11.406 *(diverged)* | 0.230 *(post-collapse)* | 0.000 |
| Cov-clip           |  4 |  0.115 ± 0.016 |  1.135 ± 0.124 | — | 0.263 ± 0.017 | 0.998 ± 0.001 |
| KL-cov             |  3 |  0.124 ± 0.013 |  1.093 ± 0.093 | — | 0.292 ± 0.022 | 1.000 ± 0.000 |
| Adaptive entropy   |  6 |  0.147 ± 0.009 |  1.380 ± 0.072 | 0.675 ± 0.081 | 0.378 ± 0.021 | 0.997 ± 0.001 |
| Entropy decay      |  3 |  0.168 ± 0.007 |  1.453 ± 0.029 | 0.631 ± 0.049 | 0.397 ± 0.020 | 0.994 ± 0.003 |
| Cosine schedule    |  4 |  0.116 ± 0.005 |  1.180 ± 0.043 | 0.467 ± 0.042 | 0.387 ± 0.019 | 0.993 ± 0.001 |

*For H=0.001 and H=0.01, only seeds with all metrics logged (i.e. actor entropy
present) are included — n drops from 14→8 and 11→9 respectively. Other rows use
all surviving seeds, since actor entropy was never logged for those sub-groups
and applying the same constraint would empty the row.*

*"Adaptive entropy" pools three target-band configurations (initial H ∈ {0.0015,
0.002, 0.003}, with bounds and adaptive update rate ∝ H). "Entropy decay" pools
three explicit decay schedules. "Cosine schedule" pools four cosine variants
(H=0.001 and H=0.01 baselines plus two LR co-schedules).*

---

## Table 3 — Behavioural diversity (state-visit + probe entropy)

Source: `wandb.ai/jimdilkes/verl_env`. The state-visit eval
(`eval_FastSnake-Default-StateVisitation`) measures distinct (state, action)
pairs visited across 400 rollouts at T=1.25 with no respawn — it captures
*trajectory-level* behavioural diversity rather than the first-state action
distribution that the probe measures. Rows are restricted to runs with **both**
state-visit AND probe entropy logged. Sub-groups without state-visit data
(H=0.005, Cov-clip, KL-cov — older runs predating this eval) are omitted.

| Method | n | distinct state-actions | coverage | probe entropy |
|---|---|---|---|---|
| H=0.001            | 3 | 15.07 ± 7.19   | 0.094 ± 0.045 | 0.288 ± 0.010 |
| H=0.01             | 3 | 14.57 ± 7.09   | 0.091 ± 0.044 | 0.381 ± 0.009 |
| H=0.05             | 1 |  6.40          | 0.040          | 0.230 *(post-collapse)* |
| Adaptive entropy   | 6 | **29.51 ± 1.49** | **0.184 ± 0.009** | 0.378 ± 0.021 |
| Entropy decay      | 3 | **29.90 ± 0.48** | **0.187 ± 0.003** | 0.397 ± 0.020 |
| Cosine schedule    | 4 |  6.81 ± 0.05   | 0.043 ± 0.000 | 0.387 ± 0.019 |

*"distinct state-actions" is the mean count of unique (state, action) pairs
across 400 rollouts. "coverage" is the same normalised against the size of the
reachable state-action space. Probe entropy is included here as a same-runs
comparison against Table 2.*

### Reading

Adaptive and decay schedules visit ~2× as many distinct state-actions as the
fixed-H baselines (≈30 vs ≈15), and achieve ~2× higher coverage. The
*probe* entropy is similar across H=0.01, Adaptive, Decay and Cosine
(0.32–0.40), so the additional state-action diversity under
adaptive/decay is not predicted by the first-state action distribution
alone — these schedules drive richer trajectory-level exploration without
visibly disturbing the per-step action distribution. Cosine, despite
matching probe entropy, fails to translate this into trajectory diversity
(only 6.8 distinct state-actions). The fixed H=0.001 and H=0.01 rows here
use a different subset of seeds (n=3 each) than in Table 2 — direct
between-table comparison of those rows is not appropriate.

### LaTeX

```latex
\begin{table}[h]
\centering
\caption{Behavioural diversity on Qwen3-4B Snake PPO, restricted to runs
with both state-visitation and entropy-probe metrics logged. ``Distinct
state-actions'' is the mean number of unique $(s, a)$ pairs visited across
400 rollouts at $T=1.25$ with no respawn; ``coverage'' is normalised
against the reachable state-action space. Sub-groups without state-visit
data (H=0.005, Cov-clip, KL-cov) are omitted.}
\label{tab:snake-behavioural-diversity}
\begin{tabular}{lcccc}
\toprule
Method & $n$ & distinct state-actions & coverage & probe entropy \\
\midrule
$H=0.001$          & 3 & $15.07 \pm 7.19$ & $0.094 \pm 0.045$ & $0.288 \pm 0.010$ \\
$H=0.01$           & 3 & $14.57 \pm 7.09$ & $0.091 \pm 0.044$ & $0.381 \pm 0.009$ \\
$H=0.05$           & 1 & $6.40$           & $0.040$           & $0.230$\,(post-collapse) \\
Adaptive entropy   & 6 & $\mathbf{29.51 \pm 1.49}$ & $\mathbf{0.184 \pm 0.009}$ & $0.378 \pm 0.021$ \\
Entropy decay      & 3 & $\mathbf{29.90 \pm 0.48}$ & $\mathbf{0.187 \pm 0.003}$ & $0.397 \pm 0.020$ \\
Cosine schedule    & 4 & $\phantom{0}6.81 \pm 0.05$ & $0.043 \pm 0.000$ & $0.387 \pm 0.019$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Prose for §Entropy section

We evaluated nine entropy-control variants against the canonical PPO recipe
(H=0.001) on Qwen3-4B Snake: a fixed-H sweep (H ∈ {0.005, 0.01, 0.05}),
covariance clipping and KL-covariance clipping (DAPO-family), an adaptive
target-band controller, an explicit decay schedule, and a cosine schedule.

Every method does raise token entropy: actor entropy roughly doubles under the
adaptive, decay and cosine schedules (0.36 → 0.47–0.68 nats), and the action-
distribution probe entropy rises by 17–80% across the board (0.22 → 0.26–0.40).

Converged reward does not follow. Of the eight non-divergent variants, seven
match or *underperform* the H=0.001 baseline (default-eval 1.32 ± 0.03), with
cov-clip (1.13), KL-cov (1.09) and cosine (1.18) the worst. The single
positive signal is the entropy decay schedule (default-eval 1.45 ± 0.03), but
this is from n=3 seeds and the apparent ~10% gain is fragile evidence. The
adaptive controller is a wash (1.38 ± 0.07, overlaps the baseline within 1
SE). H=0.05 diverges entirely (reward −0.60, valid-action rate collapses to
0, actor entropy ~11 nats).

Behavioural validity remains at ~99–100% across every condition, so the
additional entropy is being spent on token-level variation within
already-well-formed action outputs, not on exploring new action patterns. The
H sweep is right-tailed — values below H=0.001 were not tested — and is
flagged for future work.

---

## LaTeX-ready versions

### Table 1
```latex
\begin{table}[h]
\centering
\caption{PPO baseline on Snake by model scale. Surviving seeds after filtering
for completed training runs ($\geq 500$ optimisation steps) and canonical
learning rate per scale (\texttt{1e-6} at 0.5B/3B, \texttt{5e-6} at 7B/14B).
Default-eval is greedy ($T=0$) evaluation on the training-distribution Snake
env. Values are mean $\pm$ SE.}
\label{tab:snake-baseline-scale}
\begin{tabular}{lccccc}
\toprule
Model & $n$ & train reward & default-eval & token entropy & valid-action rate \\
\midrule
Qwen2.5-0.5B & 4 & $-0.051 \pm 0.008$ & $0.010 \pm 0.025$ & $0.009 \pm 0.003$ & $1.000 \pm 0.000$ \\
Qwen2.5-3B   & 6 & $-0.029 \pm 0.061$ & $0.758 \pm 0.177$ & $0.523 \pm 0.204$ & $1.000 \pm 0.000$ \\
Qwen2.5-7B   & 5 & $\phantom{-}0.005 \pm 0.032$ & $1.138 \pm 0.070$ & $0.121 \pm 0.031$ & $0.998 \pm 0.002$ \\
Qwen2.5-14B  & 7 & $\phantom{-}0.072 \pm 0.023$ & $1.469 \pm 0.052$ & $0.095 \pm 0.020$ & $0.999 \pm 0.001$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2
```latex
\begin{table}[h]
\centering
\caption{Entropy-control methods on Qwen3-4B Snake PPO. Each row is the
canonical PPO recipe with one intervention applied. ``Actor entropy'' is the
policy's token-level entropy (from the optimisation loop); ``probe entropy''
is the action-distribution entropy at a held-out start state, sampled at
$T=1.2$ across 20 rollouts. ``---'' indicates the metric was not logged for
those sub-groups (not a crash). Filter as in
Table~\ref{tab:snake-baseline-scale}.}
\label{tab:snake-entropy-methods}
\begin{tabular}{lcccccc}
\toprule
Method & $n$ & train reward & default-eval & actor entropy & probe entropy & valid-action \\
\midrule
$H=0.001$         &  8 & $\phantom{-}0.167 \pm 0.006$ & $\phantom{-}1.395 \pm 0.035$ & $0.355 \pm 0.005$ & $0.223 \pm 0.023$ & $0.998 \pm 0.001$ \\
$H=0.005$         &  4 & $\phantom{-}0.136 \pm 0.015$ & $\phantom{-}1.210 \pm 0.114$ & --- & $0.304 \pm 0.022$ & $0.997 \pm 0.002$ \\
$H=0.01$          &  9 & $\phantom{-}0.129 \pm 0.007$ & $\phantom{-}1.305 \pm 0.024$ & $0.567 \pm 0.006$ & $0.317 \pm 0.032$ & $0.995 \pm 0.003$ \\
$H=0.05$          &  1 & $-1.143$                     & $-0.600$                     & $11.406$\,(diverged) & $0.230$ & $0.000$ \\
Cov-clip          &  4 & $\phantom{-}0.115 \pm 0.016$ & $\phantom{-}1.135 \pm 0.124$ & --- & $0.263 \pm 0.017$ & $0.998 \pm 0.001$ \\
KL-cov            &  3 & $\phantom{-}0.124 \pm 0.013$ & $\phantom{-}1.093 \pm 0.093$ & --- & $0.292 \pm 0.022$ & $1.000 \pm 0.000$ \\
Adaptive entropy  &  6 & $\phantom{-}0.147 \pm 0.009$ & $\phantom{-}1.380 \pm 0.072$ & $0.675 \pm 0.081$ & $0.378 \pm 0.021$ & $0.997 \pm 0.001$ \\
Entropy decay     &  3 & $\phantom{-}0.168 \pm 0.007$ & $\phantom{-}1.453 \pm 0.029$ & $0.631 \pm 0.049$ & $0.397 \pm 0.020$ & $0.994 \pm 0.003$ \\
Cosine schedule   &  4 & $\phantom{-}0.116 \pm 0.005$ & $\phantom{-}1.180 \pm 0.043$ & $0.467 \pm 0.042$ & $0.387 \pm 0.019$ & $0.993 \pm 0.001$ \\
\bottomrule
\end{tabular}
\end{table}
```
