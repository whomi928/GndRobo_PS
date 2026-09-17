# Part 2 — Manipulator RL — Phase 6: Robustness / Stress Test

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

Evaluates the Phase 4 checkpoint under observation noise, action/control noise, and small peg perturbations — injected only at evaluation time, never seen during training. **No training occurs in this phase.**

---

## 1. Method

Three disturbance types, swept independently (one active at a time, others held at zero) so effects aren't confounded:
- **Observation noise:** Gaussian noise (std 0.0–0.2) added to the normalized observation before the policy acts — simulates imperfect state estimation.
- **Action/control noise:** Gaussian noise (std 0.0–0.2) added to the policy's chosen action before it's applied — simulates imperfect actuation.
- **Peg perturbation:** a one-time random-direction external force (0–4 N) applied to the peg at a random step within each episode — simulates an external bump.

30 episodes per condition, deterministic policy, same Phase 4 checkpoint and VecNormalize stats throughout.

---

## 2. Results

| Condition | Success | Hold Rate | Collision Rate |
|---|---|---|---|
| obs_noise = 0.0 (baseline) | 0.0% | 0.0% | 20.0% |
| obs_noise = 0.02 | 0.0% | 0.0% | 33.3% |
| obs_noise = 0.05 | 0.0% | 0.0% | 30.0% |
| obs_noise = 0.10 | 0.0% | 0.0% | 26.7% |
| obs_noise = 0.20 | 0.0% | 0.0% | 36.7% |
| action_noise = 0.0 (baseline) | 0.0% | 0.0% | 33.3% |
| action_noise = 0.02 | 0.0% | 0.0% | 36.7% |
| action_noise = 0.05 | 0.0% | 0.0% | 33.3% |
| action_noise = 0.10 | 0.0% | 0.0% | 26.7% |
| action_noise = 0.20 | 0.0% | 0.0% | 33.3% |
| perturb = 0.0 N (baseline) | 0.0% | 0.0% | 46.7% |
| perturb = 0.5 N | 0.0% | 0.0% | 33.3% |
| perturb = 1.0 N | 0.0% | 0.0% | 43.3% |
| perturb = 2.0 N | 0.0% | 0.0% | 33.3% |
| perturb = 4.0 N | 0.0% | 0.0% | 20.0% |

---

## 3. Findings — and an honest statistical caveat

**Success rate is 0% at every noise level, in every disturbance category** — a direct, expected consequence of Phase 4's baseline result (grasp never completes even with zero disturbance), not a new finding specific to this phase. **Hold rate is likewise 0% everywhere**, for the same reason.

**Collision rate does not show a clear, trustworthy degradation trend.** This needs to be stated precisely rather than glossed over: the *nominally identical* zero-disturbance baseline was run three separate times (once per sweep — `obs_noise=0.0`, `action_noise=0.0`, `perturb=0.0`), and it measured **20.0%, 33.3%, and 46.7% collision rate respectively** — a 26.7-percentage-point spread despite representing the exact same condition. This tells us directly that with only 30 episodes per condition, the environment's own episode-to-episode randomization (obstacle/peg/hole positions) produces enough natural variance that the collision-rate differences seen across the actual noise sweep (roughly 20–47% throughout) are **not distinguishable from this baseline noise floor.** No condition's collision rate falls outside the range already observed across the three baseline runs alone.

**Honest conclusion: this sweep does not provide statistically reliable evidence of either graceful or brittle degradation**, because the sample size (30 episodes/condition) is too small relative to the environment's own baseline variance to resolve the effect being measured. This is a methodological limitation of this evaluation, not a claim about the policy's actual robustness — a real answer would need either many more episodes per condition, or variance-reduction (e.g., fixing obstacle/peg/hole seeds across conditions so only the swept disturbance varies between runs) to isolate the disturbance's effect from ordinary environment randomness.

---

## 4. What Phase 6 would need to be conclusive (if time allows)

1. **Fixed seeds across conditions** — evaluate every disturbance level against the *same* set of randomized episode configurations, rather than independently re-randomizing each time, so the only thing that differs between conditions is the disturbance itself.
2. **More episodes per condition** (100+) to shrink the confidence interval enough to distinguish a real trend from sampling noise, given the baseline's own ~27-point observed spread.
3. Once Phase 4's underlying grasp-completion issue is resolved (see Phase 4 README), re-run this sweep — success-rate degradation curves are only meaningful once the baseline success rate is non-zero to begin with.

---

## 5. Files in this submission

```
evaluate_robustness.py    # this phase's evaluation script (no training)
README.md                 
```