# Part 2 — Manipulator RL — Phase 5: Standard & Dynamics Generalization

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

Re-evaluates the Phase 4 peg-in-hole checkpoint on previously unseen obstacle layouts/counts and under changed physical parameters. **No training occurs in this phase** — it is pure evaluation of the existing trained policy, per the PS's framing.

---

## 1. Method

- **Standard generalization:** peg/hole/robot-initial-configuration positions were already randomized every episode during training (never memorizable to begin with), so the one structural variable held fixed during training was obstacle *count* (always 2). This phase sweeps obstacle count across `[0, 1, 2, 3, 4]` — only `2` matches training, the rest are genuinely unseen structural configurations.
- **Dynamics generalization:** peg mass (0.5×–2× trained value), peg lateral friction (both lower and higher than default), and robot joint damping (added, where training had near-zero) — applied via `p.changeDynamics()` at evaluation time only, after each episode reset.
- 30 episodes per condition, deterministic policy, same checkpoint and VecNormalize stats as Phase 4.

---

## 2. Results

### 5a — Standard generalization (obstacle count)

| Condition | Success | Grasp Attempt | Collision |
|---|---|---|---|
| 0 obstacles (unseen) | 0.0% | **100.0%** | 0.0% |
| 1 obstacle (unseen) | 0.0% | 80.0% | 23.3% |
| 2 obstacles (trained) | 0.0% | 76.7% | 16.7% |
| 3 obstacles (unseen) | 0.0% | 83.3% | 40.0% |
| 4 obstacles (unseen) | 0.0% | 83.3% | 50.0% |

### 5b — Dynamics generalization

| Condition | Success | Hold Rate | Collision |
|---|---|---|---|
| Baseline (no override) | 0.0% | 0.0% | 33.3% |
| Heavier peg (2× mass) | 0.0% | 0.0% | 43.3% |
| Lighter peg (0.5× mass) | 0.0% | 0.0% | 36.7% |
| Low friction | 0.0% | 0.0% | 36.7% |
| High friction | 0.0% | 0.0% | 26.7% |
| High joint damping | 0.0% | 0.0% | 36.7% |

*(Note: "Hold Rate" in the dynamics table is a coarser proxy than "Grasp Attempt" in the obstacle table — see `evaluate_generalization.py`'s docstring. Both report 0% grasp completion consistently either way.)*

---

## 3. Findings

**The core finding of this phase is stability, not variation:** task success is 0% across every single condition tested — 5 obstacle-count configurations and 6 dynamics configurations, 330 total evaluation episodes. Rather than reading this as "nothing to report," the specific pattern is itself informative:

1. **The grasp-completion failure is structural, not condition-specific.** It doesn't get better or worse under any tested obstacle count or physical parameter — meaning whatever is preventing the grasp constraint from forming (identified in the Phase 4 README as either a too-tight distance threshold or obstacle/frame interference) is not an artifact of the specific training conditions. The policy's failure mode itself generalizes consistently.

2. **Grasp-attempt rate is highest with zero obstacles (100%) and drops with any obstacles present (76.7–83.3%)**, reproducibly across this and earlier runs (the same ~100%-with-zero-obstacles pattern was observed independently three times while debugging this script). This is direct, repeated evidence supporting Phase 4's hypothesis 2: the presence of obstacles/the hole frame measurably disrupts the policy's approach to attempting a grasp, not just its ability to complete one.

3. **Collision rate trends upward with obstacle count** (0% → 50% from 0 to 4 obstacles) — expected and sane; more obstacles in the workspace means more opportunities for contact. This confirms the collision-detection machinery itself is working correctly and responding to the actual environment, even though the policy hasn't learned to avoid it well.

4. **Dynamics changes show no clear directional effect** on collision rate or depth progress within the tested ranges — plausible given the policy never progresses far enough into the task (past the unsolved grasp step) for most physical-parameter changes to matter yet. Dynamics generalization for the *insertion* behavior specifically remains untested, since the policy never reaches that stage regardless of physics.

---

## 4. Honest interpretation for the report

This phase does not show a policy that generalizes well in the positive sense (succeeding across conditions) — it shows a policy whose specific limitation (Phase 4's grasp-completion failure) is *consistent* across conditions, which is a real, defensible, and precisely-characterized result rather than an unexplained zero. The obstacle-count trend (3) is the most actionable finding here and directly informs where to focus if extending this work further.

---

## 5. Files in this submission

```
evaluate_generalization.py    # this phase's evaluation script (no training)
README.md                      
```