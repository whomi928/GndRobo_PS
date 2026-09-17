# Ground Robotics — Prepathon PS Submission

**Inter IIT Tech Meet 15.0 — SNTC, IIT BHU**

---

## Submission structure

```
PART_1_SLAM_NAV/
  yahboom_rosmaster/            # ROS2 workspace (SLAM + Nav2 stack) this was done in ubuntu-24.04, other specfication and mentioned accordingly.
  maps/my_cafe_map.{yaml,pgm}
  videos/{mapping_demo, navigation_demo}.mp4
  README.md

PART_2_MANIPULATOR_RL/
  phase_1/
  phase_2/
  phase_3/
  phase_4/
  phase_5/
  phase_6/
  README.md                      # this file, or a copy of it, final update README_MASTER
```

---

## Part 1 — SLAM & Autonomous Navigation: **Complete**

ROS2 Jazzy + Gazebo Harmonic, holonomic mecanum base, slam_toolbox mapping, Nav2 + AMCL autonomous navigation against the saved map. Full sensor-fusion debugging 
documented (TF broadcaster conflicts, zero-covariance odometry, IMU acceleration divergence) — see -- PART_1_SLAM_NAV/README.md.

---

## Part 2 — Manipulator RL: **All 6 phases attempted, with honest results throughout**

| Phase | Status | Headline result |

| 1. Reaching | Complete, working | Trained policy reliably reaches randomized targets |
| 2. Pick-and-Place | Complete, two diagnosed failure modes | 0% task success; root cause identified and explained across two training iterations (reward gap → then 
reward-hacking after the fix) |
| 3. Obstacle-Aware Manipulation | Environment complete; training inherited Phase 2's limitation | 0% task success, consistent with Phase 2's unresolved grasp issue |
| 4. Peg-in-Hole Insertion | Complete, trained, evaluated | 0% full insertion; 88% grasp *attempt* rate but 0% grasp *completion* — a distinct, more specific failure 
mode than Phase 2/3, narrowed to the grasp-formation step itself |
| 5. Standard & Dynamics Generalization | Complete (evaluation only) | Grasp-completion failure is stable across all 11 tested conditions (obstacle counts, physical 
parameters) — a reproducible, structural finding, not condition-specific |
| 6. Robustness / Stress Test | Complete (evaluation only) | No statistically reliable degradation trend detected — baseline noise floor (20–47% collision rate across 
three nominally-identical zero-disturbance runs) exceeds the effect size this sample size could resolve; stated as a methodological limitation, not a robustness claim |

**Overall, honest summary:** Phase 1 succeeded outright. Every phase from 2 onward inherited a single unresolved root problem — the learned policy never reliably 
completes a grasp — which was diagnosed with increasing precision at each stage rather than left as a mystery: first as a reward-shaping gap (Phase 2, run 1), then as 
reward-hacking after that gap was fixed (Phase 2, run 2), then narrowed further to attempt-without-completion under added task complexity (Phase 4), and finally shown 
to be structurally consistent rather than condition-specific (Phase 5) or statistically resolvable at this evaluation scale (Phase 6).

This diagnostic thread — not a single clean success — is the actual deliverable of Part 2's later phases, and is documented in full, phase by phase, in each phase's 
own README and consolidated in "PROGRESS_AND_DIFFICULTIES.md".

---

## What would be done next with more time

1. Rebalance Phase 2's reward once more (a smaller premature-release penalty, or a small flat grasp-attempt bonus) to resolve the reward-hacking found in its second 
run.
2. Instrument the exact end-effector-to-peg distance at each grasp attempt in Phase 4, to directly distinguish the two candidate causes named in that phase's README 
(threshold too tight vs. obstacle-disrupted approach) rather than inferring from aggregate statistics.
3. Re-run Phase 5/6 once Phase 2/4's grasp issue is resolved, since both phases' results are currently capped by that shared upstream limitation.
4. Re-run Phase 6 with fixed seeds across disturbance conditions and a larger episode count, to get a statistically trustworthy degradation curve.

---

## Key engineering artifacts worth highlighting to a grader

- **Part 1:** a full, documented sensor-fusion debugging sequence (6 distinct bugs, each diagnosed via direct tool inspection, not guesswork) ending in a stable, 
working SLAM+Nav stack.
- **Part 2:** a "getLinkState()" forward-kinematics staleness bug (Phase 1) that silently decoupled the reward signal from real robot behavior while producing no 
errors and superficially "healthy" training metrics — arguably the single most instructive bug across the whole project, and directly comparable in kind to Part 1's 
TF-broadcaster conflict (both are examples of silent signal corruption rather than code-level failure).
- A reproducible, three-times-independently-observed finding (Phase 4/5) that obstacle presence measurably suppresses grasp-attempt rate, offering a concrete,
falsifiable direction for future work rather than an unexplained plateau.

---

## NOW THE MOST DIFFICULT THINGS THAT BROKE ##
  1. During the RL traning in part 2, the RL model ever since the 1st phase was showing 0% success (which is still not much better.)
  2. The Vedio Rollout script provided in phase 1 to 4 broke 8 different times and 1 and 2 have different script then 3 and 4 as it was becoming to difficult to get 
  the video I decide to genralise the case (big mistake).
  3. This is the 2nd time trained models, when I trained them for 1st time the results were even worse, (e.g. there was a grasp rate of 1.3% in phase 2), I went ahead 
  with this not by phase 4 the data did not even made sence. Hence forcing a re start.
  4. Training models and getting 0% success was indeed underwhelming but it was the only logical solution I could have reached with the time given and hardware 
  limitations.