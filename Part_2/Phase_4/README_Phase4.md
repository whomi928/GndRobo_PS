# Part 2 — Manipulator RL — Phase 4: Peg-in-Hole Insertion

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

Full task chain: reach → grasp → transit while avoiding obstacles → position above the hole → align → insert to required depth, remaining within positional/angular tolerance for the duration of insertion.

---

## 1. Environment

Built on `obstacle_aware_env.py` (obstacles + real collision detection) and `pick_place_env.py` (grasping), adding a hole target and insertion-tolerance tracking. Action space unchanged (8-dim: 7 joint velocities + gripper) — alignment is achieved through the same joint velocities that already control end-effector pose, consistent with the PS's "extend, don't redefine" requirement.

**Deliberate scope simplification, stated up front:** the hole is represented as a static frame (reusing the existing obstacle-collision machinery) rather than a literal drilled-bore solid — modeling true matching-bore collision geometry from primitives is a substantial geometry-engineering problem on its own, and isn't what the PS's Phase 4 text identifies as the core RL-design component (the reward/observation design for alignment/insertion is). Insertion success is judged logically: peg position/orientation within tolerance of the hole axis, sufficient depth, sustained rather than just touched once.

### Reward/observation design (the PS's stated "core RL-design component" for this phase)
Extends Phase 3's staged reward with insertion-specific shaping once the peg is held and positioned near the hole: proximity to the hole axis, orientation alignment (tilt from vertical), and depth progress are each shaped continuously rather than only rewarded at the moment of full success — giving the policy a gradient to follow even when it's still far from a complete insertion, rather than a reward that stays at zero until the task is fully solved.

---

## 2. Results

Evaluated over 50 fresh randomized episodes, deterministic policy, trained checkpoint (`evaluate_peginsert.py`), 2 obstacles (matching training):

| Metric | Value |
|---|---|
| Full insertion success rate | 0.00% |
| Grasp attempt rate | 88.00% |
| Grasp succeeded rate (constraint formed) | 0.00% |
| Episodes that ever held the peg | 0.00% |
| Episodes with a board (frame) collision | 46.00% |
| Episodes with any collision | 34.00% |
| Episodes with tolerance violated during insertion | 0.00% (never reached insertion phase) |
| Mean best insertion depth progress | 0.0149 (required: 0.04) |
| Mean best (closest) xy offset from hole | 0.3505 m |
| Mean best (closest) peg tilt from vertical | 0.0086 rad |
| Mean completion time | 300.0 steps (max — no episode terminated early) |

Video: `trained_rollout_peginsert.mp4`

---

## 3. Diagnosis — a distinct, more specific failure mode than Phase 2/3

This result is worth reading carefully against Phase 2/3's own findings, because it's a genuinely different pattern, not a repeat:

- **Phase 2's second run** showed the policy *avoiding* grasp attempts almost entirely (2% attempt rate) — a reward-hacking response to a penalty.
- **Phase 4's policy attempts grasps in 88% of episodes** — much closer to Phase 2's *original* (pre-fix) attempt rate — **but the grasp constraint never actually forms (0% success).** The policy is trying, consistently, and failing at the mechanical moment of grasp itself.

Two candidate explanations, both consistent with the data:
1. **The grasp-distance threshold may not be met in practice** given this phase's added positioning demands (navigating around obstacles and the hole's frame before ever reaching the peg) — the policy may be closing the gripper from further away than earlier phases tolerated.
2. **Frame/obstacle collisions are disrupting the approach.** 46% of episodes had a frame collision — plausible that the arm's approach to the peg is being knocked off-course by contact with the hole's frame geometry (which sits in the same general workspace) before or during a grasp attempt.

Both point toward the same next diagnostic step (see Section 5) rather than a mystery — this is a specific, falsifiable hypothesis, not just "it doesn't work."

---

## 4. Files in this submission

```
peg_insert_env.py              # environment (extends obstacle_aware_env.py's ObstacleAwareEnv)
train_peginsert.py              # PPO training script (SB3)
evaluate_peginsert.py           # baseline evaluation, PS-required metrics
evaluate_generalization.py      # Phase 5 — reuses this checkpoint, no new training
render_rollout.py               # generalized renderer (works across all phases)
models_peginsert/
  best_peginsert/best_model.zip
  vecnormalize_stats.pkl        # recovered from a periodic training checkpoint — see
                                   # PROGRESS_AND_DIFFICULTIES.md for why the final-save
                                   # version was missing and how this was resolved
trained_rollout_peginsert.mp4
README.md                        # this file
```

## 5. Known Limitations & Next Steps (if time allows)

- **0% full insertion success** — root cause narrowed to the grasp step itself failing to complete despite frequent attempts (Section 3), not a downstream alignment/insertion problem, since the policy never gets far enough to test that part of the task.
- **Immediate next diagnostic step:** log the actual end-effector-to-peg distance at the moment each grasp attempt is made, to directly test hypothesis 1 (threshold too tight) vs. hypothesis 2 (collision-disrupted approach) rather than inferring from aggregate statistics.
- Inherits Phase 2/3's constraint-based (not gripper-mechanics-based) grasping simplification, stated in the Phase 2 README.
- Given the grasp step never succeeds, the insertion-specific tolerance/depth/sustain logic (the actual novel engineering of this phase) remains functionally untested by this run — it is implemented and smoke-tested, but has not been exercised by a policy that reaches that stage.
