# Part 2 — Manipulator RL — Phase 2: Pick-and-Place

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

Extends Phase 1 (reaching) with grasping, lifting, transporting, and releasing a cylindrical peg at a randomized target — with initial robot configuration, peg position, and target position all randomized per episode.

---

## 1. Environment

Built directly on `reach_env.py`'s `ReachEnv` (robot loading, joint auto-detection, action-repeat).

| Component | Choice |
|---|---|
| Object | Cylindrical peg (radius 0.02m, height 0.08m, mass 0.05kg), PyBullet primitive |
| Grasping | Constraint-based attachment (`p.createConstraint`), not a modeled gripper |
| Action space | `Box(-1, 1, shape=(8,))` — 7 joint velocities + 1 continuous gripper command |
| Observation space | `Box(shape=(27,))` — joint pos/vel (14), ee pos (3), peg pos (3), target pos (3), relative vector to peg-or-target depending on holding state (3), holding flag (1) |

**Why constraint-based grasping, not a modeled gripper:** the bundled KUKA iiwa has no gripper, and building/tuning a multi-finger gripper URDF from scratch is a substantial sub-project that isn't what this phase evaluates. The PS requires grasping to be "represented and evaluated as part of the reward/termination logic, not scripted" — satisfied because the constraint only forms when the *learned* policy chooses to close the gripper at the right time and proximity; nothing about *when* to grasp is scripted.

### Reward (staged)
```
Not holding:  reward = -distance(ee, peg)
Holding:      reward = -distance(peg, target) + 2.0   (holding bonus, per step)
Always:       reward -= 0.01 * sum(action^2)
On grasp:     reward += 1.0                             (one-time)
On premature release (>8cm from target): reward -= 3.0
On success:   reward += 20.0   (release within 5cm of target)
```

---

## 2. Two training runs, two distinct diagnosed failure modes

This phase went through two full training runs, each converging on a different degenerate strategy rather than the intended task. Both are documented here in detail, since diagnosing *why* a policy settles on a specific wrong behavior — and connecting that to a specific line in the reward function — is the actual RL-engineering content of this phase, arguably more instructive than a clean success would be.

### Run 1: grasp-attempt without retention
The policy converged on reliably closing the gripper near the peg (~98% of episodes attempted a grasp) but released it again almost immediately — 0% full task success, ~4% transient grasp, every episode running the full 300-step limit. **Diagnosis:** the reward gave a bonus for holding (+2/step) but no actual cost for releasing early — releasing just stopped future bonus accrual rather than being penalized. This was compounded by an over-tuned entropy coefficient (`ent_coef=0.01`): training logs showed policy action-noise (`std`) climbing steadily (1.05 → 1.6+) instead of shrinking, meaning exploration pressure was actively outweighing convergence.

**Fix applied:** added an explicit premature-release penalty (-3.0, triggered when releasing >8cm from the target), and reduced `ent_coef` to 0.003.

### Run 2 (post-fix): attempt-avoidance
The fix worked exactly as intended on its own terms — `std` stayed flat (~1.02–1.08) instead of climbing, and the shaping reward improved substantially (final eval reward -146.4 vs. the first run's plateau around -320 to -374 at a comparable step count; mean end-effector-to-peg distance dropped to 0.635m). **But grasp-attempt rate collapsed to 2%.** The policy learned to hover close to the peg — collecting steady, low-risk, dense distance-shaping reward — while almost entirely avoiding the higher-variance, penalty-exposed act of actually attempting a grasp. **Diagnosis:** this is a textbook reward-hacking pattern — introducing a real cost for one failure mode (premature release) made a *different*, previously-rare failure mode (never attempting at all) become the policy's new local optimum, since "get close but never risk grasping" is a lower-variance, still-rewarded strategy under the current shaping.

---

## 3. Final Results

Evaluated over 50 fresh randomized episodes, deterministic policy, post-fix checkpoint (`eval_pickplace.py`):

| Metric | Value |
|---|---|
| Task success rate | 0.0% |
| Grasp attempt rate | 2.0% |
| Grasp success rate (formed constraint) | 0.0% |
| Grasp failure rate (attempts that didn't form a grasp) | 100.0% |
| Mean completion time | 300.0 steps (max — no episode terminated early) |
| Mean final distance (ee to peg) | 0.635 m |

*For reference — pre-fix (Run 1) numbers: 0% task success, 98% grasp attempt rate, 4% grasp success rate, 0% retention, 1.118m mean final distance.*

**Honest overall verdict:** across both runs, the policy has consistently learned strong *proximity* behavior (closing distance to the relevant object) but has not learned to reliably complete the full grasp → hold → transport → release sequence. The specific way it fails changed between runs in a way that is fully explained by the specific reward term that changed — this is documented as a genuine, causally-understood limitation rather than an unexplained failure, and as evidence of real diagnostic work across iterations rather than a single blind attempt.

**Not pursued further given time constraints:** a third training run with a rebalanced reward (e.g., a smaller premature-release penalty, or an explicit small bonus for attempting a grasp regardless of outcome, to counteract the avoidance incentive) is the obvious next step and is a well-understood fix in principle — just not executed here due to the time budget for this submission.

---

## 4. Files in this submission

```
pick_place_env.py           # environment (extends reach_env.py's ReachEnv)
train_pickplace.py           # PPO training script (SB3, post-fix hyperparameters)
eval_pickplace.py            # evaluation over 50 episodes, required metrics
render_rollout.py            # generalized renderer (works across all phases)
models_pickplace/
  best_pickplace/best_model.zip
  vecnormalize_stats.pkl     # REQUIRED alongside the model
```

## 5. Known Limitations

- Grasp completion (success rate) is 0% across both training attempts — documented above as two distinct, diagnosed failure modes rather than one unexplained one.
- End-effector has no real gripper mechanics (fingers, friction-based grip) — grasping is a rigid constraint, a deliberate scope simplification stated up front (Section 1).
- Phases 3 and 4 both build on this environment/checkpoint, so this limitation propagates forward — see the Phase 3 README for how this was handled there.
