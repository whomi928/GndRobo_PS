# Part 2 — Manipulator RL — Phase 1: Reaching

**Inter IIT Tech Meet 15.0 — Prepathon PS: Ground Robotics**

This covers Phase 1 of Part 2: a learned RL policy that drives a robotic arm's end-effector to a randomly placed target, from a randomized initial joint configuration, without using inverse kinematics.

---

## 1. Environment

| Component | Choice |
|---|---|
| Simulator | PyBullet (DIRECT/headless mode for training) |
| Robot | KUKA iiwa (7-DOF), loaded from `pybullet_data`'s bundled URDF |
| Control mode | Joint velocity control, action-repeated over 8 physics substeps per env step |
| RL algorithm | PPO (Stable-Baselines3), MLP policy |
| Observation normalization | `VecNormalize` (running mean/std, clipped to ±10) |

**Why velocity control, not position or IK:** the PS requires a *learned* policy, not an analytical inverse-kinematics solution — IK would make the "learning" trivial/absent. Velocity control also keeps the environment interface consistent with what later phases (grasping, obstacle avoidance, insertion) will need, per the PS's requirement that the environment stay consistent across Part 2's phases rather than being redefined per phase.

**Robot note:** the environment auto-detects the loaded robot's revolute joints and end-effector link via `p.getJointInfo()` at startup, rather than hardcoding indices for a specific URDF. This was a deliberate design choice after discovering mid-development that `pybullet_data` didn't actually ship a UR5 URDF in this environment and silently fell back to a 7-DOF KUKA iiwa — hardcoded 6-DOF indices from an assumed UR5 caused a crash that auto-detection made structurally impossible to hit again.

### Action space
`Box(-1, 1, shape=(7,))` — normalized joint velocity command per joint, scaled to a real velocity limit (1.0 rad/s) inside the environment.

### Observation space
`Box(shape=(23,))` — joint positions (7), joint velocities (7), end-effector position (3), target position (3), and the relative vector from end-effector to target (3). The relative-vector term was added deliberately: it's redundant information (derivable from the other two), but giving the policy this difference directly — rather than making it learn to compute the subtraction itself — is a standard piece of observation engineering that measurably eases learning for distance-based tasks.

### Reward
```
reward = -distance_to_target - 0.01 * sum(action^2) + (10.0 if success else 0.0)
success := distance_to_target < 0.03m
```
Dense distance-based shaping (not a sparse success-only reward) was necessary — a purely sparse reward gives almost no gradient signal early in training, since random exploration essentially never stumbles into a 3cm-tolerance success by chance in a 7-DOF space. The small action-magnitude penalty discourages wasteful/jerky motion.

---

## 2. Training

- Algorithm: PPO, `MlpPolicy`, `device="cpu"` (MLP policies do not benefit from GPU; SB3 itself warns against GPU for non-CNN policies)
- 4 parallel vectorized environments
- [FILL IN: total_timesteps actually used for the final successful run]
- Hyperparameters: default PPO learning rate (3e-4), `n_steps=1024`, `batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.005`

---

## 3. Results

Evaluated over 50 fresh randomized episodes (`evaluate_reach.py`):

| Metric | Value |
|---|---|
| Success rate | [FILL IN]% |
| Mean final position error | [FILL IN] m |
| Mean episode completion time | [FILL IN] s |
| Invalid-state occurrences | Not yet instrumented — see Known Limitations |

Video: `trained_rollout.mp4` — trained policy driving the end-effector toward randomized targets.

---

## 4. Debugging Notes (approach & reasoning)

Documented here since these reflect genuine engineering work, not a black box that "just worked":

1. **Wrong robot assumed (UR5 → actually KUKA iiwa).** `pybullet_data` didn't ship a UR5 URDF in this environment; the code silently fell back to a 7-DOF KUKA. Hardcoded 6-DOF joint/link indices crashed with `TypeError: 'NoneType' object is not subscriptable` on `getLinkState()`. Fixed by querying `p.getJointInfo()` at startup to auto-detect controllable (revolute) joints and infer the end-effector link, rather than hardcoding indices for an assumed URDF.

2. **Workspace/target mismatch.** The initial target-sampling box was tuned blind. Empirically sampled the arm's actual reachable envelope (extreme random joint configs → observed end-effector range) and resized the target box to sit comfortably inside it.

3. **Visually jittery rollout under random actions.** Fresh independent random actions every physics substep (240Hz) produced high-frequency, low-amplitude shaking with no real motion. Not a bug — root cause was calling a new random action every single physics tick with no temporal correlation. Added `action_repeat` (hold one action across 8 physics substeps) — standard RL practice that also improves motion smoothness and often training stability.

4. **Policy plateaued at 0% success despite "healthy-looking" training internals.** Across 300k timesteps, `ep_len_mean` stayed pinned at the max (200) — literally zero episodes ever succeeded — while `explained_variance` climbed toward 0.99 and `value_loss` fell smoothly. This combination (confident, well-fit critic; zero actual task progress; policy action-noise `std` never shrinking from its ~1.0 initialization) pointed to the reward signal itself being decoupled from real robot behavior, not an optimization/hyperparameter problem. Root cause: `p.getLinkState()` does not recompute forward kinematics by default — the end-effector position (and therefore the reward) could be stale relative to the robot's true post-step state. Fixed by passing `computeForwardKinematics=True`. This is a known PyBullet pitfall that is easy to miss because the environment runs without error either way — it fails silently in the *learning signal*, not the code.

5. **Unnormalized, mixed-scale observations.** Joint angles (~radians, ±6), positions (~meters, ±1), and velocities all fed the network on different scales. Added `VecNormalize` (running mean/std normalization) — standard practice for continuous control, used throughout SB3's own reference baselines — and saved its statistics (`vecnormalize_stats.pkl`) alongside the trained model, since evaluation/inference must apply the identical normalization the policy was trained under.

---

## 5. Known Limitations

- **Invalid/singular-state detection is not yet real.** The PS asks this to be tracked as an evaluation metric; currently only a placeholder exists. A defensible proxy (counting joint-limit violations, or a near-collinearity check between consecutive joints as a singularity heuristic) has been identified but not implemented in this submission.
- End-effector is currently the last arm link with no gripper attached — appropriate for pure reaching, but will need a proper gripper link once Phase 2 (pick-and-place) is built on top of this environment.
- Reward/target-box tuning was done empirically against one specific robot (KUKA iiwa fallback); values may need revisiting if a different URDF is used.

---

## 6. Files in this submission

```
reach_env.py          # environment (Gymnasium-style)
train_reach.py         # PPO training script (SB3)
evaluate_reach.py      # evaluation over 50 episodes, required metrics
render_rollout.py      # headless rollout-to-video renderer (Colab has no display)
models/
  best_reach/best_model.zip
  vecnormalize_stats.pkl   # REQUIRED alongside the model - policy expects normalized inputs
trained_rollout.mp4
```
