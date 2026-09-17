# Phase 3 — Obstacle-Aware Manipulation

## What was built

`obstacle_aware_env.py` extends `PickPlaceEnv` (Phase 2) per the PS's
integration requirement — same robot model, same action/observation
interface extended (not redefined), same grasp mechanism and staged
reward. Added on top:

- **3 randomized static box obstacles** per episode, placed via
  rejection sampling so they never block the peg/target positions
  outright (the task stays geometrically solvable every episode).
- **Real collision detection** via PyBullet's `getContactPoints`
  between the robot/peg and every obstacle — actual physical contact,
  not a proximity heuristic.
- **Collision penalty in the reward** (`-2.0` per step in contact), so
  a run that reaches the goal but repeatedly collides cannot get full
  credit, per the PS's explicit requirement.
- **Obstacle positions appended to the observation** (fixed-size, 3
  obstacles × 3 coords), so the policy has the information needed to
  route around them.

`train_obstacle.py` reuses Phase 2's PPO setup, entropy-coefficient
annealing, checkpointing, and telemetry callbacks, extended with
collision-rate/collision-count logging.

## Training result

Two training runs were made. Both diverged in the same way Phase 2's
Run 1 did — `std` climbing rather than settling — and the second run
diverged further than the first (peak `std` 2.41 vs. ~1.6), because
the collision penalty added a second source of large negative-reward
swings on top of the reaching/grasping task Phase 2 had not itself
fully solved. The entropy anneal used successfully in Phase 2 was not
sufficient here on its own; this reward combination needed its own
tuning (e.g., a lower starting `ent_coef`, or scaling/clipping the
collision penalty), which was not completed given the time available.

## Final evaluation (50 episodes, 3 obstacles, best checkpoint)

| Metric | Value |
|---|---|
| Task success rate | 0.00% |
| Collision rate (episodes with ≥1 collision) | 82.00% |
| Mean collision-steps per episode | 161.34 / 300 |
| Mean completion steps | 300.0 (no early success termination) |
| Mean end-effector path length | 7.06 m |
| Mean path efficiency (straight-line / actual, capped at 1.0) | 0.245 |

## Files included

|Model_pickplace
|-- best_obstacle
|-- vecnormalize_stats.pkl
|Render_rollout.py
|train_obstacle.py
|obstacle__aware.py
|evaluate_obstacle.py
|rolloout_obstacle.mp4

## Interpreting these numbers honestly

The 0% success rate is consistent with Phase 2's own policy never
reliably completing pick-and-place in its training budget — Phase 3
inherits that limitation by design (it extends the same reaching/
grasping mechanism). The 82% collision rate and 0.245 path efficiency
are not a "near miss" — with mean action `std` at 2.41, the
end-effector's motion is dominated by near-random exploration rather
than policy-directed movement, and a 7m mean path length in a
workspace roughly 1m across confirms this: the arm is thrashing, not
navigating. High collision incidence in a cluttered workspace is the
expected geometric consequence of that, not an independently
diagnosed obstacle-avoidance failure.

**What IS verified working:** the obstacle placement, collision
detection, and reward/logging mechanism itself. The 82%/161-step
numbers are the collision-tracking system correctly measuring a real
failure — the mechanism this phase needed to demonstrate is present
and functioning; the policy's ability to exploit it to avoid
collisions is what did not converge in the time available.

## Root cause (diagnosed, not yet fixed)

Same entropy/policy-std divergence pattern seen and partially
addressed in Phase 2, compounding here with:
1. A grasping/reaching foundation (from Phase 2) that had not itself
   converged to reliable success.
2. An added collision penalty increasing reward variance during early
   training, which the existing entropy anneal was not tuned to
   counteract in this budget.

## Path to completion (not executed, given time constraints)

- Lower the initial `ent_coef` (e.g., 0.003 instead of 0.01) so the
  anneal starts from a smaller exploration bonus.
- Scale or clip the collision penalty so a single contact step cannot
  produce as large a reward swing.
- Consider training Phase 3 only after Phase 2 converges on its own
  (rather than in parallel/independently), so obstacle-avoidance is
  layered onto an already-competent reach-and-grasp policy rather than
  learned simultaneously with it.
