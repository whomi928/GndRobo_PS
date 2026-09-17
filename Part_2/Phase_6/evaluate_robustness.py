"""
Phase 6 — Robustness / Stress Test evaluation.

NO TRAINING. Same trained Phase 4 checkpoint, same as Phase 5 - but
here we inject noise/perturbations ONLY AT EVALUATION TIME (never
seen during training), and SWEEP the magnitude across several levels,
per the PS's explicit requirement to characterize DEGRADATION rather
than report a single pass/fail number.

THREE KINDS OF DISTURBANCE, PER THE PS
------------------------------------------
1. Observation noise: Gaussian noise added to the (already-normalized)
   observation vector before the policy sees it - simulates imperfect
   sensing/state estimation.
2. Action/control noise: Gaussian noise added to the policy's chosen
   action before it's applied - simulates imperfect actuation.
3. Small perturbations to the peg/environment: a small random external
   force applied to the peg partway through each episode - simulates
   an external bump/disturbance.

Each is swept independently (holding the other two at zero) across a
few magnitude levels, so the report can show which disturbance type
the policy is most sensitive to - a compound sweep (all three at once)
would confound that.
"""

import os
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from peg_insert_env import PegInsertEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models_peginsert", "best_peginsert", "best_model")
VECNORM_PATH = os.path.join(SCRIPT_DIR, "models_peginsert", "vecnormalize_stats.pkl")


def evaluate_with_disturbance(obs_noise_std=0.0, action_noise_std=0.0,
                               perturbation_force_mag=0.0,
                               n_episodes=30, n_obstacles=2, seed_base=0):
    raw_env = DummyVecEnv([lambda: PegInsertEnv(render_mode=None, n_obstacles=n_obstacles)])
    norm_env = VecNormalize.load(VECNORM_PATH, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False
    model = PPO.load(MODEL_PATH)
    inner_env = norm_env.envs[0]
    rng = np.random.default_rng(seed_base)

    successes = 0
    ever_held = 0
    any_collision_episodes = 0
    max_depth_progress = []

    for ep in range(n_episodes):
        obs = norm_env.reset()
        done = False
        steps = 0
        held_this_ep = False
        collided_this_ep = False
        best_depth = -np.inf
        success = False
        perturb_at_step = rng.integers(20, 150) if perturbation_force_mag > 0 else -1

        while not done:
            steps += 1

            obs_for_policy = obs
            if obs_noise_std > 0:
                obs_for_policy = obs + rng.normal(0, obs_noise_std, size=obs.shape).astype(np.float32)

            action, _ = model.predict(obs_for_policy, deterministic=True)
            if action_noise_std > 0:
                action = action + rng.normal(0, action_noise_std, size=action.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)

            if steps == perturb_at_step:
                force = rng.normal(0, 1, size=3)
                force = force / (np.linalg.norm(force) + 1e-8) * perturbation_force_mag
                p.applyExternalForce(
                    inner_env._peg_id, -1, force.tolist(), [0, 0, 0], p.WORLD_FRAME,
                    physicsClientId=inner_env._client,
                )

            obs, reward, done, info_list = norm_env.step(action)
            info = info_list[0]

            if info.get("is_holding", False):
                held_this_ep = True
            if info.get("collision", False):
                collided_this_ep = True
            depth = info.get("insertion_depth_progress", -np.inf)
            if depth > best_depth:
                best_depth = depth
            success = info.get("insertion_success", False)

        max_depth_progress.append(best_depth)
        if held_this_ep:
            ever_held += 1
        if collided_this_ep:
            any_collision_episodes += 1
        if success:
            successes += 1

    return {
        "success_rate": successes / n_episodes,
        "hold_rate": ever_held / n_episodes,
        "any_collision_rate": any_collision_episodes / n_episodes,
        "mean_best_depth_progress": float(np.mean(max_depth_progress)),
    }


def main(n_episodes=30):
    results = {}

    print("=" * 70)
    print("PHASE 6a: OBSERVATION NOISE SWEEP")
    print("=" * 70)
    for std in [0.0, 0.02, 0.05, 0.10, 0.20]:
        print(f"\n--- obs_noise_std = {std} ---")
        r = evaluate_with_disturbance(obs_noise_std=std, n_episodes=n_episodes)
        results[f"obs_noise_{std}"] = r
        print(f"Success: {r['success_rate']:.2%} | Hold: {r['hold_rate']:.2%} | "
              f"Collision: {r['any_collision_rate']:.2%}")

    print("\n" + "=" * 70)
    print("PHASE 6b: ACTION/CONTROL NOISE SWEEP")
    print("=" * 70)
    for std in [0.0, 0.02, 0.05, 0.10, 0.20]:
        print(f"\n--- action_noise_std = {std} ---")
        r = evaluate_with_disturbance(action_noise_std=std, n_episodes=n_episodes)
        results[f"action_noise_{std}"] = r
        print(f"Success: {r['success_rate']:.2%} | Hold: {r['hold_rate']:.2%} | "
              f"Collision: {r['any_collision_rate']:.2%}")

    print("\n" + "=" * 70)
    print("PHASE 6c: PEG PERTURBATION FORCE SWEEP")
    print("=" * 70)
    for mag in [0.0, 0.5, 1.0, 2.0, 4.0]:
        print(f"\n--- perturbation_force_mag = {mag} N ---")
        r = evaluate_with_disturbance(perturbation_force_mag=mag, n_episodes=n_episodes)
        results[f"perturb_{mag}"] = r
        print(f"Success: {r['success_rate']:.2%} | Hold: {r['hold_rate']:.2%} | "
              f"Collision: {r['any_collision_rate']:.2%}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (copy into report)")
    print("=" * 70)
    print(f"{'Condition':<25}{'Success':>10}{'Hold':>10}{'Collision':>12}")
    for key, r in results.items():
        print(f"{key:<25}{r['success_rate']:>9.1%}{r['hold_rate']:>9.1%}{r['any_collision_rate']:>11.1%}")

    print("\nNOTE: with a 0% baseline success rate (see Phase 4 results), this sweep will "
          "very likely show success_rate=0% at EVERY noise level - that is an expected, "
          "honest outcome given Phase 4's own finding (grasp never completes), not a bug "
          "in this script. What's still informative here: whether HOLD rate and COLLISION "
          "rate degrade as disturbance increases, which tells you whether the arm's motion "
          "itself (independent of the unsolved grasp problem) is robust or brittle to noise.")

    return results


if __name__ == "__main__":
    main()
