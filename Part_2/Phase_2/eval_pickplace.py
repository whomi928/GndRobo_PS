"""
Evaluation script for Phase 2 (Pick-and-Place).

Runs the best checkpoint deterministically over N fresh, randomly
generated episodes and reports the PS-required metrics: success rate,
completion time, collision count (N/A here - no obstacles yet, that's
Phase 3), position error, and grasp-failure rate.

Run AFTER training finishes:
    !python /content/drive/MyDrive/ground_robotics_rl/eval_pickplace.py

Reports honestly even if success rate is 0 or near-0 - that's real
data for the technical report, not a failure to hide.
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_pickplace")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_pickplace", "best_model.zip")
VECNORM_PATH = os.path.join(MODELS_DIR, "vecnormalize_stats.pkl")

N_EVAL_EPISODES = 50

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from pick_place_env import PickPlaceEnv


def main():
    print(f"Loading best model from: {BEST_MODEL_PATH}")
    eval_env = make_vec_env(lambda: PickPlaceEnv(render_mode=None), n_envs=1)

    # Load the training run's normalization statistics rather than
    # fitting fresh ones - the policy was trained against THOSE
    # obs statistics, and evaluating without them (or with newly-fit
    # ones) silently biases every result in this script.
    if os.path.exists(VECNORM_PATH):
        eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"Loaded VecNormalize stats from: {VECNORM_PATH}")
    else:
        print("WARNING: no saved VecNormalize stats found - wrapping "
              "fresh (results will be less reliable). Expected at:",
              VECNORM_PATH)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO.load(BEST_MODEL_PATH, env=eval_env)

    successes = []
    completion_times = []
    final_distances = []       # ee/peg-to-target distance at episode end
    grasp_attempts_total = 0
    grasp_successes_total = 0
    min_dists = []

    obs = eval_env.reset()
    for ep in range(N_EVAL_EPISODES):
        done = False
        step_count = 0
        ep_grasp_attempted = False
        ep_grasp_succeeded = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            step_count += 1
            last_info = info[0]
            if last_info.get("grasp_attempted"):
                ep_grasp_attempted = True
            if last_info.get("grasp_succeeded"):
                ep_grasp_succeeded = True

        successes.append(bool(last_info.get("success", False)))
        completion_times.append(step_count)
        final_distances.append(float(last_info.get("distance", float("nan"))))
        min_dists.append(float(last_info.get("min_dist_this_ep", float("nan"))))
        grasp_attempts_total += int(ep_grasp_attempted)
        grasp_successes_total += int(ep_grasp_succeeded)

        if (ep + 1) % 10 == 0:
            print(f"  ...completed {ep + 1}/{N_EVAL_EPISODES} episodes")

    success_rate = np.mean(successes)
    grasp_attempt_rate = grasp_attempts_total / N_EVAL_EPISODES
    grasp_success_rate = grasp_successes_total / N_EVAL_EPISODES
    grasp_failure_rate = (
        (grasp_attempts_total - grasp_successes_total) / grasp_attempts_total
        if grasp_attempts_total > 0 else float("nan")
    )

    print("\n" + "=" * 60)
    print(f"PICK-AND-PLACE EVALUATION ({N_EVAL_EPISODES} episodes, deterministic)")
    print("=" * 60)
    print(f"Task success rate:         {success_rate:.1%}")
    print(f"Grasp attempt rate:        {grasp_attempt_rate:.1%}  (episodes where gripper was closed near the peg)")
    print(f"Grasp success rate:        {grasp_success_rate:.1%}  (episodes with a formed grasp constraint)")
    print(f"Grasp failure rate:        {grasp_failure_rate:.1%}  (attempts that did not form a grasp)")
    print(f"Mean completion time:      {np.mean(completion_times):.1f} steps (max {PickPlaceEnv().max_steps})")
    print(f"Mean final distance:       {np.nanmean(final_distances):.3f} m")
    print(f"Mean per-episode min dist: {np.nanmean(min_dists):.3f} m  (closest ee-peg approach)")
    print("=" * 60)
    print("\nNote: episodes that never terminate early (no success) run the")
    print("full max_steps by design - 'completion time' is only meaningful")
    print("relative to successful episodes if success_rate > 0.")


if __name__ == "__main__":
    main()
