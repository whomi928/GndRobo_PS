import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from reach_env import ReachEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model_path=None, vecnorm_path=None, n_episodes=50):
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, "models", "best_reach", "best_model")
    if vecnorm_path is None:
        vecnorm_path = os.path.join(SCRIPT_DIR, "models", "vecnormalize_stats.pkl")

    raw_env = DummyVecEnv([lambda: ReachEnv(render_mode=None)])
    norm_env = VecNormalize.load(vecnorm_path, raw_env)
    norm_env.training = False    # freeze running stats - don't keep updating from eval data
    norm_env.norm_reward = False  # we want to read real (unnormalized) rewards/distances

    model = PPO.load(model_path)

    successes = 0
    final_errors = []
    completion_times = []
    invalid_state_count = 0  

    for ep in range(n_episodes):
        obs = norm_env.reset()
        start = time.time()
        done = False
        final_distance = None
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = norm_env.step(action)
            info = info_list[0]
            final_distance = info["distance"]
            success = info["success"]

        completion_times.append(time.time() - start)
        final_errors.append(final_distance)
        if success:
            successes += 1

    print(f"--- Phase 1 Evaluation over {n_episodes} episodes ---")
    print(f"Success rate: {successes / n_episodes:.2%}")
    print(f"Mean final position error: {np.mean(final_errors):.4f} m")
    print(f"Mean episode completion time (wall-clock): {np.mean(completion_times):.4f} s")
    print(f"Invalid-state occurrences: {invalid_state_count}  "
          f"(NOTE: wire up real detection in reach_env.py before reporting this number)")


if __name__ == "__main__":
    evaluate()
