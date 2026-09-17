import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from obstacle_aware_env import ObstacleAwareEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model_path=None, vecnorm_path=None, n_episodes=50, n_obstacles=3):
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, "models_obstacle", "best_obstacle", "best_model")
    if vecnorm_path is None:
        vecnorm_path = os.path.join(SCRIPT_DIR, "models_obstacle", "vecnormalize_stats.pkl")

    raw_env = DummyVecEnv([lambda: ObstacleAwareEnv(render_mode=None, n_obstacles=n_obstacles)])
    norm_env = VecNormalize.load(vecnorm_path, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(model_path)

    successes = 0
    collision_step_counts = []
    completion_times = []
    completion_steps = []
    path_lengths = []       # actual distance traveled by end-effector
    straight_line_dists = []  # ee start-to-target straight-line distance, for path efficiency

    for ep in range(n_episodes):
        obs = norm_env.reset()
        start_time = time.time()
        done = False
        steps = 0
        prev_ee_pos = None
        start_ee_pos = norm_env.envs[0]._get_ee_position()
        target_pos = norm_env.envs[0]._target_pos
        straight_line_dist = float(np.linalg.norm(target_pos - start_ee_pos))
        path_length = 0.0
        success = False
        collision_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = norm_env.step(action)
            info = info_list[0]
            steps += 1

            ee_pos = norm_env.envs[0]._get_ee_position()
            if prev_ee_pos is not None:
                path_length += float(np.linalg.norm(ee_pos - prev_ee_pos))
            prev_ee_pos = ee_pos

            if info.get("collision", False):
                collision_steps += 1
            success = info.get("success", False)

        completion_times.append(time.time() - start_time)
        completion_steps.append(steps)
        collision_step_counts.append(collision_steps)
        path_lengths.append(path_length)
        straight_line_dists.append(straight_line_dist)
        if success:
            successes += 1

    success_rate = successes / n_episodes
    collision_rate = np.mean([c > 0 for c in collision_step_counts])  # fraction of EPISODES with >=1 collision
    mean_collision_steps = np.mean(collision_step_counts)  # avg collision-steps per episode (severity, not just occurrence)
    efficiencies = [
        min(sl / pl, 1.0) for sl, pl in zip(straight_line_dists, path_lengths) if pl > 1e-6
    ]

    print(f"--- Phase 3 Evaluation over {n_episodes} episodes ({n_obstacles} obstacles) ---")
    print(f"Task success rate: {success_rate:.2%}")
    print(f"Collision rate (episodes with >=1 collision): {collision_rate:.2%}")
    print(f"Mean collision-steps per episode: {mean_collision_steps:.2f}")
    print(f"Mean completion time (wall-clock): {np.mean(completion_times):.4f} s")
    print(f"Mean completion steps: {np.mean(completion_steps):.1f}")
    print(f"Mean end-effector path length: {np.mean(path_lengths):.4f} m")
    print(f"Mean path efficiency (straight-line / actual, capped at 1.0): "
          f"{np.mean(efficiencies):.4f}")


if __name__ == "__main__":
    evaluate()
