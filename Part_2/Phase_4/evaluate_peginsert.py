import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from peg_insert_env import PegInsertEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model_path=None, vecnorm_path=None, n_episodes=50, n_obstacles=2):
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, "models_peginsert", "best_peginsert", "best_model")
    if vecnorm_path is None:
        vecnorm_path = os.path.join(SCRIPT_DIR, "models_peginsert", "vecnormalize_stats.pkl")

    raw_env = DummyVecEnv([lambda: PegInsertEnv(render_mode=None, n_obstacles=n_obstacles)])
    norm_env = VecNormalize.load(vecnorm_path, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(model_path)

    successes = 0
    ever_held = 0
    ever_grasp_attempted = 0
    grasp_succeeded_count = 0
    tolerance_ever_violated_during_insertion = 0
    board_collision_episodes = 0
    any_collision_episodes = 0
    max_depth_progress = []
    min_xy_offset = []
    min_tilt = []
    completion_times = []
    completion_steps = []

    for ep in range(n_episodes):
        obs = norm_env.reset()
        start_time = time.time()
        done = False
        steps = 0
        held_this_ep = False
        grasp_attempted_this_ep = False
        grasp_succeeded_this_ep = False
        tol_violated_this_ep = False
        board_collision_this_ep = False
        any_collision_this_ep = False
        best_depth = -np.inf
        best_xy_offset = np.inf
        best_tilt = np.inf
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = norm_env.step(action)
            info = info_list[0]
            steps += 1

            if info.get("is_holding", False):
                held_this_ep = True
            if info.get("grasp_attempted", False):
                grasp_attempted_this_ep = True
            if info.get("grasp_succeeded", False):
                grasp_succeeded_this_ep = True
            if info.get("tolerance_violated_during_insertion", False):
                tol_violated_this_ep = True
            if info.get("board_collision", False):
                board_collision_this_ep = True
            if info.get("collision", False):
                any_collision_this_ep = True

            depth = info.get("insertion_depth_progress", -np.inf)
            if depth > best_depth:
                best_depth = depth
            xy_off = info.get("xy_offset_from_hole", np.inf)
            if xy_off < best_xy_offset:
                best_xy_offset = xy_off
            tilt = info.get("peg_tilt_rad", np.inf)
            if tilt < best_tilt:
                best_tilt = tilt

            success = info.get("insertion_success", False)

        completion_times.append(time.time() - start_time)
        completion_steps.append(steps)
        max_depth_progress.append(best_depth)
        min_xy_offset.append(best_xy_offset)
        min_tilt.append(best_tilt)
        if held_this_ep:
            ever_held += 1
        if grasp_attempted_this_ep:
            ever_grasp_attempted += 1
        if grasp_succeeded_this_ep:
            grasp_succeeded_count += 1
        if tol_violated_this_ep:
            tolerance_ever_violated_during_insertion += 1
        if board_collision_this_ep:
            board_collision_episodes += 1
        if any_collision_this_ep:
            any_collision_episodes += 1
        if success:
            successes += 1

    print(f"--- Phase 4 Evaluation over {n_episodes} episodes ({n_obstacles} obstacles) ---")
    print(f"Full insertion success rate:            {successes / n_episodes:.2%}")
    print(f"Grasp attempt rate:                      {ever_grasp_attempted / n_episodes:.2%}")
    print(f"Grasp succeeded rate:                    {grasp_succeeded_count / n_episodes:.2%}")
    print(f"Episodes that ever held the peg:         {ever_held / n_episodes:.2%}")
    print(f"Episodes with a board (frame) collision:  {board_collision_episodes / n_episodes:.2%}")
    print(f"Episodes with any collision:              {any_collision_episodes / n_episodes:.2%}")
    print(f"Episodes with tolerance violated in ins.: {tolerance_ever_violated_during_insertion / n_episodes:.2%}")
    print(f"Mean best insertion depth progress:      {np.mean(max_depth_progress):.6f}")
    print(f"Mean best (closest) xy offset from hole: {np.mean(min_xy_offset):.4f} m")
    print(f"Mean best (closest) peg tilt:             {np.mean(min_tilt):.4f} rad")
    print(f"Mean completion time (wall-clock):        {np.mean(completion_times):.4f} s")
    print(f"Mean completion steps:                    {np.mean(completion_steps):.1f}")

    return {
        "success_rate": successes / n_episodes,
        "grasp_attempt_rate": ever_grasp_attempted / n_episodes,
        "grasp_succeeded_rate": grasp_succeeded_count / n_episodes,
        "hold_rate": ever_held / n_episodes,
        "board_collision_rate": board_collision_episodes / n_episodes,
        "any_collision_rate": any_collision_episodes / n_episodes,
        "mean_best_depth_progress": float(np.mean(max_depth_progress)),
        "mean_best_xy_offset": float(np.mean(min_xy_offset)),
    }


if __name__ == "__main__":
    evaluate()
