

import os
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from peg_insert_env import PegInsertEnv
from evaluate_peginsert import evaluate as evaluate_standard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models_peginsert", "best_peginsert", "best_model")
VECNORM_PATH = os.path.join(SCRIPT_DIR, "models_peginsert", "vecnormalize_stats.pkl")


def evaluate_with_dynamics_override(peg_mass=None, peg_friction=None, joint_damping=None,
                                     n_episodes=50, n_obstacles=2):
    raw_env = DummyVecEnv([lambda: PegInsertEnv(render_mode=None, n_obstacles=n_obstacles)])
    norm_env = VecNormalize.load(VECNORM_PATH, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False
    model = PPO.load(MODEL_PATH)

    inner_env = norm_env.envs[0]

    def apply_overrides():
        if peg_mass is not None:
            p.changeDynamics(inner_env._peg_id, -1, mass=peg_mass,
                              physicsClientId=inner_env._client)
        if peg_friction is not None:
            p.changeDynamics(inner_env._peg_id, -1, lateralFriction=peg_friction,
                              physicsClientId=inner_env._client)
        if joint_damping is not None:
            for j in inner_env._controlled_joints:
                p.changeDynamics(inner_env._robot_id, j, jointDamping=joint_damping,
                                  physicsClientId=inner_env._client)

    successes = 0
    ever_held = 0
    any_collision_episodes = 0
    max_depth_progress = []

    for ep in range(n_episodes):
        obs = norm_env.reset()
        apply_overrides()  # AFTER reset, since reset() rebuilds/repositions bodies
        done = False
        held_this_ep = False
        collided_this_ep = False
        best_depth = -np.inf
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
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
        "grasp_attempt_rate": ever_held / n_episodes,  # approximated as hold-rate here;
                                                          # see evaluate_peginsert.py for the
                                                          # more precise grasp_attempted-based metric
        "any_collision_rate": any_collision_episodes / n_episodes,
        "mean_best_depth_progress": float(np.mean(max_depth_progress)),
    }


def main(n_episodes=30):
    results = {}

    print("=" * 70)
    print("PHASE 5a: STANDARD GENERALIZATION (unseen obstacle counts)")
    print("=" * 70)
    for n_obs in [0, 1, 2, 3, 4]:
        tag = f"n_obstacles={n_obs}" + (" (TRAINED ON THIS)" if n_obs == 2 else " (UNSEEN COUNT)")
        print(f"\n--- {tag} ---")
        results[f"obstacles_{n_obs}"] = evaluate_standard(n_episodes=n_episodes, n_obstacles=n_obs)

    print("\n" + "=" * 70)
    print("PHASE 5b: DYNAMICS GENERALIZATION (unseen physical parameters)")
    print("=" * 70)

    dynamics_conditions = {
        "baseline (no override)": {},
        "heavier_peg (2x mass)": {"peg_mass": 0.10},      # trained peg mass was 0.05
        "lighter_peg (0.5x mass)": {"peg_mass": 0.025},
        "low_friction_peg": {"peg_friction": 0.2},
        "high_friction_peg": {"peg_friction": 1.5},
        "high_joint_damping": {"joint_damping": 0.5},
    }
    for name, overrides in dynamics_conditions.items():
        print(f"\n--- {name} ---")
        r = evaluate_with_dynamics_override(n_episodes=n_episodes, **overrides)
        results[f"dynamics_{name}"] = r
        print(f"Success rate: {r['success_rate']:.2%} | Hold rate: {r['grasp_attempt_rate']:.2%} | "
              f"Collision rate: {r['any_collision_rate']:.2%} | "
              f"Mean best depth progress: {r['mean_best_depth_progress']:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (copy into report)")
    print("=" * 70)
    print(f"{'Condition':<35}{'Success':>10}{'Hold/Grasp':>12}{'Collision':>12}")
    for key, r in results.items():
        grasp_key = "grasp_attempt_rate" if "grasp_attempt_rate" in r else "hold_rate"
        collision_key = "any_collision_rate" if "any_collision_rate" in r else "collision_rate"
        print(f"{key:<35}{r['success_rate']:>9.1%}{r.get(grasp_key, 0):>11.1%}{r.get(collision_key, 0):>11.1%}")

    return results


if __name__ == "__main__":
    main()
