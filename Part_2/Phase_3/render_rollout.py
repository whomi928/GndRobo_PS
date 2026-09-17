"""
Rollout renderer — records a video of the robot in the environment,
since Colab has no display for PyBullet's live GUI window.

HOW THIS WORKS
--------------
PyBullet's GUI mode (p.GUI) opens an actual OS window - that only works
on a machine with a display attached (your own laptop, not a headless
Colab VM). Instead, we stay in DIRECT (headless) mode and manually grab
camera images every step using p.getCameraImage(), pointed at a fixed
virtual camera looking at the robot. Stack enough of those images in
order and you have a video - same idea as a flipbook.

USAGE
-----
Random-action rollout (before any training - useful to sanity check the
world/camera setup):
    python render_rollout.py

Rollout using a trained model:
    python render_rollout.py --model ./models/best_reach/best_model.zip
"""

"""
Rollout renderer — records a video of the robot in ANY of this
project's environments, since Colab has no display for PyBullet's live
GUI window.

GENERALIZED ACROSS PHASES
----------------------------
Rather than one renderer per phase (reach/pickplace/obstacle/peginsert),
this loads the target environment CLASS dynamically by name, so the
same script works for every phase - you tell it which module/class to
import and it does the rest. This mirrors the PS's own requirement
that Part 2 share consistent infrastructure across phases rather than
duplicating it per phase.

USAGE
-----
    python render_rollout.py --env-module reach_env --env-class ReachEnv
    python render_rollout.py --env-module pick_place_env --env-class PickPlaceEnv \\
        --model ./models_pickplace/best_pickplace/best_model.zip \\
        --vecnorm ./models_pickplace/vecnormalize_stats.pkl
    python render_rollout.py --env-module obstacle_aware_env --env-class ObstacleAwareEnv \\
        --env-kwargs '{"n_obstacles": 3}' \\
        --model ./models_obstacle/best_obstacle/best_model.zip \\
        --vecnorm ./models_obstacle/vecnormalize_stats.pkl
"""

import argparse
import importlib
import json
import numpy as np
import pybullet as p
import imageio


def load_env_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def render_rollout(env_module, env_class, env_kwargs=None,
                    model_path=None, vecnorm_path=None,
                    out_path="rollout.mp4", n_steps=150, fps=30):
    EnvClass = load_env_class(env_module, env_class)
    env_kwargs = env_kwargs or {}
    env = EnvClass(render_mode=None, **env_kwargs)  # still DIRECT mode - rendered manually below
    obs, info = env.reset()

    # --- Virtual camera setup ---
    # Positioned to look at the robot from a 3/4 angle, similar to a
    # typical "watching the robot work" viewpoint. Tune distance/yaw/
    # pitch/target if the robot is out of frame once you see the result.
    width, height = 640, 480
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.3],
        distance=1.5,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=5.0
    )

    model = None
    obs_rms = None
    if model_path is not None:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"Rendering with trained policy from: {model_path}")

        if vecnorm_path is not None:
            # IMPORTANT: the policy was trained on NORMALIZED
            # observations. Without applying the same normalization
            # here, model.predict() gets inputs on the wrong scale and
            # the trained policy will look no better than random -
            # same issue fixed in evaluate_reach.py.
            import pickle
            with open(vecnorm_path, "rb") as f:
                vecnorm = pickle.load(f)
            obs_rms = vecnorm.obs_rms
            print(f"Loaded normalization stats from: {vecnorm_path}")
        else:
            print("WARNING: rendering a trained model WITHOUT normalization stats - "
                  "the policy will likely behave poorly. Pass --vecnorm to fix this.")
    else:
        print("No model given - rendering RANDOM actions (sanity check only, "
              "not a trained policy).")

    def normalize(o):
        if obs_rms is None:
            return o
        return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)

    frames = []
    for step in range(n_steps):
        if model is not None:
            action, _ = model.predict(normalize(obs), deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        img = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,  # software renderer - works headless, no GPU needed
            physicsClientId=env._client,
        )
        rgb_array = np.reshape(img[2], (height, width, 4))[:, :, :3]  # drop alpha channel
        frames.append(rgb_array.astype(np.uint8))

        if terminated or truncated:
            print(f"Episode ended at step {step} | success={info['success']} | "
                  f"final distance={info['distance']:.3f}")
            obs, info = env.reset()

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved video: {out_path} ({len(frames)} frames at {fps} fps)")
    env.close()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-module", type=str, required=True,
                         help="Python module containing the environment class, "
                              "e.g. 'obstacle_aware_env' (no .py extension).")
    parser.add_argument("--env-class", type=str, required=True,
                         help="Environment class name within that module, "
                              "e.g. 'ObstacleAwareEnv'.")
    parser.add_argument("--env-kwargs", type=str, default=None,
                         help="Optional JSON string of extra kwargs for the env "
                              "constructor, e.g. '{\"n_obstacles\": 3}'.")
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a trained SB3 model .zip. Omit for random-action rollout.")
    parser.add_argument("--vecnorm", type=str, default=None,
                         help="Path to saved VecNormalize stats (.pkl). Required when "
                              "--model is set, since the policy expects normalized inputs.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    args = parser.parse_args()

    kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}
    render_rollout(env_module=args.env_module, env_class=args.env_class, env_kwargs=kwargs,
                    model_path=args.model, vecnorm_path=args.vecnorm,
                    out_path=args.out, n_steps=args.steps)
