import argparse
import numpy as np
import pybullet as p
import imageio
from reach_env import ReachEnv


def render_rollout(model_path=None, vecnorm_path=None, out_path="rollout.mp4",
                    n_steps=150, fps=30):
    env = ReachEnv(render_mode=None)  # still DIRECT mode - we render manually below
    obs, info = env.reset()
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
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a trained SB3 model .zip. Omit for random-action rollout.")
    parser.add_argument("--vecnorm", type=str, default=None,
                         help="Path to saved VecNormalize stats (.pkl). Required when "
                              "--model is set, since the policy expects normalized inputs.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    args = parser.parse_args()

    render_rollout(model_path=args.model, vecnorm_path=args.vecnorm,
                    out_path=args.out, n_steps=args.steps)
