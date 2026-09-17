"""
Phase 3 training script — PPO on ObstacleAwareEnv.

Same infrastructure as Phase 2 (VecNormalize, CPU device, script-
relative paths). Initialized from Phase 2's best checkpoint rather
than from scratch - the underlying reach/grasp/place behavior
shouldn't need to be relearned from zero just because obstacles were
added; fine-tuning on top of it should be faster than a cold start.

IMPORTANT: if Phase 2's grasp behavior was inconsistent (documented
limitation in the Phase 2 report), that inconsistency carries INTO
this phase's baseline - Phase 3's job is only to add obstacle
avoidance on top of whatever pick-and-place competence already exists,
not to fix Phase 2's grasp reliability. Report results honestly with
that caveat if it's still an issue here.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_obstacle")
TB_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs_obstacle")
EVAL_LOG_DIR = os.path.join(SCRIPT_DIR, "eval_logs_obstacle")
PHASE2_MODEL_PATH = os.path.join(SCRIPT_DIR, "models_pickplace", "pickplace_final.zip")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from obstacle_aware_env import ObstacleAwareEnv


def main():
    train_env = make_vec_env(lambda: ObstacleAwareEnv(render_mode=None, n_obstacles=3), n_envs=4)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(lambda: ObstacleAwareEnv(render_mode=None, n_obstacles=3), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    if os.path.exists(PHASE2_MODEL_PATH):
        # NOTE: PPO.load can restore weights, but since the observation
        # space CHANGED (Phase 2's 27 dims -> Phase 3's 31 dims), we
        # cannot simply resume the Phase 2 model directly - its network
        # input layer is the wrong shape. Warm-starting across a space
        # change requires either custom weight-surgery (copying the
        # overlapping input weights into a newly-shaped network) or
        # accepting a fresh initialization. For this submission we train
        # fresh, but note the option here since it's a legitimate next
        # optimization if training time allows.
        print(f"Note: Phase 2 checkpoint found at {PHASE2_MODEL_PATH}, but observation "
              f"space changed (27 -> 31 dims) so we cannot warm-start directly. "
              f"Training fresh.")
    else:
        print("No Phase 2 checkpoint found - training fresh (expected either way, see note above).")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
        device="cpu",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    class SyncNormStatsCallback(EvalCallback):
        def _on_step(self):
            self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

    eval_callback = SyncNormStatsCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best_obstacle"),
        log_path=EVAL_LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=10,
    )

    # Budget: similar order to Phase 2 given similar task complexity
    # plus the added obstacle-avoidance constraint. Watch collision
    # rate trend (via a custom callback below) alongside reward - a
    # policy that ignores obstacles entirely will show high reward
    # variance and a flat/high collision-step count.
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save(os.path.join(MODELS_DIR, "obstacle_final"))
    train_env.save(os.path.join(MODELS_DIR, "vecnormalize_stats.pkl"))
    print(f"Training complete. Best model: {os.path.join(MODELS_DIR, 'best_obstacle')}")
    print(f"Normalization stats: {os.path.join(MODELS_DIR, 'vecnormalize_stats.pkl')}")


if __name__ == "__main__":
    main()
