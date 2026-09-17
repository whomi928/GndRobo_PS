import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_pickplace")
TB_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs_pickplace")
EVAL_LOG_DIR = os.path.join(SCRIPT_DIR, "eval_logs_pickplace")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from pick_place_env import PickPlaceEnv


def main():
    train_env = make_vec_env(lambda: PickPlaceEnv(render_mode=None), n_envs=4)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(lambda: PickPlaceEnv(render_mode=None), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

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
        ent_coef=0.003,
    )

    class SyncNormStatsCallback(EvalCallback):
        def _on_step(self):
            self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

    eval_callback = SyncNormStatsCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best_pickplace"),
        log_path=EVAL_LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=1_500_000, callback=eval_callback)

    model.save(os.path.join(MODELS_DIR, "pickplace_final"))
    train_env.save(os.path.join(MODELS_DIR, "vecnormalize_stats.pkl"))
    print(f"Training complete. Best model: {os.path.join(MODELS_DIR, 'best_pickplace')}")
    print(f"Normalization stats: {os.path.join(MODELS_DIR, 'vecnormalize_stats.pkl')}")


if __name__ == "__main__":
    main()
