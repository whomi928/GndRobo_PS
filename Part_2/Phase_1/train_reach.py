from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from reach_env import ReachEnv


def main():
    train_env = make_vec_env(lambda: ReachEnv(render_mode=None), n_envs=4)

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(lambda: ReachEnv(render_mode=None), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        policy="MlpPolicy",         
        verbose=1,
        tensorboard_log="./tb_logs",
        device="cpu",                 
        learning_rate=3e-4,          
        n_steps=1024,                 
        batch_size=256,
        gamma=0.99,                   
        gae_lambda=0.95,              
        clip_range=0.2,               
        ent_coef=0.005,               
    )

    class SyncNormStatsCallback(EvalCallback):
        def _on_step(self):
            self.eval_env.obs_rms = self.training_env.obs_rms
            return super()._on_step()

    eval_callback = SyncNormStatsCallback(
        eval_env,
        best_model_save_path="./models/best_reach",
        log_path="./eval_logs",
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save("./models/reach_final")
    train_env.save("./models/vecnormalize_stats.pkl") 
    print("Training complete. Best model saved to ./models/best_reach")
    print("Normalization stats saved to ./models/vecnormalize_stats.pkl")


if __name__ == "__main__":
    main()
