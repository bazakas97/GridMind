"""
Train a PPO agent on the battery dispatch environment.

Usage:
    python src/train_rl_agent.py
    python src/train_rl_agent.py --timesteps 1000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.battery_env import BatteryEnv, EnvConfig

PRICE_PATH  = Path(__file__).parent.parent / "outputs" / "metrics" / "price_forecasts.csv"
LOAD_PATH   = Path(__file__).parent.parent / "outputs" / "metrics" / "load_forecasts.csv"
MODEL_DIR   = Path(__file__).parent.parent / "outputs" / "rl_agent"
MODEL_PATH  = MODEL_DIR / "ppo_battery"


def load_training_data():
    prices_df = pd.read_csv(PRICE_PATH, index_col="datetime", parse_dates=True).sort_index()
    loads_df  = pd.read_csv(LOAD_PATH,  index_col="datetime", parse_dates=True).sort_index()

    # Use actual prices and a single model's forecasted load (first model found)
    prices = prices_df["price_actual"].values.astype(np.float32)

    # Filter one model if multiple exist
    if "model" in loads_df.columns:
        first_model = loads_df["model"].iloc[0]
        loads_df = loads_df[loads_df["model"] == first_model]
    loads = loads_df["actual_kw"].values.astype(np.float32)

    # Align lengths
    n = min(len(prices), len(loads))
    prices, loads = prices[:n], loads[:n]

    # 80/20 train/eval split
    split = int(n * 0.80)
    return prices[:split], loads[:split], prices[split:], loads[split:]


def make_env(prices, loads, config):
    def _init():
        return BatteryEnv(prices, loads, config)
    return _init


def train(timesteps: int = 500_000):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    p_train, l_train, p_eval, l_eval = load_training_data()
    print(f"  Train hours: {len(p_train)}   Eval hours: {len(p_eval)}")

    cfg = EnvConfig()

    # Vectorised training env (4 parallel workers)
    train_env = make_vec_env(make_env(p_train, l_train, cfg), n_envs=4)
    eval_env  = BatteryEnv(p_eval, l_eval, cfg)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(MODEL_DIR / "logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_battery",
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=str(MODEL_DIR / "tb_logs"),
    )

    print(f"\nTraining PPO for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=[eval_cb, checkpoint_cb])

    model.save(str(MODEL_PATH))
    print(f"\nModel saved to: {MODEL_PATH}.zip")

    # Quick eval
    obs, _ = eval_env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)
        total_reward += reward
    print(f"Eval total reward (negative cost): {total_reward:.4f} €")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()
    train(args.timesteps)
