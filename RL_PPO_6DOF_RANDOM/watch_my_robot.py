import time
from pathlib import Path
from stable_baselines3 import PPO
from my_robot_env import MyRobotEnv

script_dir = Path(__file__).resolve().parent
model_path = script_dir / "ppo_my_robot.zip"

env = MyRobotEnv(render_mode="human")
model = PPO.load(model_path, device="cpu")

obs, info = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(0.02)  # Slow down for better visualization

env.close()
