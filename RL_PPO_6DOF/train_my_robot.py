from stable_baselines3 import PPO
from my_robot_env import MyRobotEnv

# Train without rendering for faster training
# (rendering slows it down significantly)
env = MyRobotEnv(render_mode=None)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    ent_coef=0.01,
    device="cpu",
)

print("Training device:", model.device)

model.learn(total_timesteps=3_000_000)  # 3m steps for random start + random target

model.save("ppo_my_robot")

env.close()