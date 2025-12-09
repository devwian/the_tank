import gymnasium as gym
from stable_baselines3 import PPO
from gyming import TankTroubleEnv  # 确保导入了你的环境类
from stable_baselines3.common.callbacks import CheckpointCallback

def train_with_checkpoint():
    env = TankTroubleEnv(render_mode=None)
    model = PPO("MlpPolicy", env, verbose=1)

    # 每 20000 步保存一次模型到 ./logs/ 文件夹
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./logs/",
        name_prefix="tank_model"
    )

    # 添加 callback 参数
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    
    # 最后保存最终版
    model.save("tank_model_final")
    env.close()

def train():
    # 1. 创建训练环境
    # render_mode=None 表示不显示画面，训练速度最快
    env = TankTroubleEnv(render_mode=None)

    # 2. 定义模型
    # 使用 PPO 算法，MlpPolicy 适合这种纯数值输入的观察空间
    # verbose=1 会打印训练进度
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    print("开始训练...")
    # 3. 开始学习
    # total_timesteps 决定训练多久。建议至少 100,000 步，想要强力效果可能需要 1,000,000+
    model.learn(total_timesteps=1000000)

    # 4. 保存模型
    save_path = "tank_ppo_model"
    model.save(save_path)
    print(f"模型已保存到: {save_path}.zip")
    
    env.close()

if __name__ == "__main__":
    train()