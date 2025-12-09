import gymnasium as gym
from stable_baselines3 import PPO
from gyming import TankTroubleEnv
import pygame

def play():
    # 1. 创建测试环境
    # render_mode="human" 开启画面
    env = TankTroubleEnv(render_mode="human")

    # 2. 加载模型
    # 只需要文件名，不需要加 .zip 后缀
    model_path = "tank_ppo_model"
    try:
        model = PPO.load(model_path)
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print("未找到模型文件，请先运行训练脚本！")
        return

    # 3. 运行游戏
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"--- Episode {ep + 1} ---")
        
        while not done:
            # 让模型预测动作
            # deterministic=True 表示让模型输出它认为概率最大的动作（不再随机探索）
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # 处理关闭窗口事件
            if env.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()

        print(f"Episode {ep + 1} Score: {total_reward:.2f}")
        pygame.time.wait(1000) # 休息1秒

    env.close()

if __name__ == "__main__":
    play()