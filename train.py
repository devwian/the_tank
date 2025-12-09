"""
坦克大战 RL 模型训练脚本
使用 Stable Baselines3 的 PPO 算法训练玩家坦克对抗 AI Bot
"""

import gymnasium as gym
from stable_baselines3 import PPO
from environment import TankTroubleEnv  # 从模块化的 environment.py 导入
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def train_with_checkpoint(total_timesteps=500000, checkpoint_freq=20000):
    """
    带检查点保存的训练函数
    
    Args:
        total_timesteps: 总训练步数
        checkpoint_freq: 每多少步保存一次检查点
    """
    # 创建 logs 目录
    os.makedirs("./logs", exist_ok=True)
    
    env = TankTroubleEnv(render_mode=None)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # 每 checkpoint_freq 步保存一次模型到 ./logs/ 文件夹
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./logs/",
        name_prefix="tank_model"
    )

    print(f"开始训练... 总步数: {total_timesteps}")
    # 添加 callback 参数
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 最后保存最终版
    model.save("./logs/tank_model_final")
    print("✓ 最终模型已保存到: ./logs/tank_model_final.zip")
    env.close()

def train(total_timesteps=1000000):
    """
    基础训练函数
    
    Args:
        total_timesteps: 总训练步数，建议至少 100,000，强力效果可能需要 1,000,000+
    """
    # 1. 创建训练环境
    # render_mode=None 表示不显示画面，训练速度最快
    print("正在初始化环境...")
    env = TankTroubleEnv(render_mode=None)

    # 2. 定义模型
    # 使用 PPO 算法，MlpPolicy 适合这种纯数值输入的观察空间
    # verbose=1 会打印训练进度
    # learning_rate 设置学习率
    print("正在创建 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    print(f"开始训练... 总步数: {total_timesteps}")
    print("="*60)
    
    # 3. 开始学习
    model.learn(total_timesteps=total_timesteps)

    # 4. 保存模型
    save_path = "tank_ppo_model"
    model.save(save_path)
    print(f"\n✓ 模型已保存到: {save_path}.zip")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="坦克大战 PPO 训练")
    parser.add_argument(
        "--mode",
        choices=["basic", "checkpoint"],
        default="basic",
        help="训练模式: basic=基础训练, checkpoint=带检查点保存"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000000,
        help="总训练步数 (默认: 1000000)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20000,
        help="检查点保存频率 (默认: 20000)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("坦克大战 RL 训练")
    print("="*60)
    
    if args.mode == "basic":
        print(f"模式: 基础训练 ({args.steps} 步)")
        train(total_timesteps=args.steps)
    else:  # checkpoint
        print(f"模式: 检查点训练 ({args.steps} 步, 每 {args.checkpoint_freq} 步保存)")
        train_with_checkpoint(
            total_timesteps=args.steps,
            checkpoint_freq=args.checkpoint_freq
        )
    
    print("="*60)
    print("训练完成!")
    print("="*60)