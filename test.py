"""
坦克大战 RL 模型测试和推理脚本
用于评估训练好的模型性能
"""

from stable_baselines3 import PPO
from environment import TankTroubleEnv
import argparse
import os


def test_model(model_path, num_episodes=5, render=True):
    """
    测试已训练的模型
    
    Args:
        model_path: 模型文件路径 (不需要 .zip 后缀)
        num_episodes: 测试回合数
        render: 是否渲染画面
    """
    # 检查模型文件是否存在
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 错误: 找不到模型文件 {model_path}.zip")
        print("可用的模型:")
        if os.path.exists("./logs"):
            for f in os.listdir("./logs"):
                if f.endswith(".zip"):
                    print(f"  - ./logs/{f}")
        if os.path.exists("tank_ppo_model.zip"):
            print(f"  - tank_ppo_model.zip")
        return
    
    render_mode = "human" if render else None
    env = TankTroubleEnv(render_mode=render_mode)
    
    # 加载模型
    print(f"正在加载模型: {model_path}...")
    model = PPO.load(model_path)
    print("✓ 模型加载成功")
    
    print(f"\n开始测试 ({num_episodes} 回合)...")
    print("="*60)
    
    total_reward = 0
    total_steps = 0
    wins = 0  # 击败敌人的次数
    losses = 0  # 被击中的次数
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated
            
            # 处理窗口关闭事件
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
        
        total_reward += episode_reward
        total_steps += episode_steps
        
        # 根据奖励判断胜负
        if episode_reward > 5:
            wins += 1
            status = "🎉 胜利"
        elif episode_reward < -5:
            losses += 1
            status = "💥 失败"
        else:
            status = "➖ 平局"
        
        print(f"[第 {ep + 1}/{num_episodes} 回合] {status} | 步数: {episode_steps:4d} | 奖励: {episode_reward:7.2f}")
    
    print("="*60)
    print("\n📊 测试统计:")
    print(f"  总回合数: {num_episodes}")
    print(f"  胜利次数: {wins}")
    print(f"  失败次数: {losses}")
    print(f"  平局次数: {num_episodes - wins - losses}")
    print(f"  平均步数: {total_steps / num_episodes:.1f}")
    print(f"  平均奖励: {total_reward / num_episodes:.2f}")
    print(f"  胜率: {wins / num_episodes * 100:.1f}%")
    
    env.close()


def play_interactive(num_episodes=1):
    """
    交互模式：使用训练好的模型进行演示
    """
    render_mode = "human"
    env = TankTroubleEnv(render_mode=render_mode)
    
    try:
        model = PPO.load("tank_ppo_model")
        print("✓ 已加载 tank_ppo_model")
    except FileNotFoundError:
        print("❌ 找不到 tank_ppo_model.zip")
        print("请先运行: python train.py")
        env.close()
        return
    
    print(f"\n开始交互演示 ({num_episodes} 回合)...")
    print("按 ESC 或关闭窗口退出")
    print("="*60)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        print(f"\n[第 {ep + 1}/{num_episodes} 回合]")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            
            # 处理事件
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
        
        print(f"  完成: {steps} 步")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="坦克大战模型测试")
    parser.add_argument(
        "--mode",
        choices=["test", "play"],
        default="test",
        help="模式: test=测试模式, play=交互演示"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tank_ppo_model",
        help="模型路径 (不需要 .zip 后缀)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="回合数 (默认: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不显示画面 (仅测试模式)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("坦克大战 RL 模型测试")
    print("="*60)
    
    if args.mode == "test":
        render = not args.no_render
        test_model(args.model, num_episodes=args.episodes, render=render)
    else:  # play
        play_interactive(num_episodes=args.episodes)
    
    print("\n✓ 完成!")
