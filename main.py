"""
主程序入口
"""

import pygame
from environment import TankTroubleEnv


def run_demo(num_episodes=5, render=True):
    """
    运行演示
    num_episodes: 运行的回合数
    render: 是否渲染
    """
    render_mode = "human" if render else None
    env = TankTroubleEnv(render_mode=render_mode)
    
    print("=" * 60)
    print("坦克大战 RL 环境")
    print("=" * 60)
    print(f"运行模式: {'可视化' if render else '无渲染'}")
    print(f"回合数: {num_episodes}")
    print("=" * 60)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n[第 {ep + 1}/{num_episodes} 回合] 开始...")
        
        while not done:
            # 随机动作（可替换为智能策略）
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # 处理事件
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
        
        print(f"  回合完成 - 步数: {steps}, 累积奖励: {episode_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    # 可视化模式运行
    run_demo(num_episodes=3, render=True)
