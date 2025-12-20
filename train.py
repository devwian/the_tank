"""
坦克动荡 RL 模型训练脚本
使用 Stable Baselines3 的 PPO 算法训练玩家坦克对抗 AI Bot
"""

import gymnasium as gym
from stable_baselines3 import PPO
from environment import TankTroubleEnv  # 从模块化的 environment.py 导入
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime


class RewardLoggerCallback(BaseCallback):
    """
    自定义回调，用于在控制台打印每个回合的奖励和结果，并记录到 TensorBoard
    """
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_count = 0
        self.win_count = 0

    def _on_step(self) -> bool:
        # 检查 infos 中是否有 episode 信息（由 Monitor 包装器提供）
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                # 从环境返回的 info 中获取自定义结果
                result = info.get("result", "N/A")
                
                # 记录到 TensorBoard
                self.logger.record("custom/episode_reward", reward)
                self.logger.record("custom/episode_length", length)
                
                result_emoji = "🏁"
                if result == "win":
                    result_emoji = "🎯 胜利"
                    self.win_count += 1
                    self.logger.record("custom/is_win", 1)
                elif result == "lose":
                    result_emoji = "💀 失败"
                    self.logger.record("custom/is_win", 0)
                elif result == "timeout":
                    result_emoji = "⏰ 超时"
                    self.logger.record("custom/is_win", 0)
                
                # 计算胜率并记录
                win_rate = self.win_count / self.episode_count
                self.logger.record("custom/win_rate", win_rate)
                
                # 强制将记录写入 TensorBoard (在 rollout 结束时会自动写入，但这里可以手动触发或等待)
                # self.logger.dump(self.num_timesteps)
                
                print(f"  [回合 {self.episode_count}] {result_emoji} | 奖励: {reward:7.2f} | 步数: {length} | 胜率: {win_rate:.1%}")
        return True


def train_curriculum(stage_steps=None):
    """
    课程学习训练函数 - 分阶段逐步提升难度
    
    阶段1: 无墙体，Bot不移动不攻击 (学习基础操作和射击)
    阶段2: 有墙体，Bot移动但不攻击 (学习导航和追踪移动目标)
    阶段3: 有墙体，Bot完整行为 (学习完整对战)
    
    Args:
        stage_steps: 每个阶段的训练步数列表 [阶段1, 阶段2, 阶段3]
    """
    if stage_steps is None:
        stage_steps = [400000, 600000, 1000000]  # 增加训练步数
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/curriculum_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    stages = [
        {"difficulty": 1, "name": "阶段1: 静态目标", "desc": "无墙体，Bot静止"},
        {"difficulty": 2, "name": "阶段2: 移动目标", "desc": "有墙体，Bot只移动"},
        {"difficulty": 3, "name": "阶段3: 完整对战", "desc": "有墙体，Bot完整AI"},
    ]
    
    model = None
    
    for i, stage in enumerate(stages):
        print("\n" + "="*60)
        print(f"🎯 {stage['name']} - {stage['desc']}")
        print(f"   训练步数: {stage_steps[i]:,}")
        print("="*60)
        
        # 创建对应难度的环境
        stage_log_dir = f"{log_dir}/stage{i+1}"
        os.makedirs(stage_log_dir, exist_ok=True)
        
        env = Monitor(
            TankTroubleEnv(render_mode=None, difficulty=stage["difficulty"]),
            stage_log_dir
        )
        
        if model is None:
            # 第一阶段：创建新模型
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0002,  # 稍微降低学习率以提高稳定性
                n_steps=2048,
                batch_size=256,        # 增大 batch_size 提高梯度估计稳定性
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,         # 增加熵系数，鼓励探索
                tensorboard_log=log_dir
            )
        else:
            # 后续阶段：复用模型，更新环境
            model.set_env(env)
        
        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=stage_log_dir,
            name_prefix=f"stage{i+1}_model"
        )
        
        # 奖励日志回调
        reward_logger = RewardLoggerCallback()
        
        # 组合回调
        callbacks = CallbackList([checkpoint_callback, reward_logger])
        
        # 训练
        model.learn(
            total_timesteps=stage_steps[i],
            callback=callbacks,
            reset_num_timesteps=False,  # 保持总步数计数
            tb_log_name=f"stage{i+1}"
        )
        
        # 保存阶段模型
        model.save(f"{stage_log_dir}/stage{i+1}_final")
        print(f"✓ {stage['name']} 完成，模型已保存")
        
        env.close()
    
    # 保存最终模型
    model.save(f"{log_dir}/tank_curriculum_final")
    print(f"\n🎉 课程学习完成！最终模型: {log_dir}/tank_curriculum_final.zip")
    print(f"📊 TensorBoard: tensorboard --logdir {log_dir}")


def train_with_checkpoint(total_timesteps=500000, checkpoint_freq=20000, difficulty=3):
    """
    带检查点保存的训练函数
    
    Args:
        total_timesteps: 总训练步数
        checkpoint_freq: 每多少步保存一次检查点
    """
    # 创建日志目录（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用 Monitor 包装环境以记录 episode 统计
    env = Monitor(TankTroubleEnv(render_mode=None), log_dir)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log=log_dir  # 启用 TensorBoard 日志
    )

    # 每 checkpoint_freq 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix="tank_model"
    )
    
    # 奖励日志回调
    reward_logger = RewardLoggerCallback()
    
    # 组合回调
    callbacks = CallbackList([checkpoint_callback, reward_logger])

    print(f"开始训练... 总步数: {total_timesteps}")
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"📊 运行 `tensorboard --logdir {log_dir}` 查看训练曲线")
    
    # 添加 callback 参数
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # 最后保存最终版
    model.save(f"{log_dir}/tank_model_final")
    print(f"✓ 最终模型已保存到: {log_dir}/tank_model_final.zip")
    env.close()

def train(total_timesteps=3000000):
    """
    基础训练函数（带 TensorBoard 日志）
    
    Args:
        total_timesteps: 总训练步数，建议至少 100,000，强力效果可能需要 1,000,000+
    """
    # 创建日志目录（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 创建训练环境，使用 Monitor 包装以记录 episode 统计
    print("正在初始化环境...")
    env = Monitor(TankTroubleEnv(render_mode=None), log_dir)

    # 2. 定义模型
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
        clip_range=0.2,
        tensorboard_log=log_dir  # 启用 TensorBoard 日志
    )

    print(f"开始训练... 总步数: {total_timesteps}")
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"📊 运行 `tensorboard --logdir {log_dir}` 查看训练曲线")
    print("="*60)
    
    # 3. 开始学习
    reward_logger = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)

    # 4. 保存模型
    save_path = f"{log_dir}/tank_ppo_model"
    model.save(save_path)
    print(f"\n✓ 模型已保存到: {save_path}.zip")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="坦克大战 PPO 训练")
    parser.add_argument(
        "--mode",
        choices=["basic", "checkpoint", "curriculum"],
        default="basic",
        help="训练模式: basic=基础训练, checkpoint=带检查点保存, curriculum=课程学习"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000000,
        help="总训练步数 (默认: 1000000)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20000,
        help="检查点保存频率 (默认: 20000)"
    )
    parser.add_argument(
        "--stage-steps",
        type=str,
        default="200000,300000,500000",
        help="课程学习各阶段步数，逗号分隔 (默认: 200000,300000,500000)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("坦克大战 RL 训练")
    print("="*60)
    
    # if args.mode == "basic":
    #     print(f"模式: 基础训练 ({args.steps} 步)")
    #     train(total_timesteps=args.steps)
    # elif args.mode == "checkpoint":
    #     print(f"模式: 检查点训练 ({args.steps} 步, 每 {args.checkpoint_freq} 步保存)")
    #     train_with_checkpoint(
    #         total_timesteps=args.steps,
    #         checkpoint_freq=args.checkpoint_freq
    #     )
    # else:  # curriculum
    #     stage_steps = [int(s) for s in args.stage_steps.split(",")]
    #     total = sum(stage_steps)
    #     print(f"模式: 课程学习 (总步数: {total:,})")
    #     print(f"  阶段1 (静态目标): {stage_steps[0]:,} 步")
    #     print(f"  阶段2 (移动目标): {stage_steps[1]:,} 步")
    #     print(f"  阶段3 (完整对战): {stage_steps[2]:,} 步")
    #     train_curriculum(stage_steps=stage_steps)
    print(f"模式: 基础训练 ({args.steps} 步)")
    train(total_timesteps=args.steps)
    
    print("="*60)
    print("训练完成!")
    print("="*60)