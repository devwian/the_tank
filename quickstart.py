#!/usr/bin/env python3
"""
快速启动脚本
提供交互式菜单，简化命令行操作
"""

import os
import sys
import subprocess
from pathlib import Path


def print_menu():
    """打印主菜单"""
    print("\n" + "="*60)
    print("🎮 坦克大战 RL 快速启动")
    print("="*60)
    print("1. 查看环境演示")
    print("2. 开始训练（快速，500k 步）")
    print("3. 开始训练（标准，1M 步）")
    print("4. 开始训练（长期，5M 步，分段保存）")
    print("5. 测试模型（可视化）")
    print("6. 测试模型（无渲染）")
    print("7. 交互演示")
    print("8. 查看已有检查点")
    print("0. 退出")
    print("="*60)


def run_command(cmd):
    """运行命令"""
    print(f"\n执行: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """主循环"""
    os.chdir(os.path.dirname(__file__) or ".")
    
    while True:
        print_menu()
        choice = input("请选择 (0-8): ").strip()
        
        if choice == "1":
            print("\n▶️  运行演示...")
            run_command("python main.py")
        
        elif choice == "2":
            print("\n▶️  开始快速训练 (500k 步)...")
            run_command("python train.py --mode basic --steps 500000")
        
        elif choice == "3":
            print("\n▶️  开始标准训练 (1M 步)...")
            run_command("python train.py --mode basic --steps 1000000")
        
        elif choice == "4":
            print("\n▶️  开始长期训练 (5M 步，每 100k 步保存)...")
            run_command("python train.py --mode checkpoint --steps 5000000 --checkpoint-freq 100000")
        
        elif choice == "5":
            model = input("请输入模型名称 (默认: tank_ppo_model): ").strip() or "tank_ppo_model"
            episodes = input("测试回合数 (默认: 10): ").strip() or "10"
            print(f"\n▶️  测试模型 {model} ({episodes} 回合，可视化)...")
            run_command(f"python test.py --mode test --model {model} --episodes {episodes}")
        
        elif choice == "6":
            model = input("请输入模型名称 (默认: tank_ppo_model): ").strip() or "tank_ppo_model"
            episodes = input("测试回合数 (默认: 20): ").strip() or "20"
            print(f"\n▶️  测试模型 {model} ({episodes} 回合，无渲染)...")
            run_command(f"python test.py --mode test --model {model} --episodes {episodes} --no-render")
        
        elif choice == "7":
            episodes = input("演示回合数 (默认: 3): ").strip() or "3"
            print(f"\n▶️  交互演示 ({episodes} 回合)...")
            run_command(f"python test.py --mode play --episodes {episodes}")
        
        elif choice == "8":
            print("\n📁 已有检查点:")
            print("\n本地模型:")
            for f in Path(".").glob("tank_*.zip"):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
            
            logs_path = Path("./logs")
            if logs_path.exists():
                print(f"\nlogs/ 目录:")
                for f in logs_path.glob("*.zip"):
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  - {f.name} ({size_mb:.1f} MB)")
            else:
                print(f"\nlogs/ 目录不存在")
        
        elif choice == "0":
            print("\n👋 再见!")
            sys.exit(0)
        
        else:
            print("\n❌ 无效选择，请重试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  已中断")
        sys.exit(0)
