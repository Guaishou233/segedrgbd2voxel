#!/usr/bin/env python3
"""
便捷脚本：将 dataset2 转换为 rgb2voxel 兼容格式

使用方法：
    python convert_dataset2.py

或者使用自定义路径：
    python convert_dataset2.py --source /path/to/source --target /path/to/target
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_converter.converter import Dataset2Converter


def main():
    # 默认路径配置
    # 源数据目录：dataset2/task_2
    source_dir = Path(__file__).parent.parent.parent / "dataset2" / "task_2"
    # 目标数据目录：rgb2voxel/data/dataset2
    target_dir = Path(__file__).parent.parent / "data" / "dataset2"
    
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="将 dataset2 转换为 rgb2voxel 兼容格式")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=str(source_dir),
        help=f"源数据目录路径 (默认: {source_dir})"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=str(target_dir),
        help=f"目标数据目录路径 (默认: {target_dir})"
    )
    parser.add_argument(
        "--task-name", "-n",
        type=str,
        default="task_2",
        help="任务名称 (默认: task_2)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset2 数据转换器")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"目标目录: {args.target}")
    print(f"任务名称: {args.task_name}")
    print("=" * 60)
    
    # 确认源目录存在
    if not Path(args.source).exists():
        print(f"错误: 源目录不存在: {args.source}")
        sys.exit(1)
    
    # 执行转换
    try:
        converter = Dataset2Converter(
            source_dir=args.source,
            target_dir=args.target,
            task_name=args.task_name
        )
        converter.convert()
        print("\n✓ 转换成功完成!")
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

