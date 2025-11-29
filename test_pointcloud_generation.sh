#!/bin/bash
# 快速测试点云生成脚本

echo "=========================================="
echo "点云生成测试脚本"
echo "=========================================="
echo ""

# 测试模式：处理前3个样本
echo "步骤1: 测试模式 - 处理前3个样本"
echo "运行命令: python generate_segmented_pointcloud.py --test 3"
echo ""
read -p "按Enter继续测试，或Ctrl+C取消..."

python generate_segmented_pointcloud.py --test 3

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "如果测试成功，可以运行完整数据集生成："
echo "  python generate_segmented_pointcloud.py"
echo ""
echo "或者处理更多样本："
echo "  python generate_segmented_pointcloud.py --test 10"
echo ""

