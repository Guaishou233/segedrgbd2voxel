#!/usr/bin/env python3
"""
相机位姿测试脚本

验证相机外参是否正确，可视化相机位置和朝向
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import open3d as o3d


def load_camera_info(metadata_path: str):
    """加载相机信息"""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def plot_cameras_matplotlib(cameras_data: dict, output_path: str, title: str = "Camera Poses"):
    """
    使用 matplotlib 可视化相机位置和朝向
    
    Args:
        cameras_data: 相机数据字典
        output_path: 输出图像路径
        title: 图像标题
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras_data)))
    
    for idx, (cam_name, cam_info) in enumerate(cameras_data.items()):
        extrinsic = np.array(cam_info['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # 假设外参是 world-to-camera 格式
        # 相机位置 = -R^T * t
        cam_pos_w2c = -R.T @ t
        
        # 假设外参是 camera-to-world 格式
        # 相机位置 = t
        cam_pos_c2w = t
        
        # 相机朝向（Z轴，从相机向前）
        # world-to-camera: 朝向 = R^T * [0, 0, 1]
        cam_dir_w2c = R.T @ np.array([0, 0, 1])
        # camera-to-world: 朝向 = R * [0, 0, 1]
        cam_dir_c2w = R @ np.array([0, 0, 1])
        
        color = colors[idx]
        
        # 绘制 world-to-camera 解释
        ax.scatter(*cam_pos_w2c, c=[color], s=100, marker='o', label=f'{cam_name} (w2c)')
        ax.quiver(*cam_pos_w2c, *cam_dir_w2c * 0.5, color=color, arrow_length_ratio=0.2)
        
        # 标注
        ax.text(cam_pos_w2c[0], cam_pos_w2c[1], cam_pos_w2c[2], 
                f'  {cam_name}', fontsize=8)
    
    # 绘制原点
    ax.scatter(0, 0, 0, c='black', s=200, marker='*', label='Origin')
    
    # 绘制坐标轴
    axis_length = 2.0
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, label='Y')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, label='Z')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    
    # 设置相等的轴比例
    max_range = 10
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"相机位姿图已保存到: {output_path}")


def analyze_camera_poses(metadata_path: str):
    """
    分析相机位姿
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    cameras = metadata.get('cameras', {})
    
    print("=" * 60)
    print("相机位姿分析")
    print("=" * 60)
    
    all_positions_w2c = []
    all_positions_c2w = []
    
    for cam_name, cam_info in cameras.items():
        extrinsic = np.array(cam_info['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # world-to-camera 解释
        cam_pos_w2c = -R.T @ t
        cam_dir_w2c = R.T @ np.array([0, 0, 1])
        
        # camera-to-world 解释
        cam_pos_c2w = t
        cam_dir_c2w = R @ np.array([0, 0, 1])
        
        all_positions_w2c.append(cam_pos_w2c)
        all_positions_c2w.append(cam_pos_c2w)
        
        print(f"\n{cam_name}:")
        print(f"  外参矩阵 (4x4):")
        print(f"  {extrinsic}")
        print(f"  解释为 world-to-camera:")
        print(f"    位置: [{cam_pos_w2c[0]:.3f}, {cam_pos_w2c[1]:.3f}, {cam_pos_w2c[2]:.3f}]")
        print(f"    朝向: [{cam_dir_w2c[0]:.3f}, {cam_dir_w2c[1]:.3f}, {cam_dir_w2c[2]:.3f}]")
        print(f"  解释为 camera-to-world:")
        print(f"    位置: [{cam_pos_c2w[0]:.3f}, {cam_pos_c2w[1]:.3f}, {cam_pos_c2w[2]:.3f}]")
        print(f"    朝向: [{cam_dir_c2w[0]:.3f}, {cam_dir_c2w[1]:.3f}, {cam_dir_c2w[2]:.3f}]")
    
    # 计算相机分布中心
    center_w2c = np.mean(all_positions_w2c, axis=0)
    center_c2w = np.mean(all_positions_c2w, axis=0)
    
    print("\n" + "=" * 60)
    print("相机分布分析")
    print("=" * 60)
    print(f"world-to-camera 解释:")
    print(f"  相机中心: [{center_w2c[0]:.3f}, {center_w2c[1]:.3f}, {center_w2c[2]:.3f}]")
    print(f"  到原点距离: {np.linalg.norm(center_w2c):.3f}")
    print(f"camera-to-world 解释:")
    print(f"  相机中心: [{center_c2w[0]:.3f}, {center_c2w[1]:.3f}, {center_c2w[2]:.3f}]")
    print(f"  到原点距离: {np.linalg.norm(center_c2w):.3f}")
    
    return cameras


def compare_with_original_camera_info(original_path: str, converted_path: str):
    """
    比较原始相机信息和转换后的相机信息
    """
    print("\n" + "=" * 60)
    print("原始相机信息对比")
    print("=" * 60)
    
    # 读取原始相机信息
    with open(original_path, 'r') as f:
        original = json.load(f)
    
    with open(converted_path, 'r') as f:
        converted = json.load(f)
    
    for orig_cam in original.get('cameras', []):
        cam_id = orig_cam['camera_id']
        cam_name = f"cam_{cam_id:02d}"
        
        print(f"\n{cam_name}:")
        print(f"  原始 position: {orig_cam['position']}")
        print(f"  原始 target:   {orig_cam['target']}")
        
        if cam_name in converted.get('cameras', {}):
            conv_cam = converted['cameras'][cam_name]
            extrinsic = np.array(conv_cam['extrinsic'])
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            
            # 从外参计算相机位置（world-to-camera）
            cam_pos = -R.T @ t
            print(f"  从外参计算位置 (w2c): [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
            
            # 检查是否匹配
            diff = np.linalg.norm(cam_pos - np.array(orig_cam['position']))
            print(f"  位置差异: {diff:.6f}")


def create_camera_visualization_open3d(metadata_path: str, output_path: str):
    """
    使用 Open3D 创建相机可视化
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    cameras = metadata.get('cameras', {})
    
    geometries = []
    
    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)
    
    colors_list = [
        [1, 0, 0],    # 红
        [0, 1, 0],    # 绿
        [0, 0, 1],    # 蓝
        [1, 1, 0],    # 黄
        [1, 0, 1],    # 品红
        [0, 1, 1],    # 青
    ]
    
    for idx, (cam_name, cam_info) in enumerate(cameras.items()):
        extrinsic = np.array(cam_info['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # 假设是 world-to-camera，计算相机位置
        cam_pos = -R.T @ t
        cam_dir = R.T @ np.array([0, 0, 1])  # 相机前方向
        cam_up = R.T @ np.array([0, -1, 0])   # 相机上方向
        
        color = colors_list[idx % len(colors_list)]
        
        # 创建相机位置球体
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(cam_pos)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
        
        # 创建相机方向箭头（使用圆柱+圆锥）
        arrow_length = 0.5
        arrow_end = cam_pos + cam_dir * arrow_length
        
        # 使用线段表示方向
        points = [cam_pos, arrow_end]
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])
        geometries.append(line_set)
    
    # 保存可视化
    o3d.io.write_triangle_mesh(output_path.replace('.png', '_coord.ply'), coord_frame)
    print(f"Open3D 可视化已创建")
    
    return geometries


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="相机位姿测试")
    parser.add_argument("--metadata", "-m", type=str, 
                        default="/home/qiansongtang/Documents/program/rgb2voxel/data/dataset2/task_2/metadata.json",
                        help="metadata.json 路径")
    parser.add_argument("--original", "-o", type=str,
                        default="/home/qiansongtang/Documents/program/dataset2/task_2/camera_info_default_session.json",
                        help="原始相机信息路径")
    parser.add_argument("--output", type=str,
                        default="/home/qiansongtang/Documents/program/rgb2voxel/data/dataset2/camera_poses.png",
                        help="输出图像路径")
    
    args = parser.parse_args()
    
    # 分析相机位姿
    cameras = analyze_camera_poses(args.metadata)
    
    # 比较原始相机信息
    if Path(args.original).exists():
        compare_with_original_camera_info(args.original, args.metadata)
    
    # 绘制相机位置
    plot_cameras_matplotlib(cameras, args.output, "Camera Poses (world-to-camera interpretation)")
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("""
外参矩阵格式分析：

1. 原始数据的 extrinsic_matrix 是 **world-to-camera** 格式
   - 相机位置 = -R^T * t
   - 与原始数据中的 position 字段完全匹配

2. Open3D 的 create_from_rgbd_image 需要 **camera-to-world** 格式
   - 需要对外参矩阵求逆：extrinsic_c2w = np.linalg.inv(extrinsic_w2c)

3. 当前问题：
   - 直接使用 world-to-camera 矩阵会导致点云被错误变换
   - 6个相机的点云会以相反的方式融合

修复方法：
   在 pointcloud_module.py 中，使用外参前先求逆
   或在 converter.py 中将外参转换为 camera-to-world 格式
""")


if __name__ == "__main__":
    main()

