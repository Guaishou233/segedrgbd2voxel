#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式体素可视化工具
支持鼠标拖拽从不同角度观察体素，体素之间无间隙
"""

import os
import sys
import numpy as np
import json
import argparse

try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    print("Error: Open3D未安装，请安装: pip install open3d")
    sys.exit(1)


def load_voxel_data(voxelized_dir):
    """
    加载体素数据
    
    Args:
        voxelized_dir: 体素数据目录路径
    
    Returns:
        voxel_grid: 体素占用网格 (voxel_size, voxel_size, voxel_size)
        voxel_colors: 体素颜色 (voxel_size, voxel_size, voxel_size, 3)
        voxel_semantics: 体素语义 (voxel_size, voxel_size, voxel_size)
        metadata: 元数据字典
    """
    print(f"正在加载体素数据: {voxelized_dir}")
    
    # 加载元数据
    metadata_path = os.path.join(voxelized_dir, "voxel_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"体素大小: {metadata['voxel_size']}x{metadata['voxel_size']}x{metadata['voxel_size']}")
    print(f"物理尺寸: {metadata['physical_size']}m")
    print(f"体素分辨率: {metadata['voxel_resolution']}m/体素")
    print(f"占用体素数: {metadata['occupied_voxels']} / {metadata['total_voxels']}")
    
    # 加载体素网格
    voxel_grid_path = os.path.join(voxelized_dir, "voxel_grid.npy")
    if not os.path.exists(voxel_grid_path):
        raise FileNotFoundError(f"体素网格文件不存在: {voxel_grid_path}")
    voxel_grid = np.load(voxel_grid_path)
    
    # 加载体素颜色
    voxel_colors_path = os.path.join(voxelized_dir, "voxel_colors.npy")
    if not os.path.exists(voxel_colors_path):
        raise FileNotFoundError(f"体素颜色文件不存在: {voxel_colors_path}")
    voxel_colors = np.load(voxel_colors_path)
    
    # 加载体素语义
    voxel_semantics_path = os.path.join(voxelized_dir, "voxel_semantics.npy")
    if not os.path.exists(voxel_semantics_path):
        raise FileNotFoundError(f"体素语义文件不存在: {voxel_semantics_path}")
    voxel_semantics = np.load(voxel_semantics_path)
    
    print("体素数据加载完成！")
    
    return voxel_grid, voxel_colors, voxel_semantics, metadata


def create_voxel_mesh(voxel_grid, voxel_colors, voxel_semantics, metadata, use_semantic_colors=False):
    """
    创建体素网格，确保体素之间无间隙
    
    Args:
        voxel_grid: 体素占用网格
        voxel_colors: 体素颜色
        voxel_semantics: 体素语义
        metadata: 元数据
        use_semantic_colors: 是否使用语义颜色（True）或RGB颜色（False）
    
    Returns:
        mesh: Open3D TriangleMesh对象
        voxel_info: 字典，包含体素索引信息，用于快速切换颜色
    """
    voxel_size = metadata['voxel_size']
    voxel_resolution = metadata['voxel_resolution']
    voxel_origin = np.array(metadata['voxel_origin_array'])
    
    print(f"正在创建体素网格（无间隙）...")
    
    # 创建基础立方体（单位立方体，边长为1）
    # 这个立方体将被缩放和移动到正确的位置
    unit_cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    unit_cube.compute_vertex_normals()
    
    # 获取单位立方体的顶点和三角形
    unit_vertices = np.asarray(unit_cube.vertices)
    unit_triangles = np.asarray(unit_cube.triangles)
    num_unit_vertices = len(unit_vertices)
    num_unit_triangles = len(unit_triangles)
    
    # 遍历所有占用的体素
    occupied_indices = np.argwhere(voxel_grid)
    num_voxels = len(occupied_indices)
    
    print(f"处理 {num_voxels} 个占用的体素...")
    
    # 预分配数组以提高性能
    all_vertices = np.zeros((num_voxels * num_unit_vertices, 3), dtype=np.float32)
    all_triangles = np.zeros((num_voxels * num_unit_triangles, 3), dtype=np.int32)
    all_vertex_colors = np.zeros((num_voxels * num_unit_vertices, 3), dtype=np.float32)
    
    # 保存每个体素的索引信息，用于快速切换颜色
    voxel_info = {
        'occupied_indices': occupied_indices,
        'num_unit_vertices': num_unit_vertices,
        'voxel_colors': voxel_colors,
        'voxel_semantics': voxel_semantics
    }
    
    for idx, (x, y, z) in enumerate(occupied_indices):
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx + 1}/{num_voxels}")
        
        # 计算体素在世界坐标系中的最小角位置（左下前角）
        # 体素索引 (x, y, z) 对应体素的最小角位置
        voxel_min = voxel_origin + np.array([x, y, z]) * voxel_resolution
        
        # 创建该体素的立方体
        # 将单位立方体缩放到体素分辨率大小，并移动到正确位置
        # 单位立方体的顶点范围是 [0, 1]，我们需要将其缩放到 [0, voxel_resolution]
        # 然后平移到 voxel_min 位置
        voxel_vertices = unit_vertices * voxel_resolution + voxel_min
        
        # 获取颜色
        if use_semantic_colors:
            semantic = voxel_semantics[x, y, z]
            # 根据语义设置颜色
            if semantic == 1:
                color = np.array([1.0, 0.0, 0.0])  # 红色 - robot arm
            elif semantic == 2:
                color = np.array([0.5, 0.5, 0.5])  # 灰色 - others
            else:
                color = np.array([0.0, 0.0, 1.0])  # 蓝色 - 其他
        else:
            # 使用RGB颜色
            color = voxel_colors[x, y, z].astype(np.float32) / 255.0  # 转换为0-1范围
        
        # 计算数组索引
        vertex_start = idx * num_unit_vertices
        vertex_end = vertex_start + num_unit_vertices
        triangle_start = idx * num_unit_triangles
        triangle_end = triangle_start + num_unit_triangles
        
        # 添加顶点
        all_vertices[vertex_start:vertex_end] = voxel_vertices
        
        # 添加三角形（需要调整索引）
        all_triangles[triangle_start:triangle_end] = unit_triangles + vertex_start
        
        # 为每个顶点添加颜色
        all_vertex_colors[vertex_start:vertex_end] = color
    
    print("创建网格对象...")
    
    # 创建网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)
    mesh.compute_vertex_normals()
    
    print(f"体素网格创建完成！")
    print(f"  总顶点数: {len(all_vertices):,}")
    print(f"  总三角形数: {len(all_triangles):,}")
    
    return mesh, voxel_info


def update_voxel_colors(mesh, voxel_info, use_semantic_colors):
    """
    更新体素网格的颜色（不重新创建网格）
    
    Args:
        mesh: Open3D TriangleMesh对象
        voxel_info: 体素信息字典
        use_semantic_colors: 是否使用语义颜色
    """
    occupied_indices = voxel_info['occupied_indices']
    num_unit_vertices = voxel_info['num_unit_vertices']
    voxel_colors = voxel_info['voxel_colors']
    voxel_semantics = voxel_info['voxel_semantics']
    
    all_vertex_colors = np.zeros((len(occupied_indices) * num_unit_vertices, 3), dtype=np.float32)
    
    for idx, (x, y, z) in enumerate(occupied_indices):
        # 获取颜色
        if use_semantic_colors:
            semantic = voxel_semantics[x, y, z]
            # 根据语义设置颜色
            if semantic == 1:
                color = np.array([1.0, 0.0, 0.0])  # 红色 - robot arm
            elif semantic == 2:
                color = np.array([0.5, 0.5, 0.5])  # 灰色 - others
            else:
                color = np.array([0.0, 0.0, 1.0])  # 蓝色 - 其他
        else:
            # 使用RGB颜色
            color = voxel_colors[x, y, z].astype(np.float32) / 255.0  # 转换为0-1范围
        
        # 计算数组索引
        vertex_start = idx * num_unit_vertices
        vertex_end = vertex_start + num_unit_vertices
        
        # 为每个顶点添加颜色
        all_vertex_colors[vertex_start:vertex_end] = color
    
    # 更新网格颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(all_vertex_colors)


def visualize_voxels_interactive(voxelized_dir, use_semantic_colors=False):
    """
    交互式可视化体素数据
    
    Args:
        voxelized_dir: 体素数据目录路径
        use_semantic_colors: 初始是否使用语义颜色
    """
    if not USE_OPEN3D:
        print("Error: Open3D未安装，无法使用交互式可视化")
        return
    
    # 加载体素数据
    voxel_grid, voxel_colors, voxel_semantics, metadata = load_voxel_data(voxelized_dir)
    
    # 创建初始体素网格
    current_semantic_mode = use_semantic_colors
    mesh, voxel_info = create_voxel_mesh(voxel_grid, voxel_colors, voxel_semantics, metadata, current_semantic_mode)
    
    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="体素可视化（交互式）", width=1920, height=1080)
    
    # 添加网格
    vis.add_geometry(mesh)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    render_option.light_on = True
    
    # 设置视角
    view_control = vis.get_view_control()
    # 计算体素空间的中心点
    voxel_origin = np.array(metadata['voxel_origin_array'])
    voxel_size = metadata['voxel_size']
    voxel_resolution = metadata['voxel_resolution']
    center = voxel_origin + (voxel_size * voxel_resolution) / 2.0
    
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_lookat(center)
    view_control.set_up([0.0, 1.0, 0.0])
    view_control.set_zoom(0.7)
    
    # 定义键盘回调函数
    def toggle_color_mode(vis):
        nonlocal mesh, current_semantic_mode
        current_semantic_mode = not current_semantic_mode
        
        # 更新颜色（不重新创建网格）
        update_voxel_colors(mesh, voxel_info, current_semantic_mode)
        vis.update_geometry(mesh)
        
        mode_str = "语义颜色" if current_semantic_mode else "RGB颜色"
        print(f"已切换到: {mode_str}")
        return False
    
    def reset_view(vis):
        view_control = vis.get_view_control()
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_lookat(center)
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_zoom(0.7)
        print("视角已重置")
        return False
    
    # 注册键盘回调
    vis.register_key_callback(ord('S'), toggle_color_mode)
    vis.register_key_callback(ord('s'), toggle_color_mode)
    vis.register_key_callback(ord('R'), reset_view)
    vis.register_key_callback(ord('r'), reset_view)
    
    # 打印操作说明
    print("\n" + "=" * 60)
    print("交互式体素可视化")
    print("=" * 60)
    print("操作说明：")
    print("  - 鼠标左键拖拽：旋转视角")
    print("  - 鼠标右键拖拽：平移视角")
    print("  - 滚轮：缩放")
    print("  - 'R'：重置视角")
    print("  - 'S'：切换颜色模式（RGB颜色 / 语义颜色）")
    print("  - 'Q' 或关闭窗口：退出")
    print("=" * 60)
    
    if use_semantic_colors:
        print("当前模式: 语义颜色")
        print("  - 红色 = robot arm (类别1)")
        print("  - 灰色 = others (类别2)")
        print("  - 蓝色 = 其他类别")
    else:
        print("当前模式: RGB颜色")
    
    print("=" * 60 + "\n")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()
    
    print("可视化已关闭")


def main():
    parser = argparse.ArgumentParser(description="交互式体素可视化工具")
    parser.add_argument(
        "--voxelized_dir",
        type=str,
        default="/home/qiansongtang/Documents/program/rgb2voxel/data/6cam_dataset/task_0001_user_0016_scene_0001_cfg_0003/voxelized",
        help="体素数据目录路径"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="使用语义颜色而不是RGB颜色"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.voxelized_dir):
        print(f"Error: 体素数据目录不存在: {args.voxelized_dir}")
        return
    
    visualize_voxels_interactive(args.voxelized_dir, use_semantic_colors=args.semantic)


if __name__ == "__main__":
    main()

