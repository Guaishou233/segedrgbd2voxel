# 带语义分割的点云生成脚本

## 功能说明

`generate_segmented_pointcloud.py` 用于从 `segmented_dataset` 生成带语义分割信息的点云数据，保存到 `pointed_dataset` 目录。

## 依赖安装

```bash
# 安装 OpenCV
pip install opencv-python-headless

# 安装 Open3D (推荐使用 conda)
conda install -c open3d-admin open3d

# 或者使用 pip (可能需要特定版本)
pip install open3d-python
```

## 使用方法

### 1. 小样本测试（推荐先运行）

先测试少量样本，确保脚本正常工作：

```bash
# 测试前5个样本
python generate_segmented_pointcloud.py --test 5

# 或者使用 --max_samples
python generate_segmented_pointcloud.py --max_samples 5
```

### 2. 基本使用（处理所有数据）

```bash
python generate_segmented_pointcloud.py
```

### 3. 自定义参数

```bash
python generate_segmented_pointcloud.py \
    --input_dir /path/to/segmented_dataset \
    --output_dir /path/to/pointed_dataset \
    --min_depth 0.3 \
    --max_depth 0.8 \
    --depth_scale 1000.0 \
    --downsample 0.001 \
    --test 10  # 先测试10个样本
```

### 参数说明

- `--input_dir`: 输入数据集目录（默认: `data/segmented_dataset`）
- `--output_dir`: 输出点云数据集目录（默认: `data/pointed_dataset`）
- `--min_depth`: 最小深度值（米，默认: 0.3）
- `--max_depth`: 最大深度值（米，默认: 0.8）
- `--depth_scale`: 深度缩放因子，毫米转米（默认: 1000.0）
- `--downsample`: 体素下采样大小（米，默认: 0.001）
- `--use_rgb_color`: 使用RGB颜色而不是语义颜色
- `--test N`: 测试模式，仅处理前N个样本（推荐先使用此参数测试）
- `--max_samples N`: 最大处理样本数，None表示处理所有样本

## 输出格式

生成的点云文件保存在 `pointed_dataset/POINTCLOUD/` 目录下，格式为 PLY 文件。

点云中的颜色编码：
- **灰色** (0.5, 0.5, 0.5): `others` / 背景
- **红色** (1.0, 0.0, 0.0): `robot_arm` / 机器人手臂

## 注意事项

1. 确保所有依赖已正确安装
2. Depth图像是RGB格式存储的灰度图，脚本会自动提取第一个通道
3. 语义分割图是单通道灰度图，标签值为 0 和 255
4. 点云会进行体素下采样和统计离群点过滤
