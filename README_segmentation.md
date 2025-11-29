# SAM3语义分割数据集构建说明

## 概述

本脚本使用SAM3模型对RGB图像进行语义分割，识别"robot arm"和"others"两个类别，并构建包含RGB、DEPTH、语义分割mask和相机参数的新数据集。

## 功能特点

1. **使用SAM3进行语义分割**：通过文本提示词"robot arm"进行分割
2. **自动过滤**：如果图像中没有检测到robot arm，则自动跳过该样本
3. **完整数据集构建**：保留RGB、DEPTH、语义分割mask和相机内外参数
4. **测试模式**：支持测试模式，可以只处理少量图像进行验证

## 使用方法

### 基本使用

```bash
conda activate sam3
cd /data/tangqiansong/rgb2voxel
python segment_with_sam3.py
```

### 测试模式

```bash
# 只处理前5张图像进行测试
python segment_with_sam3.py --test --test_num 5
```

### 自定义输入输出目录

```bash
python segment_with_sam3.py --input_dir /path/to/input --output_dir /path/to/output
```

## 参数说明

- `--test`: 启用测试模式，只处理少量图像
- `--test_num N`: 测试模式下处理的图像数量（默认5）
- `--input_dir PATH`: 输入数据集目录（默认: `/data/tangqiansong/rgb2voxel/seg_raw`）
- `--output_dir PATH`: 输出数据集目录（默认: `/data/tangqiansong/rgb2voxel/segmented_dataset`）

## 数据集结构

### 输入数据集结构（seg_raw）
```
seg_raw/
├── RGB/          # RGB图像
├── DEPTH/        # 深度图
└── params/       # 相机参数JSON文件
```

### 输出数据集结构（segmented_dataset）
```
segmented_dataset/
├── RGB/          # RGB图像（已过滤）
├── DEPTH/        # 深度图（已过滤）
├── SEG/          # 语义分割mask（新增）
└── params/       # 相机参数JSON文件（已更新，包含分割信息）
```

## 语义分割mask格式

- **格式**: PNG图像
- **像素值**: 
  - 0 = others（背景/其他物体）
  - 1 = robot arm（机器人手臂）
- **尺寸**: 与原始RGB图像相同

## 相机参数更新

每个参数JSON文件会添加以下字段：
- `has_segmentation`: true
- `segmentation_classes`: ["others", "robot_arm"]
- `segmentation_file`: 对应的分割mask文件名

## 模型加载

脚本会自动尝试以下路径加载SAM3模型：
1. `/data/tangqiansong/Sam3/model` (本地模型，优先)
2. `./checkpoints/sam3`
3. ModelScope自动下载

## 处理进度

脚本会显示：
- 总样本数
- 保留样本数（检测到robot arm）
- 跳过样本数（未检测到robot arm）

## 注意事项

1. 确保已激活`sam3` conda环境
2. 需要GPU支持（CUDA）
3. 处理大量图像可能需要较长时间
4. 如果模型文件不存在，脚本会自动从ModelScope下载

## 当前状态

- ✅ 脚本已创建并测试通过
- ✅ 测试模式验证成功
- 🔄 完整数据集处理正在进行中（7370张图像）

查看处理进度：
```bash
tail -f /data/tangqiansong/rgb2voxel/segmentation.log
```

检查已处理的样本数：
```bash
ls /data/tangqiansong/rgb2voxel/segmented_dataset/SEG/*.png | wc -l
```

