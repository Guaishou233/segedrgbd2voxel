import struct

def parse_ply_header(file_path):
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == "end_header":
                break
    return header


def parse_header_structure(header):
    format_type = None
    elements = {}  # {element_name: {count: x, properties:[...]}}

    current_element = None

    for line in header:
        if line.startswith("format"):
            format_type = line.split()[1]   # ascii / binary_little_endian
        elif line.startswith("element"):
            _, name, count = line.split()
            current_element = name
            elements[current_element] = {
                "count": int(count),
                "properties": []
            }
        elif line.startswith("property"):
            elements[current_element]["properties"].append(line)
    
    return format_type, elements


def inspect_ply(file_path):
    print(f"=== Inspecting PLY: {file_path} ===")

    # 1. 读取 header
    header = parse_ply_header(file_path)
    print("\n--- Header ---")
    for h in header:
        print(h)

    # 2. 解析结构信息
    format_type, elements = parse_header_structure(header)

    print("\n--- Parsed Structure ---")
    print(f"Format: {format_type}")
    for name, info in elements.items():
        print(f"\nElement: {name}, count={info['count']}")
        print("Properties:")
        for p in info["properties"]:
            print("   ", p)

    # 3. 简单展示点云统计信息（仅 ASCII）
    if format_type == "ascii":
        with open(file_path, "r") as f:
            lines = f.readlines()

        end_header_idx = header.index("end_header")
        data_lines = lines[end_header_idx + 1:]

        if "vertex" in elements:
            count = elements["vertex"]["count"]
            pts = data_lines[:count]
            xyz = []
            for pt in pts:
                parts = pt.split()
                try:
                    x, y, z = map(float, parts[:3])
                    xyz.append([x, y, z])
                except:
                    pass

            import numpy as np
            xyz = np.array(xyz)

            print("\n--- Vertex Statistics (ASCII only) ---")
            print(f"Vertex count: {len(xyz)}")
            print("X range:", xyz[:,0].min(), "→", xyz[:,0].max())
            print("Y range:", xyz[:,1].min(), "→", xyz[:,1].max())
            print("Z range:", xyz[:,2].min(), "→", xyz[:,2].max())
        else:
            print("\nNo vertex data found.")

    else:
        print("\nBinary PLY detected (数据读取可以扩展，但结构已正确解析).")



if __name__ == "__main__":
    inspect_ply("/home/qiansongtang/Documents/program/rgb2voxel/data/6cam_dataset/task_0001_user_0016_scene_0001_cfg_0003/POINTCLOUDS_MULTIVIEW/1631270646918.ply")
