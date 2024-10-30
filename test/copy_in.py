import os
import shutil

# 定义源目录和目标目录的相对路径
source_dir = os.path.join('testcase', 'TestTensor')
target_dir = os.path.join('testcase', 'TestSysy')

# 获取当前工作目录（应该是test目录）
current_dir = os.getcwd()

# 构建完整的源目录和目标目录路径
full_source_dir = os.path.join(current_dir, source_dir)
full_target_dir = os.path.join(current_dir, target_dir)

# 确保目标目录存在
if not os.path.exists(full_target_dir):
    os.makedirs(full_target_dir)

# 遍历源目录中的所有文件
for filename in os.listdir(full_source_dir):
    # 检查文件是否以.in结尾
    if filename.endswith('.in'):
        # 构建完整的文件路径
        source_file = os.path.join(full_source_dir, filename)
        target_file = os.path.join(full_target_dir, filename)
        
        # 复制文件
        shutil.copy2(source_file, target_file)
        print(f"Copied: {filename}")

print("All .in files have been copied.")