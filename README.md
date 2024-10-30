# 2024年编译系统设计大赛-编译系统挑战赛-并行优化赛题



### 高校名称：国防科技大学

### 团队名称：这行并行行不行

### 项目名称：sysypc



# 注意：技术报告在[doc](./doc)目录中

## 介绍视频链接 

## 链接: https://pan.baidu.com/s/13V3tsUGfMu0uk_koVxUzRg?pwd=ns4r 提取码: ns4r



# 当前实现的操作

- 创建张量
  - DeclTensorOp
  - DefTensorOp
  - InitTensorOp
- BroadcastOp
- element-wise操作
  - MulOp
  - AddOp
  - SubOp
  - DivOp
- MatMulOp
- ValidConvOp
- SameConvOp
- MaxPoolOp
- ReLUOp
- FlattenOp
- FullyConnectedOp
- TransposeOp
- CopyOp
- 规约操作
  - MaxOp
  - MinOp
  - SumOp
- 用于内联
  - FunctionOp
  - RetOp
  - GenericCallOp
  - CastOp
- ScanOp
- PrintOp
- ClearOp

# 通过cmake构建
```bash
mkdir build
cd build
cmake ..
make -j8
```

# 编译SysY文件
```bash
Usage: ./sysypc [options] <input_file>
Options:
  -o <file>    Specify output file (default: output to console)
  -m           Output mixed MLIR instead of object file
  -l           Output LLVM dialect instead of object file
  -L           Output LLVM IR instead of object file
  -O<level>    Set llvm optimization level (0, 1, 2, 3, default: 0)
  -a           Compile for ARMv7 architecture
  -h           Display this help message
```

# 测试
```bash
1. cd /test
2. python build.py:   创建&&进入到 build 目录, 执行 cmake .. & make -j8
3. python copy_in.py:   将./test/TestTensor/的测试样例输入数据拷贝到./test/TestSysy/ 
4. python compile_library.py [-a]:    编译运行时库，默认为本地架构，-a为armv7架构，下同
5. python compile_object.py [-a]:    编译目标文件
6. python test_official_cases.py [-a]:    测试官方样例
7. python test_all_custom_cases.py [-a]:    测试自定义样例
```

# .vscode/launch.json
用于使用vscode调试代码，需要安装插件：
- C++ Extension Pack
- CMake
- CMake Tools

# 测试样例下载地址
https://gitlab.eduxiji.net/csc1/nscscc/compiler2023/-/tree/master/
