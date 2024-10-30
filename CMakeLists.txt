# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.20)

# 定义项目信息
project(
  "sysypc"
  DESCRIPTION "sysy parallel compiler"
  LANGUAGES CXX C
)

# 设置编译选项
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# 查找 LLVM 和 MLIR 包并输出配置信息
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# 添加 CMake 模块路径
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# 包含必要的 CMake 模块
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 设置 LLVM 链接组件
set(LLVM_LINK_COMPONENTS Support)

# 添加包含目录并输出 LLVM 和 MLIR 目录信息
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/antlr4-runtime)
include_directories(${PROJECT_BINARY_DIR}/lib)
include_directories(${PROJECT_BINARY_DIR}/include)
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

# 添加子目录
add_subdirectory(include)
add_subdirectory(lib)

# 收集源文件
file(GLOB_RECURSE DIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp")
file(GLOB_RECURSE DIR_LIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/*.cpp")

# 添加可执行文件
add_executable(sysypc
  driver.cpp
  ${DIR_SRC}
  ${DIR_LIB_SRC}
)

add_dependencies(sysypc
  SysyOpsIncGen
  SysyCombineIncGen
  SysyShapeInferenceInterfaceIncGen
  SysyEffectInterfaceIncGen
)

# 链接必要的库
target_link_libraries(sysypc
  PRIVATE
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRCallInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRPass
  MLIRControlFlowDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRSupport
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRVectorDialect
  MLIROpenMPDialect
  MLIRAffineUtils
  MLIRAffineAnalysis
  MLIRVectorToLLVM
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIRIndexToLLVM
  MLIROpenMPToLLVMIRTranslation
  LLVMCore
  LLVMSupport
  LLVMTransformUtils
  LLVMMC
  LLVMScalarOpts
  LLVMInstCombine
  LLVMAggressiveInstCombine
  LLVMAnalysis
  LLVMTarget
  LLVMOption
  LLVMPasses
  LLVMIRReader
  LLVMVectorize
  LLVMARMCodeGen
  LLVMAArch64CodeGen
  LLVMX86CodeGen
  LLVMARMAsmParser
  LLVMAArch64AsmParser
  LLVMX86AsmParser
)

# 移除不需要的编译选项
string(REPLACE "-fno-lifetime-dse" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fno-lifetime-dse" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")