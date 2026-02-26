# 项目交手文档 (Handoff Documentation)

致后继开发者：

本项目目前已完成核心物理层和动力学架构的搭建。以下是项目的关键架构说明及后续开发指南。

## 1. 核心架构说明

### 1.1 命名空间与宏
- 所有核心代码均位于 `AeroSim` 命名空间下。
- **关键宏**：`CUDA_HOST_DEVICE` (定义在 `include/common.hpp`)。
  - **重要原则**：所有需要在 GPU 核函数中调用的函数/方法，必须在头文件中使用此宏声明，并确保其实现不依赖于仅限 CPU 的库。

### 1.2 物理常量
- 所有 WGS-84 常量均在 `include/constants.hpp` 中以 `inline constexpr` 函数形式定义（例如 `Earth::A()`）。
- **注意**：必须使用函数调用形式以确保在 CUDA 设备端的正确链接。

### 1.3 坐标系约定
- **ECEF**: 地心地固坐标系。
- **NED**: 北-东-地坐标系。
- **Body**: 前-右-下（机体坐标系）。
- **姿态表示**：使用四元数 `quat_be` 表示从 ECEF 到 Body 的转换。

## 2. 现有模块细节

### 2.1 重力模型 (`GravityModel`)
- 数据源：`data/EGM2008.gfc`。
- 当前实现：已实现基础框架和 J2/J3 修正。
- **待办项**：目前 CUDA 核函数 `gravity_kernel` 采用的是简化模型。若需更高精度，需将 `gravity_model.cu` 中的递归勒让德多项式计算移入核函数（需注意 GPU 栈深度限制）。

### 2.2 动力学 (`Dynamics6DOF`)
- 积分器：RK4 (Template-based)。
- **架构更新**：已重构 `step_rk4` 为 `integrate_rk4` 模板函数，支持传入 System Dynamics Functor，从而在积分的每个中间步（sub-step）中重新计算引力和气动力，修正了之前将力视为常量的数学错误。
- **待办项**：目前质量变化率 `dot.mass` 默认为 0，需后续对接发动机推力模型。

## 3. 开发建议与提效

### 3.1 构建建议
- **不要频繁运行完整构建**：链接大型可执行文件很慢。
- **使用 `quick_check.ps1`**：它只编译 `gravity_lib` 静态库，能瞬间反馈语法错误。
- **VS Code 提示**：确保安装了 C++ 和 CUDA 扩展。`common.hpp` 的设计已确保 IntelliSense 不会因为 `__host__` 等关键字报错。

## 4. 后续开发优先级 (Roadmap)

1.  **气动特性集成**：
    - 创建气动插值类（Lookup Table），读取 `.csv` 或 `.txt` 格式的气动系数表。
    - 在 `compute_derivatives` 中计算气动力和气动力矩。
2.  **执行机构模型**：
    - 增加发动机推力模型（考虑高度补偿）。
    - 增加舵机动力学模型。
3.  **制导与控制 (G&C)**：
    - 在 `src/main.cpp` 中构建控制循环。
    - 实现基础的比例导引 (PNG) 或现代制导算法。
4.  **环境完善**：
    - 将 NRLMSISE-00 的 C 代码库集成到 `atmosphere_model.cu` 中。

## 5. 常见问题 (FAQ)

- **Q: 为什么 Eigen 报错？**
  - A: 确保在 CMake 中将 Eigen 包含路径设置为 `SYSTEM`。当前的 `CMakeLists.txt` 已处理此问题。
- **Q: GPU 计算结果与 CPU 不一致？**
  - A: 运行 `test_gravity.exe`。如果差异超过 `1e-10`，通常是由于浮点数精度（`double` vs `float`）或常量链接错误。目前项目全线使用 `double`。

祝开发顺利！
