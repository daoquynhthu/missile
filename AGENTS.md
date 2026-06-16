# AGENTS 工作区规范

## 项目概述

高保真弹道仿真引擎，C++/CUDA，零第三方 CFD 依赖。核心产品：高超音速飞行器气动表生成 + 6-DOF 弹道仿真。

## 文档体系

| 文件 | 用途 | 何时读 |
|------|------|--------|
| `AERO_ACCURACY_UPGRADE.md` | 架构设计（Stage A/B/C/D/E） | 理解系统设计 |
| `PLAN.md` | 执行计划（任务列表） | 开始工作前，决定下一步做什么 |
| `ISSUES.md` | 活跃阻塞问题 | 遇到阻塞或异常时 |
| `progress.md` | 进度日志（纯时间线） | 完成工作后追加记录 |

**规则**: 不要修改 `AERO_ACCURACY_UPGRADE.md`（只读架构）。`PLAN.md` 可勾选/取消任务。`ISSUES.md` 可新增/关闭 issue。`progress.md` 只追加。

## 技术栈

- C++17, CUDA 12+, CMake
- 编译器: MSVC (Windows), NVCC
- 测试: 自定义宏 (TEST/FAIL/PASS)，无第三方测试框架
- 构建: `cmake -B build` → `cmake --build build`

## 代码约定

- CUDA 核函数命名: `_kernel` 后缀 (如 `fvm_solver_kernel`)
- 命名空间: `AeroSim::Solver`
- 源文件: `src/aero_solver/`, `include/aero_solver/`
- 测试文件: `tests/`
- 新增文件应在 `CMakeLists.txt` 注册

## 代码风格

- 不要添加注释（除非必须解释非直观的算法决策）
- 不要使用 emoji
- 不要创建 .md 文件（除非用户明确要求）
- 遵循现有文件的缩进和命名模式

## 工作流程

1. 读 `PLAN.md` 决定下一步任务
2. 读 `ISSUES.md` 确认当前无阻塞
3. 实现代码改动
4. 编译验证
5. 运行相关测试
6. 追加 `progress.md`
7. 如果遇到新的阻塞问题，更新 `ISSUES.md`
8. 更新 `PLAN.md` 勾选完成的任务

## 关键约束

- 不要提交 git commit（除非明确要求）
- 每个改动后运行编译器验证
- NaN/Inf 检查是必须的
- CUDA 调用后必须检查错误 (`CUDA_CHECK`/`CUDA_KERNEL_CHECK`)
- 所有浮点比较用相对容差，不要用精确相等
