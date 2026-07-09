# AeroHighPrecisionSim — 工程规格说明书

## 1. 产品定位

统一气动模拟平台。覆盖以下场景于同一套引擎：

| 场景 | Mach 范围 | 用途 |
|------|-----------|------|
| 低速空气动力学 | 0.01–0.3 | 无人机、竞技弹丸、风洞对标 |
| 跨/超声速 | 0.8–5.0 | 作战飞行器、导弹 |
| 高超声速 | 5.0–25 | 滑翔飞行器、再入体 |
| 稀薄气体/高高层 | >25 | 卫星再入、高超声速稀薄效应 |

所有场景共享同一套物理基础（引力/大气/坐标/6-DOF），可选择不同的气动求解器（工程估算、面板法、FVM CFD）。

## 2. 词汇规范

禁止出现具体平台名称作为代码标识符。以下为替换映射：

| 原词汇 | 替换 | 说明 |
|--------|------|------|
| `missile` | `flowsim` (仓库名) / `aerosp` (命名空间) | 仓库顶级命名 |
| `missile_config` | `config/vehicle_config` | 场景配置 |
| `MissileDesign` | `config` | 命名空间 |
| `HGV1Config` | `SimConfig` 或 `VehicleConfig` | 飞行器配置 |
| `rm_dart_*` / `RM` | 移入 `examples/dart/` | 低速验证案例 |
| `AeroSim` (命名空间) | `aerosp` | 新命名空间 |
| `AeroSim::Cfd` | `aerosp::aero::cfd` | CFD 子空间 |
| `AeroSim::GNC` | `aerosp::control` | 制导控制 |
| `AeroSim::Solver` | `aerosp::aero::panel` | 面板法 |

## 3. 目录结构

```
flowsim/
├── CMakeLists.txt              # 顶层入口
├── cmake/                      # CMake 模块
│   ├── compiler_flags.cmake   # 编译器标志配置
│   └── fetch_eigen.cmake      # Eigen FetchContent
├── src/                        # 源码（按领域分层）
│   ├── infra/                  # 零依赖基础设施
│   │   ├── math/              # Real, Vec3, Quat, constants
│   │   ├── util/              # PID, progress_bar, diagnostics
│   │   └── cuda/              # CUDA_CHECK, launch_config
│   ├── physics/                # 仿真物理层
│   │   ├── coord/            # 坐标变换 LLA/ECEF/NED/Body/ECI
│   │   ├── gravity/          # 引力场 (EGM2008, J2/J3/J4)
│   │   ├── atmosphere/       # 大气 (USSA76 ≤86km, NRLMSISE-00 全高)
│   │   ├── propulsion/       # 推进 (固体火箭、RCS、TVC)
│   │   ├── dynamics/         # 动力学 (6-DOF、RK4)
│   │   └── control/          # 制导与控制 (autopilot、导引律)
│   ├── aero/                   # 气动求解
│   │   ├── engineering/    # 工程估算 (DATCOM、van Driest II)
│   │   ├── panel/          # 牛顿撞击面板法 (GPU)
│   │   └── cfd/            # FVM 求解器 (Euler/NS/SA RANS)
│   │       ├── core/       # 求解器循环、边界条件
│   │       ├── scheme/     # MUSCL、limiter、Riemann 求解器
│   │       ├── physics/    # 粘性通量、湍流源项
│   │       └── gpu/        # GPU 核函数
│   └── config/                # 输入配置 + 场景定义
├── app/                        # 可执行文件入口
│   ├── sim_main/              # 主仿真入口 (原 main.cpp)
│   ├── aero_calc/             # 命令行气动计算工具
│   ├── aero_table_gen/        # 气动表生成器
│   └── shape_optimizer/       # 外形优化工具
├── examples/                   # 场景示例/验证案例
│   ├── hgv/                   # HGV 高超声速弹道
│   └── dart/                  # 低速竞技弹丸验证 (Dart heritage)
├── data/                       # 输入数据
│   ├── examples/hgv/
│   └── examples/dart/
├── scripts/                    # Python 脚本
│   ├── examples/hgv/
│   └── examples/dart/
└── tests/                      # 测试 (保持原位)
```

## 4. 构建系统

### 4.1 语言支持

```
LANGUAGES CXX CUDA Fortran
```

Fortran: `gfortran` (MinGW-W64) 编译 `nrlmsise00.f` 为静态库。CMake 原生编译，不提交 .dll/.lib 到仓库。

### 4.2 构建目标

| 目标名 | 类型 | 描述 |
|--------|------|------|
| `aerosp` | INTERFACE 库 | 核心库的所有头文件 |
| `aerosp::nrlmsise00` | STATIC 库 | NRLMSISE-00 Fortran 编译 |
| `aerosp::aerosp` | STATIC 库 | 所有 pkgs 代码 |
| `aerosp::sim` | 可执行文件 | 主仿真 (原 AeroSim) |
| `aerosp::aero_calc` | 可执行文件 | 命令行气动计算 |
| `aerosp::table_gen` | 可执行文件 | 气动表生成 |
| `aerosp::shape_opt` | 可执行文件 | 外形优化 |
| `aerosp::python_bridge` | 共享库 | Python ctypes 接口 |

### 4.3 第三方依赖

- CUDA 12+ (sm_75, RDC ON)
- Eigen 3.4 (FetchContent)
- **gfortran** (MinGW-W64, 系统安装) — 仅用于 NRLMSISE-00

### 4.4 精度

`cmake -DAEROSIM_REAL_DOUBLE=ON` 切换 Real 精度。默认 float。

## 5. 代码约定

### 5.1 文件命名与位置

- 头文件: `.hpp` (C++), `.cuh` (CUDA 头)
- 源文件: `.cpp` (纯 CPU), `.cu` (含 GPU 核函数)
- Fortran 源码: `.f` (置于 `src/sim/atmosphere/`)
- 文件路径 ≈ 命名空间路径。`src/sim/coord/coordinate_transform.cpp` → `aerosp::sim::coord`

### 5.2 命名规则

- 命名空间: 全小写 (`aerosp::sim::coord`)
- 类/结构体: `PascalCase`
- 函数/变量: `snake_case`
- 宏/常量: `UPPER_SNAKE_CASE`
- CUDA 核函数: `_kernel` 后缀
- 成员变量: `trailing_underscore_`

### 5.3 强制校验

- CUDA 调用后必须 `CUDA_CHECK` / `CUDA_KERNEL_CHECK`
- 所有浮点比较用相对容差，不用精确相等
- NaN/Inf 检查必须在求解器关键路径上

### 5.4 禁止

- 禁止注释（除非解释非直观算法决策）
- 禁止 emoji
- 禁止二进制文件提交到仓库（库/构建产物）
- 禁止 `file(GLOB_RECURSE)`

## 6. 测试

- 框架: 自定义 `TEST`/`FAIL`/`PASS` 宏 (无第三方测试框架)
- CTest 注册
- 测试套件:
  - 物理层: TestPropulsion, TestAero, TestGuidance, TestAutopilot, TestLaunch, TestIntegrator, TestAeroTableGen, TestAeroViscous
  - CFD 套件: TestCfdGpu, TestCfdMesh, TestCfdEuler, TestCfdViscous, TestCfdReconstruction, TestCfdDiagnostics
  - 大气: 新增 NRLMSISE-00 验证测试 (原 `compare_atm.cpp` 纳入构建)

## 7. 命名空间 — 完整映射

```
aerosp                                  顶层
aerosp::infra::math                     数学基础设施
aerosp::infra::util                     通用工具
aerosp::infra::cuda                     CUDA 工具
aerosp::sim::coord                      坐标变换
aerosp::sim::gravity                    引力模型
aerosp::sim::atmosphere                 大气模型
aerosp::sim::propulsion                 推进
aerosp::sim::dynamics                   动力学
aerosp::sim::control                    制导控制
aerosp::aero::cfd                        FVM 求解器
aerosp::aero::panel                      牛顿面板法
aerosp::aero::eng                        工程估算
aerosp::config                          场景配置
```

命名空间 = 目录路径。`#include` 路径 = 命名空间路径（`infra/math/real.hpp` → `aerosp::infra::math`）。

## 8. 大气系统架构

### 8.1 层次化查询

```
AtmosphereModel::calculate(alt, ...)
│
├─ alt ≤ 86 km → USSA76 (7层递推，精确)
│
└─ alt > 86 km → NRLMSISE-00 (Fortran 编译静态库)
                  │
                  └─ CUDA_ARCH 路径 → 指数律外推 (降级保护)
```

### 8.2 性能策略

- 弹道积分主循环: USSA76（~100x 快于 NRLMSISE-00）
- 后处理/精确分析: NRLMSISE-00
- 设计决策: 接口统一（`calculate` + `calculate_ussa76`），调用方按场景选择

## 9. NRLMSISE-00 构建

### 9.1 源码管理

- 提交文件: `src/sim/atmosphere/nrlmsise00.f`（Fortran 固定格式源码）
- 不提交: `.lib` / `.dll` / `.o`（构建产物）

### 9.2 CMake 集成

```cmake
enable_language(Fortran)
add_library(nrlmsise00 STATIC
    src/sim/atmosphere/nrlmsise00.f)
target_link_libraries(aerosp PRIVATE nrlmsise00)
```

系统需要 MinGW-W64 gfortran 在 PATH 上。

### 9.3 GPU 回退

`AtmosphereModel::calculate(NRLMSISE00Input)` 的 `#if !defined(__CUDA_ARCH__)` 路径保留 Fortran 调用；设备路径回退 USSA76。

## 10. 关键工程决策

| # | 决策 | 内容 |
|---|------|------|
| D1 | 源码目录 | 不搞多层 `pkgs/`，保留 `src/` + `include/`，内部按 `sim/coord/` 分层 |
| D2 | NRLMSISE-00 | Fortran 源码提交，CMake 编译，不提交 .lib/.dll。gfortran 系统依赖。 |
| D3 | 场景无关词汇 | 代码中不出现 `missile`/`dart`/`RM`。具体案例放 `examples/`。命名空间用 `aerosp::` |
| D4 | 无 GLOB_RECURSE | 显式 `target_sources`，每个模块的 CMakeLists.txt 列自己 |
| D5 | 无强制 NVCC 编译 | 纯 C++ 文件只通过 CXX 编译器 |
| D6 | Git 历史 | `git mv` 保留完整历史 |

## 11. 迁移计划

分 5 步，每步独立 commit，保留历史：

### Step 1: Fortran 工程化
- `src/nrlmsise00.f` → `src/sim/atmosphere/nrlmsise00.f`
- CMake 添加 Fortran 编译
- 移除 `.dll`/`.lib` 的提交
- `compare_atm.cpp` 纳入构建为测试

（此步先做，之后按顺序：目录重组 → CMake 子目录化 → 命名空间改名 → 清理品牌词汇）

### Step 2: 目录重组
- `src/*` → `src/{infra,sim,aero,config}/`
- `include/*` → `include/infra/` + `include/sim/` + `include/aero/`
- `app/` 新建，接收可执行入口
- `examples/` 新建，接收 dart/HGV 案例

### Step 3: CMake 子目录化
- 单 CMakeLists.txt → 每模块一个 `CMakeLists.txt`
- `cmake/` 目录放 CMake 模块

### Step 4: 命名空间改名
- `AeroSim::*` → `aerosp::*`
- Include guard 同步

### Step 5: 品牌清理
- 文档更新（AGENTS.md, PLAN.md, README.md）
- 根目录文件清理
- 验证构建 + 测试

## 11. 文件索引 (根目录竣工状态)

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 构建入口 |
| `AGENTS.md` | 工作区规范 (根目录) |
| `LICENSE` | GPL 3.0 许可 (根目录) |
| `docs/REPO_SPEC.md` | 本文件 (工程规格，只读) |
| `docs/AERO_ACCURACY_UPGRADE.md` | CFD 精度升级架构 (只读) |
| `docs/PLAN.md` | 执行计划 |
| `docs/ISSUES.md` | 问题追踪 |
| `docs/progress.md` | 进度日志 (仅追加) |
| `README.md` | 项目介绍 (根目录) |
| `.gitignore` | Git 忽略规则 |
| `LICENSE` | 许可 |
| `cmake/` | CMake 模块目录 |