# AeroSim 仿真管线

仓库中现有文件的完整数据流梳理。

---

## 现有文件清单

```
STL 几何文件:
  data/missile/hgv_model_optimized.stl   ← HGV 气动外形
  data/dart/rm_dart_model.stl             ← RM 飞镖 v3
  data/dart/rm_dart_v4_square_capsule.stl ← RM 飞镖 v4
  data/dart/rm_dart_v5_full_square.stl    ← RM 飞镖 v5

气动系数表:
  aerodynamics_table.csv            ← HGV: AeroCalc.exe 从 STL 生成 (239条, 17 Mach × 14 Alpha)
  data/dart/rm_dart_aero_table.csv  ← Dart: generate_dart_aero_table.py 解析法生成

配置文件:
  data/missile/hgv_config_optimized.json ← HGV 配置 JSON
```

---

## 正确管线（应当是这样的）

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. 几何设计                                                         │
│  generate_cone_waverider.py  →  STL 文件                             │
│  optimize_shape.py           →  hgv_model_optimized.stl              │
│  generate_rm_dart_model.py   →  rm_dart_model.stl                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. 气动系数计算                                                      │
│  STL 文件 + AeroCalc.exe (GPU Newtonian Impact Solver)  →  CSV 表    │
│  遍历 Mach×Alpha 网格 → aerodynamics_table.csv                       │
│  (这是唯一从 STL 推导气动数据的路径)                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. 仿真运行时加载 CSV 表                                             │
│  AerodynamicsModel.load_csv_table(aerodynamics_table.csv)            │
│  → BilinearInterpolation(Mach, Alpha) → CX/CY/CZ/Cl/Cm/Cn           │
│  → compute_forces_moments() → 6-DOF 动力学                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. 6-DOF 仿真                                                       │
│  main.cpp / rm_dart_sim.cpp                                          │
│  RK4 积分 + EGM2008 重力 + NRLMSISE-00 大气 + 制导/飞控               │
│  → 输出 trajectory.csv                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 实际管线（目前的状态）

```
STL 文件 ──→ AeroCalc.exe ──→ aerodynamics_table.csv ──→ AerodynamicsModel
 (存在)     (可用)             (存在, 239点)              ↓
                                                        │
                 missile_config.cpp:141-200              │
                 解析法生成 cl_table_2d, cd_table_2d     │
                 cm_table_2d (9Mach × 9Alpha)           │
                        ↓ (从未被使用)                    │
                  AerodynamicsModel.compute_coeffs()     │
                        ↓                                │
                 compute_coeffs_analytical()              │
                 → cd0=0.1, cl_alpha=2.0 (硬编码)        │
                 → 忽略表格                                │
```

---

## 核心问题

### 问题 1: 两条气动路径并存但都不完整

**路径 A（CSV 表）**:
- ✅ CSV 文件存在（`aerodynamics_table.csv`），由 AeroCalc 从 STL 生成
- ✅ `AerodynamicsModel::compute_coeffs_table()` 实现了完整的双线性插值
- ✅ `load_csv_table()` 解析器正常工作
- ❌ `missile_config.cpp:125` 路径指向 `"e:/missile/aerodynamics_table.csv"` — 硬编码
- ❌ 运行时 CSV 文件可能不存在（取决于工作目录）

**路径 B（解析法 + 内存表）**:
- ❌ `missile_config.cpp:141-200` 生成了 `cl_table_2d/cd_table_2d/cm_table_2d`
- ❌ `AerodynamicsModel::compute_coeffs_analytical()` 永远不读这些表
- ❌ 分析回退使用 `cd0=0.1, cl_alpha=2.0`（`aerodynamics_model.hpp:308-309`）
- ❌ 添加了假插值兜底（`cd0 = m_config.cd0_table[0]` 永远取第一个元素，带 `// TODO: Interpolate`）

两条路径互相不知道对方的存在，且路径 B 的数据生成是死代码。

### 问题 2: STL → CSV 的生成环节需要 AeroCalc.exe

`generate_aero_table.py` 的流程:
1. 遍历 Mach×Alpha 网格（238 个点）
2. 对每一点调用 `AeroCalc.exe <stl> <mach> <alpha> ...` 子进程
3. 从 stdout 中提取 JSON 结果
4. 汇总写入 CSV

这意味着:
- 生成气动表的前置条件是有编译好的 `AeroCalc.exe`
- `AeroCalc.exe` 调用了 GPU AeroSolver，需要 CUDA 环境
- 如果 GPU 不可用，整个 CSV 表无法重新生成
- 这是正确的流程，但依赖链脆弱

### 问题 3: Dart 系统的气动表不需要 STL

`generate_dart_aero_table.py` 使用纯解析公式计算亚音速细长体气动系数，不需要 STL 文件。Dart 的 STL 文件仅用于几何验证/CFD，不参与运行时气动计算。

---

## 正确使用流程

```
# 如果你有新的 STL 模型:
python scripts/missile/generate_aero_table.py
  → 输入: data/missile/hgv_model_optimized.stl
  → 调用: build/bin/Release/AeroCalc.exe (GPU CUDA 牛顿撞击法)
  → 输出: aerodynamics_table.csv

# 然后修改仿真配置指向新的 CSV:
missile_config.cpp:124-126:
  aero.use_csv_table = true;
  aero.csv_path = "aerodynamics_table.csv";  // 或相对路径

# 运行仿真:
build/bin/Release/AeroSim.exe
  → 加载 CSV 表 → bilinear interpolate → 6-DOF
```

Dart 的流程类似但不经过 AeroCalc:

```
python scripts/dart/generate_dart_aero_table.py
  → 解析法计算系数
  → 输出: data/dart/rm_dart_aero_table.csv

rm_dart_sim.cpp:
  → 加载 data/dart/rm_dart_aero_table.csv
  → DartAeroTable 做插值
  → 6-DOF 仿真
```

---

## 问题 4: CSV 气动表数据本身是无效的（最严重）

即使打通了 CSV 加载路径，`aerodynamics_table.csv` 的数据也是错误的。根源在 GPU AeroSolver 的 `aero_solver.cu`。

### 4a. 牛顿撞击法在亚/跨/超音速下无效

`aero_solver.cu:34-88` 实现的 Newtonian Impact 方法**只在 Mach > 5 的高超声速下成立**，但 CSV 表的 Mach 网格起点是 0.5。

看实际数据（Mach=0.5, 各Alpha）:
```
Alpha   CD      CY        Cn         备注
-10°   16.19   -24.19    -132.6      ❌ 对称体 beta=0, CY/Cn 应为 0
  0°    2.92     0.00       0.0      ✅ 对称正确
+10°   16.19    24.19     132.6      ❌ CY/Cn 与 -10° 正负对称但量级荒谬
```

**逐项问题**:
1. **CD=2.92（Mach=0.5, α=0°）**: 真实高超声速飞行器的零升阻力系数约 0.02-0.05，CSV 表大 **100-150 倍**。这是把牛顿撞击法的 `Cp_max=2` 直接当成了全机 CD。
2. **CY/Cn 不为零（β=0°）**: 对称体在零侧滑时侧力和偏航力矩应为零。STL 模型对称，但力和力矩不对称。这是 GPU kernel 中的 STL 法线方向或力矩参考点问题。
3. **L/D=13（Mach=0.5, α=2°）**: 亚音速下不可能。最优滑翔机 L/D~60，但导弹尺寸的 L/D 在 Mach 0.5 下约 5-10。但这里的 L/D 包含错误的 CD，所以看似合理是巧合。

### 4b. AeroSolver 核函数存在根本性缺陷

`aero_solver.cu:50`:
```cuda
float3 flow_dir = make_float3(-ca * cb, -sb, -sa * cb);
```
这一行正确地将来流方向定义在体坐标系下（从头部指向尾部）。但是：

`aero_solver.cu:78`:
```cuda
float3 force_term = tri.normal * (-Cp * tri.area);
```
力的方向是 `-Cp × 法线`。压缩侧 `Cp>0`，力指向 `-法线`（压入表面），方向正确。但 132K 的 Cy 值说明法线方向或 STL 模型本身有问题。

**问题根源**: 力矩参考点固定在原点(0,0,0)（`aero_solver.cu:85`），而 STL 模型的几何原点在头部（`missile_config.cpp:27` `x_nose = 0.0`）。力矩臂巨大导致力矩系数量级完全错误。

### 4c. AeroCalc.exe 无法运行

即使在正确的气动条件下，AeroCalc.exe 也依赖编译好的 CUDA binary 和 GPU（sm_75）。macOS/Linux 或无 NVIDIA GPU 的 Windows 机器无法生成 CSV 表。

---

## 问题 5: Dart 气动表严重不足

`data/dart/rm_dart_aero_table.csv` 只有 **2 条记录**（Mach=0.1, Alpha=±3°），31 格的 Alpha 网格和 6 格 Mach 网格全部缺失。

而 `DartAeroTable::get_coeffs()` 做的双线性插值面对 2 个数据点等价于常数映射，永远返回 ±3° 的那两个值，不管输入什么 Mach/Alpha。

Dart 高保真仿真 `rm_dart_sim.cpp:219` 实际使用的是 `DartAeroTable::get_coeffs()`（从 CSV 加载），但回退的解析法 `rm_dart_aero.hpp` 也有问题。

---

## 问题 6: 三个气动计算方法互相独立且全部不可用

| 方法 | 文件 | 真实度 | 问题 |
|------|------|--------|------|
| GPU 牛顿撞击法 | `aero_solver.cu` | ❌ 不可用 | (1) 牛顿法只在 Mach>5 有效 (2) 力矩参考点错误 (3) CY/Cn 不为零 |
| CSV 查表 | `aerodynamics_table.csv` | ❌ 数据错误 | 由 GPU 求解器生成，继承了上述所有问题 |
| 解析法回退 | `aerodynamics_model.hpp:302-336` | ❌ 硬编码 | `cd0=0.1, cl_alpha=2.0` 与飞行器尺寸/重量无关 |
| 导弹配置生成 | `missile_config.cpp:141-200` | 💀 死代码 | `cl_table_2d/cd_table_2d/cm_table_2d` 生成后永不消费 |
| Dart 解析法 | `rm_dart_aero.hpp` | ⚠️ 简化 | 线性气动，无 stall，`CD0=0.35` 偏大 |
| Dart CSV 表 | `rm_dart_aero_table.csv` | ❌ 数据不足 | 仅 2 个数据点，插值退化为常量 |

---

## 问题 7: SubsonicSolver GPU 代码从未验证

`subsonic_solver.cu:914` 实现了完整的 GPU 亚音速求解器（切片法 + 耦合迭代），但：
1. **无验证**: 没有与风洞数据或 CFD（OpenFOAM/ANSYS）的对比
2. **经验参数**: `coupling_gain=0.16, slice_core_scale=0.55, suction_factor=1.10, backflow_factor=-0.16` 靠猜
3. **自动回退**: `build_geometry_data()` 中如果 STL 加载失败（`subsonic_solver.cu:213`），自动生成一个虚拟圆柱体作为几何体，静默产生假数据
4. **收敛伪造**: `evaluate_batch_kernel:497` — `result.converged = 1; result.residual = entry.residual_target * 0.2;` 直接写收敛标志和残差，不是实际收敛判定

---

## 修复优先级（更新版）

| 优先级 | 问题 | 文件 | 影响 |
|--------|------|------|------|
| 🔴 P0 | CSV 气动表数据错误 | `aero_solver.cu:50-85` | 整个 HGV 仿真的气动系数是垃圾 → 射程、弹道全部不可信 |
| 🔴 P0 | 力矩参考点错误 | `aero_solver.cu:85` | 力矩系数 (Cm) 量级错误 → 配平和稳定性计算全错 |
| 🟠 P1 | Newtonian 方法在亚音速下无效 | 牛顿法仅限 Mach>5 | 全速域仿真中低马赫段气动不准 |
| 🟠 P1 | Dart CSV 表只有 2 条数据 | `rm_dart_aero_table.csv` | 插值退化为常量，双线性插值形同虚设 |
| 🟡 P2 | 解析法硬编码 | `aerodynamics_model.hpp:308-309` | HGV 主仿真气动数据与几何无关 |
| 🟡 P2 | AeroCalc.exe 无法在无 GPU 环境运行 | 依赖 CUDA sm_75 | 气动表无法重新生成 |
| 🟢 P3 | SubsonicSolver 收敛标志伪造成 | `subsonic_solver.cu:497` | 信任收敛判断的代码会出错 |
| 🟢 P3 | SubsonicSolver STL 加载失败静默回退 | `subsonic_solver.cu:213-232` | 几何加载失败 → 假圆柱体 → 用户不知情 |
| 🟢 P3 | 解析法 `Cd0=0.35` 偏高 | `rm_dart_aero.hpp:43` | Dart 阻力偏大 → 射程偏短 |
