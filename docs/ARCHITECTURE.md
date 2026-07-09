# AeroSim 最终管线设计

**目标**: 一条完整、正确、高度精密的仿真管线。无平行路径，无死代码，分级验证。

---

## 设计原则

1. **数据单向流动**: 几何 → 气动 → 仿真 → 后处理，禁止反馈环和数据重复
2. **GPU 做它擅长的**: 大规模并行独立计算（每 Mach×Alpha 点、每三角形面元）
3. **Python 不做运行时计算**: 只在离线阶段做编排、ODE 求解、几何生成
4. **C++/CUDA 做运行时**: 气动插值、6-DOF 积分、制导控制——每步都要跑，性能关键
5. **级间验证**: 每个阶段的输出必须有可量化的正确性检查，不合格不允许进入下一阶段

---

## 总图

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 1: 几何生成 (离线, Python)                                         │
│                                                                          │
│  输入: 设计参数 (长度, 半锥角, 马赫数...)                                 │
│  工具: waverider_physics.py (Taylor-Maccoll ODE)                         │
│        generate_cone_waverider.py (网格生成)                              │
│  输出: data/missile/hgv_model.stl (单一权威几何)                          │
│  验证: STL 合法 (水密/法线向外/流形)                                     │
└───────────────────────────────────────────┬──────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 2: 气动系数生成 (离线, GPU + Python 编排)                          │
│                                                                          │
│  2a. 高超声速段 (Mach > 5)                                               │
│      工具: AeroCalc.exe (C++/CUDA, 牛顿撞击法)                           │
│      输入: hgv_model.stl + Mach×Alpha 网格                               │
│      输出: CSV 子表 (高超声速部分)                                        │
│      验证: 与 Cone 理论 Cp 对比, 与已知基准比对                           │
│                                                                          │
│  2b. 亚/跨/超声速段 (Mach 0.1 ~ 5)                                      │
│      工具: generate_aero_table_engineering.py (Python, 工程估算)          │
│            或 AeroCalc.exe 切换到工程估算模式 (C++, component buildup)     │
│      方法: DATCOM 组件组合法 (机身 + 翼面 + 尾翼)                         │
│      输入: 几何参数 (长细比, 翼展, 尾翼尺寸...)                          │
│      输出: CSV 子表 (低马赫部分)                                          │
│                                                                          │
│  2c. 合并与验证                                                          │
│      工具: merge_aero_tables.py (Python)                                 │
│      输出: data/missile/aerodynamics_table.csv (完整, 验证通过)           │
│      验证: CD(α=0), CL_α, Cm_α 在 Mach 过渡处连续                        │
│            对称体在 β=0 时 CY=Cn=0                                       │
│            导出 CSV 文件头格式正确                                        │
└───────────────────────────────────────────┬──────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 3: 6-DOF 仿真 (运行时, C++/CUDA, 零 Python 依赖)                   │
│                                                                          │
│  主入口: build/bin/Release/AeroSim.exe                                   │
│  配置: data/missile/hgv_config.json (JSON, 运行时加载)                    │
│                                                                          │
│  数据流:                                                                  │
│  aerodynamics_table.csv ─→ AerodynamicsModel.load_csv_table()             │
│       ↓  双线性插值 (CUDA_HOST_DEVICE, 可在 GPU 运行)                     │
│  compute_forces_moments() ─→ ForcesMoments                                │
│       ↓                                                                   │
│  Dynamics6DOF.compute_derivatives()                                       │
│       ↓                                                                   │
│  Guidance.update() ─→ 目标姿态                                           │
│  Autopilot.update() ─→ TVC/RCS/Aero 指令                                 │
│       ↓                                                                   │
│  integrate_rk4() ─→ 新状态                                               │
│       ↑                                                                   │
│  GravityModel (EGM2008, GPU batch)                                        │
│  AtmosphereModel (NRLMSISE-00/USSA76)                                    │
│       ↓                                                                   │
│  日志: output/trajectory.csv                                              │
│                                                                          │
│  运行时路径解析 (非硬编码):                                                │
│    - 可执行文件同级目录下 data/ 子目录                                     │
│    - 或通过命令行参数 --data-dir 指定                                     │
│    - CMake 编译时写入 DATA_DIR 宏                                         │
└───────────────────────────────────────────┬──────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Phase 4: 后处理与优化 (离线, Python)                                     │
│                                                                          │
│  单弹道分析: analyze_trajectory.py                                       │
│    输入: output/trajectory.csv                                           │
│    输出: 高度曲线/速度曲线/射程/L/D/热流                                  │
│                                                                          │
│  蒙特卡洛打靶: monte_carlo.py                                            │
│    输入: AeroSim.exe (multirun mode)                                     │
│    输出: 散布椭圆/CDF/命中概率                                            │
│                                                                          │
│  参数优化: tune_parameters.py                                            │
│    方法: 差分进化 (scipy)                                                │
│    子进程: AeroSim.exe (每次评估跑一条弹道)                               │
│    目标: 最大化射程 + 约束 (最高点<100km, 不坠毁)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 相变细节

### Phase 1: 几何生成

```
输入参数:
  body_length = 12.0 m
  body_radius = 0.6 m
  nose_fineness = 3.0 (鼻锥长细比)
  fin_span = 3.0 m
  design_mach = 15.0

生成步骤:
  1. waverider_physics.py
     → solve_ivp 求解 Taylor-Maccoll ODE
     → 输出 θ_cone, Vr(θ), Vθ(θ)
  2. generate_cone_waverider.py
     → 构造乘波体下表面 (流线追踪)
     → 构造上表面 (自由流面)
     → 构建网格 → numpy → 二进制 STL
  3. 验证:
     → STL 水密性: 每条边恰好被两个三角形共享
     → 法线指向: 全部 outward
     → 几何尺寸: 与设计参数一致 (±1mm)
```

### Phase 2: 气动系数生成

#### 2a. GPU 高超声速求解器 (修复后)

`aero_solver.cu` 需要的修复（重要，这是当前数据的根本问题）：

```
修复 1: 力矩参考点
  当前: moment_term = cross(tri.center, force_term)     // 绕 (0,0,0)
  修复: moment_term = cross(tri.center - ref_point, force_term)  // 绕质心

修复 2: Cp 公式 (Mach 修正)
  当前: Cp = 2.0 * dot_val²
  修复: 
    Cp_max = (γ+1)/(γ-1) * [1 - 1/(γ*Mach²)]  ← 含 Mach 效应
    或 Prandtl-Meyer 扩展: 膨胀侧 Cp 用 -1/(γ*Mach²)
    保留: Cp = Cp_max * dot_val² (压缩侧)
          Cp = -1/(γ*Mach²) (膨胀侧, 当前已有)

修复 3: 对称性检查
  在 CPU 端求和后验证:
    if |β| < 1e-6: assert |CY| < 1e-3, |Cn| < 1e-3
```

网格规格 (覆盖 HGV 飞行包线):

```
Mach 网格: [5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
Alpha 网格: [-5, 0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30] (度)
Beta 网格: [0] (对称体, 仅配平用, 可用小扰动法)
```

#### 2b. 低马赫段工程估算 (Mach 0.5 ~ 5)

对于 Mach < 5 的段，牛顿撞击法不成立。需要改用**组件组合法**（类似 DATCOM/Axisymmetric Missile DATCOM）：

```
对每个 Mach × Alpha 点:
  1. 机身 (Body)
     → 法向力: Cn_body = Cn_α * sin(α)cos(α)  (细长体理论)
     → 轴向力: Ca_body = Cf * Swet / Sref + Cabase
     → 压心: xcp_body ≈ 0.45~0.55 * L_body (亚音速→超音速后移)
     
  2. 翼面/尾翼 (Fins)
     → 法向力: 等效翼理论, 考虑马赫数影响的下洗/遮挡
     → 轴向力: 波阻 (超音速) + 摩阻
     
  3. 舵面效率
     → Cm_δ, Cn_δ: 舵偏转引起的力矩系数
     
  4. 部件干扰修正
     → 机身/尾翼 interference
```

**实现位置**: 可以在 `missile_config.cpp` 中实现（纯 C++），或改为 Python 脚本。推荐 C++ 实现，因为参数固定、离线运行一次后输出 CSV，以后运行时不再需要这些公式。

#### 2c. 合并验证

合并器确保:
1. Mach=5 处高超声速/工程估算结果连续 (C0 连续, 最好 C1 连续)
2. α=0, β=0 时 CY=Cn=Cl=0 (对称约束)
3. CD 单调递增随 Mach 增加 (物理合理)
4. 所有系数在物理范围内

### Phase 3: 运行时仿真

配置文件 `data/missile/hgv_config.json`:

```json
{
  "aero_table": "data/missile/aerodynamics_table.csv",
  "gravity_file": "data/EGM2008.gfc",
  "atmosphere": "nrlmsise00",
  "stl_model": "data/missile/hgv_model.stl",
  "initial_state": {
    "lat_deg": 40.96,
    "lon_deg": 100.30,
    "alt_m": 1000.0,
    "heading_deg": 90.0
  },
  "guidance": {
    "boost_end_time": 50.0,
    "glide_aoa_bias": 12.0,
    ...
  }
}
```

运行命令:
```powershell
# 单次运行
AeroSim.exe --config data/missile/hgv_config.json --output output/trajectory.csv

# 蒙特卡洛
AeroSim.exe --config data/missile/hgv_config.json --monte-carlo 1000 --output output/mc_dir/
```

---

## 气动数据验证清单

每个 Phase 2 输出的 CSV 文件必须通过以下检查才能进入 Phase 3:

| 检查项 | 方法 | 通过标准 |
|--------|------|----------|
| 对称性 | β=0 列检查 | max(|CY|, |Cn|, |Cl|) < 1e-6 |
| 零升阻力合理性 | CD(α=0) | 0.02 ~ 0.15 (取决于 Mach) |
| 最大升阻比 | max(CL/CD) | 2.0 ~ 5.0 (高超声速滑翔体) |
| Cm_α 稳定性 | ∂Cm/∂α < 0 | 全 Mach 域负值 (静稳定) |
| 连续性 | Mach 相邻行检查 | 系数变化 < 20% (无跳变) |
| 网格覆盖 | min/max 检查 | Mach 覆盖仿真包线 + 裕度 |
| 文件完整性 | 行数验证 | total_rows = nMach × nAlpha × nBeta |

---

## 删除清单（当前仓库中的废弃内容）

执行新管线前应删除以下内容:

| 文件 | 原因 | 状态 |
|------|------|------|
| `include/rm_dart_aero.hpp` | 仿真不使用，被 CSV 管线取代 | ✅ 已删除 |
| `src/subsonic_solver/subsonic_solver.cu` | 未经验证的 GPU 代码，收敛标志伪造 | ✅ 已删除 |
| `include/subsonic_solver/subsonic_solver.hpp` | 同上 | ✅ 已删除 |
| `include/subsonic_solver/subsonic_solver_schema.hpp` | 同上 | ✅ 已删除 |
| `src/subsonic_solver_tool.cpp` | 同上 | ✅ 已删除 |
| `scripts/dart/run_subsonic_stl_pipeline.py` | 依赖上述求解器 | ✅ 已删除 |
| `scripts/dart/build_subsonic_cfd_cases.py` | 同上 | ✅ 已删除 |
| `scripts/dart/import_subsonic_cfd_results.py` | 同上 | ✅ 已删除 |
| `scripts/dart/subsonic_solver_schema.py` | 同上 | ✅ 已删除 |
| `data/missile/hgv_config_optimized.json` | 未使用 | ✅ 已删除 |
| `core/` 目录 | 空目录 | ✅ 已删除 |
| `aerodynamics_table.csv` | 数据被标注为无效 | ✅ 已重命名为 .INVALID |

---

## 文件结构 (修复后)

```
missile/
├── data/
│   ├── EGM2008.gfc                  # 重力场系数
│   ├── NRLMSISE-00/                 # 大气模型数据
│   ├── missile/
│   │   ├── hgv_model.stl            # 权威几何 (唯一)
│   │   ├── aerodynamics_table.csv   # 气动系数表 (唯一)
│   │   ├── hgv_config.json          # 仿真配置
│   └── dart/
│       ├── rm_dart_model.stl
│       ├── rm_dart_aero_table.csv
│       └── dart_config.json
│
├── include/                 # C++ 头文件
│   ├── gravity_model.hpp
│   ├── aerodynamics_model.hpp     ← 只读 CSV, 不做死代码
│   ├── dynamics_6dof.hpp
│   ├── gnc/guidance.hpp           ← PN 已重写
│   ├── gnc/autopilot.hpp          ← 简化为 3 PID
│   └── ...
│
├── src/                    # C++/CUDA 源文件
│   ├── main.cpp
│   ├── aero_solver/
│   │   └── aero_solver.cu         ← 已修正 (力矩参考点 + Cp)
│   ├── aero_calc_tool.cpp
│   └── ...
│
├── scripts/
│   ├── missile/
│   │   ├── generate_aero_table.py  ← 编排: 调用 AeroCalc + 低马赫段
│   │   ├── waverider_physics.py
│   │   ├── generate_cone_waverider.py
│   │   ├── tune_parameters.py
│   │   └── validate_aero_table.py  ← 新增: 自动验证 CSV
│   └── dart/
│       ├── generate_dart_aero_table.py
│       └── monte_carlo_dart.py
│
└── output/                 # 运行时输出
```

---

## 最终总结

**一句话**: 一条管线从 STL 到气动表到仿真，GPU 算高超声速、工程估算补低马赫、Python 只编排不参与运行时、C++ 负责全部性能关键路径。

**核心改动** (相比现状):
1. 修复 GPU 求解器 (力矩参考点 + Cp 修正)
2. 补低马赫段工程估算 (取代牛顿法在 Mach<5 的无效结果)
3. 删掉 4 条平行的废弃管线
4. 每阶段输出经过自动化验证再进入下一阶段
5. 路径硬编码改为配置文件和相对路径
