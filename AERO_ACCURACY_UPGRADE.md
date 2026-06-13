# 气动精度升级方案 Aero Accuracy Upgrade Plan

全部基于自研 C++/CUDA 代码，零第三方 CFD 依赖。

---

## 现状基线

| 当前 GPU 求解器 | 公式 | 精度 |
|----------------|------|------|
| 表面压力 | 修正牛顿法 `Cp = Cp_stag * cos²θ` | M≥5 时 ±15-25% |
| 表面摩擦 | **无** | CD 低估 20-30% |
| 底阻 | **无** | CD 低估 5-15% |
| 黏性干扰 | **无** | 高压区压心偏前 |
| 真实气体 | γ = 1.4 常数 | M≥12 时 Cp 偏高 15-25% |

六个缺失的物理项全部在 GPU kernel 内补全，逐三角面计算，不依赖任何外部工具。

---

## 阶段 A — GPU 附面层 + 真实物理解（当前，预计 2-3 周）

### 目标

在每个表面三角面片上，力系数从纯牛顿压力扩展为三项叠加：

```
dF = (Cp * n̂ + Cf * t̂) * dA
├── Cp:  修正牛顿法压力 (已实现，需升级)
├── Cf:  van Driest II 可压湍流摩阻 (新增)
└── 全局修正: 底阻 + 黏性干扰 + 真实气体 γ
```

精度目标：M ≥ 3 时 ±8-15%，M ≥ 10 时 ±5-10%。

### A.1 表面摩阻 — van Driest II 公式

**物理依据**：van Driest (1956) 推导了可压缩湍流边界层中平板摩擦系数的变换关系，将不可压流 Cf 映射到可压流。van Driest II 是航空航天工业标准方法（NASA TM X-74335, ESDU 78020），直接由第一性原理：边界层动量积分 + 混合长度湍流模型。

**输入条件**：
```
每个三角面已知:
  M_local           — 局部马赫数 (= M_∞ · sin(θ), 或更精确的牛顿解)
  Re_∞              — 来流雷诺数 (由 ρ, V, μ 确定)
  x_running         — 从驻点沿流线到本面的距离 (mm)
  T_w/T_0           — 壁温/总温比 (默认绝热壁, T_w = T_r = T_∞ · (1 + r*(γ-1)/2 * M²))
```

**Cf 计算公式 (van Driest II, 绝热壁)**：

```
F = 1 + 0.178 * (γ - 1) * M_local²                  (可压修正因子)

Cf_incompressible = 0.455 / [log₁₀(Re_x)]²·⁵⁸       (Prandtl-Schlichting 湍流平板)

Cf_compressible = Cf_incompressible / F               (van Driest II 可压映射)

其中 Re_x = ρ_∞ * V_∞ * x_running / μ_∞            (局部雷诺数)
```

这是标准的绝热壁 van Driest II 公式，没有任何可调参数。μ_∞ 由 Sutherland 公式从温度确定。

**流线迹线算法**：
```
输入: STL 三角网格 (法线 n̂_i, 三角形中心 p_i)
输出: 每个三角面的 x_running

1. 识别驻点区域:
   - 对每个三角面，计算来流在法线方向的投影 v_n = V_∞ · n̂_i
   - v_n 最大且符号为正的三角面群为驻点区 (迎风面法线正对来流)
   - 取这些三角面中心的平均位置为驻点 p_stag

2. 面到面的流线追踪:
   - 对每个三角 i，计算面内流动方向:
      t_s = V_∞ - (V_∞ · n̂_i) * n̂_i        (来流在三角面上的投影)
     (t_s 沿面由驻点指向下游)

   - 沿 t_s 方向找到相邻三角面 i→j:
      邻面条件:   n̂_i · n̂_j > 0 (光滑连接)
                 (p_j - p_i) · t_s > 0 (下游方向)
      最匹配:     (p_j - p_i) × n̂_i 与 t_s × n̂_i 夹角最小

   - x_running_i = ‖p_i - p_stag‖ × cos(φ)
     其中 φ = 弧线修正角度 (流线不在三角面中心连线上，但在细长体假设下 cosφ ≈ 1)

3. 对于高 α 迎风面，流线沿体轴方向为主:
   细长体简化: x_running ≈ 三角面中心到驻点的轴向距离
              ≈ ‖(p_i - p_stag) · V_∞/‖V_∞‖‖
```

### A.2 高超声速黏性干扰

**物理依据**：M > 5 时附面层位移厚度使外部无黏流偏转，产生额外压力。这一效应用黏性干扰参数 χ̄ 描述（Hayes & Probstein, 1966; Anderson, 1989）。

```
χ̄ = M_∞³ · √(C / Re_∞)
  其中 C = ρ_w μ_w / (ρ_∞ μ_∞)  (Chapman-Rubesin 因子, 低温壁取 ~0.1-1.0)
```

**压力增量**（由附面层对无黏流的等效位移产生）：

```
Δp / p_∞ = 0.15 · χ̄         (湍流)
Δp / p_∞ = 0.50 · χ̄         (层流, 仅用于极高海拔段)

对迎风面三角面片 (n̂ · V̂_∞ > 0):
  Cp 增加:
    Cp_viscous = Cp_newtonian + ΔCp_VI
    ΔCp_VI = Δp / q_∞ = (0.15 · χ̄) · p_∞ / q_∞
           = (0.15 · χ̄) · 2 / (γ · M_∞²)

对背风面 (n̂ · V̂_∞ < 0):
  黏性干扰效应较小，忽略
```

### A.3 底阻

**物理依据**：底部低压区的阻力由底压系数决定，与马赫数、底面积/参考面积比、附面层状态相关（Hoerner, 1965; AGARDograph 264）。

```
Cp_base = -1 / (γ · M_∞²) · (p_base / p_∞ - 1)

p_base / p_∞ = 0.10 + 0.90 · exp(-0.50 · M_∞)     (湍流, M ≥ 2)
p_base / p_∞ = 0.35 · exp(-0.15 · M_∞) + 0.10     (湍流, M < 2)

CD_base = -Cp_base · A_base / A_ref
```

底部压力只与来流条件相关，不依赖具体三角面。底阻作为一个标量值加到总 CD 上。

### A.4 真实气体 γ 修正

**物理依据**：M ≥ 10 时激波层内温度超过 2000K，氧氮分子振动激发、离解，导致 γ 从 1.4 降至 1.2-1.3。修正牛顿法的驻点 Cp 公式依赖 γ，γ 每降低 0.1，Cp 约降 6%（Anderson, 1989 §11.4）。

**γ 经验模型**（基于 Park (1990) 空气热力学数据的拟合）：

```
γ_eff(M, T_∞) = 1.4 - Δγ(T_eq)

其中平衡温度 T_eq 由激波关系估算:
  正激波后 T_post ≈ T_∞ · (2γM² - (γ-1)) · ((γ-1)M² + 2) / ((γ+1)²M²)
  简化:  T_post ≈ T_∞ · (2γ(γ-1)M⁴ - (γ-1)(γ-3)M²) / (γ+1)²

Δγ(T) 查表:
  T ≤ 2000K:  Δγ = 0.0          (无激发)
  2000K:      Δγ = 0.02         (振动激发开始)
  4000K:      Δγ = 0.08         (O₂ 离解 50%)
  6000K:      Δγ = 0.14         (N₂ 离解 50%)
  8000K:      Δγ = 0.18         (完全离解)
  T ≥ 10000K: Δγ = 0.20         (电离开始)

在 kernel 中线性插值查表中值，每个 Mach 条件执行一次 (不逐三角面)。
```

**修正后的驻点 Cp**：

```
γ = γ_eff 代入修正牛顿法:
  Cp_stag = (γ + 3) / (γ + 1) · [1 - 1 / (γ · M²)]
```

### A.5 实现要点

```
数据结构:
  struct TriangleData {
      float3 v0, v1, v2;          // 顶点
      float3 center;              // 面心
      float3 normal;              // 外法线
      float  area;                // 面积
      float  x_running;           // 流线距离 (从驻点) ← 新增
  };

GPU kernel 计算流 (逐三角面):
  for each triangle:
    // 1. 牛顿法压力 Cp (已实现, 用 γ_eff)
    dot_val = -dot(normal, flow_dir);
    Cp = ... // 修正牛顿法

    // 2. 黏性干扰修正
    Cp += ΔCp_VI(dot_val, M_∞, Re_∞)

    // 3. 摩阻系数 (迎风面)
    if (dot_val > 0) {
      Cf = van_driest_II(M_local, Re_x_running, T_w_T_0)
    }

    // 4. 合力
    dF = Cp * normal * area + Cf * tan_dir * area

  // 5. 全局: 加底阻
  CD_total += CD_base

  // 6. 力矩积分 (不变)
```

### A.6 交付物

| 组件 | 文件 | 行数估计 |
|------|------|---------|
| van Driest II Cf 函数 | 新增 `engineer/aero_skin_friction.hpp` | 80 |
| 黏性干扰参数 χ̄ 计算 | 同上 | 30 |
| 底压系数关联式 | 同上 | 30 |
| 真实气体 γ(T) 表 + 插值 | 新增 `engineer/real_gas.hpp` | 60 |
| 流线迹线算法 | 新增 `src/streamline_tracer.cpp` | 200 |
| GPU kernel 升级 | 修改 `src/aero_solver.cu` | +80 |
| 单元测试 | `tests/test_aero_viscous.cu` | 150 |
| **合计** | | **~630 行新增** |

---

## 阶段 B — GPU 有限体积 Euler 求解器（自研 CFD，2-3 月）

### 目标

用自研的完整 GPU FVM 求解器取代阶段 A 的近似方法，直接计算三维流动的压强场和密度场，精度提升至 ±3-8%。

### B.1 架构总览

```
STL 表面三角网格
    │
    ▼
四面体体积网格生成器 (CPU)
    │  Delaunay 逐点插入 + Bowyer-Watson
    │  输出: 节点坐标 + 四面体连接关系
    │
    ▼
FVM 求解器 (GPU)
    │  每四面体一线程
    │  守恒变量 Q = [ρ, ρu, ρv, ρw, ρE]
    │  HLLC Riemann 通量 + MUSCL 2阶重构 + RK4
    │  远场 Riemann 不反射边界
    │  壁面滑移 (法向无通量)
    │
    ▼
结果提取
    │  壁面压力分布 → 积分 → CL, CD, Cm
    │
    ▼
aerodynamics_table.csv
```

### B.2 四面体网格生成器（纯 C++ CPU，约 800 行）

```
算法: 3D Delaunay 逐点插入 (Bowyer-Watson)
输入: STL 表面三角网格 (已知, 约 2000 面)
输出: 内部四面体网格 (约 5-10 万单元)

步骤:
  1. 边界盒: 计算 STL 的最小包围盒 + 放大 10 倍为外边界
  2. 超级四面体: 建立包含所有点的初始 Delaunay 四面体
  3. 表面点插入: 将 STL 所有顶点插入 Delaunay 结构
  4. 边界恢复: 恢复 STL 三角面为网格中的面 (edge/face swapping)
  5. 内部点插入: 在已有四面体中心加点, 直到密度满足要求
  6. 去除外部: 标记超级四面体/外部区域, 保留内部
  7. 导出: 节点坐标 + 四面体 → GPU 内存

数据结构:
  - 点: float3[N_nodes]
  - 四面体: int4[N_cells] (四个节点索引)
  - 面: 从四面体边界隐式提取, 不做独立存储
```

### B.3 FVM 求解器核心（CUDA，约 600 行）

**控制方程（守恒型欧拉方程）**：

```
∂Q/∂t + ∂F/∂x + ∂G/∂y + ∂H/∂z = 0

Q = [ρ, ρu, ρv, ρw, ρE]ᵀ

F = [ρu, ρu²+p, ρuv, ρuw, u(ρE+p)]ᵀ
G = [ρv, ρvu, ρv²+p, ρvw, v(ρE+p)]ᵀ
H = [ρw, ρwu, ρwv, ρw²+p, w(ρE+p)]ᵀ

p = (γ-1)(ρE - ½ρ(u²+v²+w²))
```

**Riemann 求解器：HLLC**（Toro, 1999 §10.4）：

```
HLLC flux = 
 如果 S_L > 0:             F_L
 如果 S_L ≤ 0 ≤ S_M:      F_*L
 如果 S_M ≤ 0 ≤ S_R:      F_*R
 如果 S_R < 0:             F_R

S_L = min(u_L - a_L, u_R - a_R)
S_R = max(u_L + a_L, u_R + a_R)
S_M = (p_R - p_L + ρ_L u_L (S_L - u_L) - ρ_R u_R (S_R - u_R))
    / (ρ_L (S_L - u_L) - ρ_R (S_R - u_R))
```

**MUSCL 2 阶重构**（van Albada 限制器）：

```
Q_L = Q_i + 0.5 * ψ(r) * ∇Q_i · (x_face - x_i)
Q_R = Q_j - 0.5 * ψ(r) * ∇Q_j · (x_face - x_j)

ψ(r) = (r² + r) / (r² + 1)         (van Albada 限制器)
r = (Q_i - Q_{i-1}) / (Q_{i+1} - Q_i)
```

**时间推进：低存储 RK4（Williamson, 1980）**：

```
Q₀ = Qⁿ
Q₁ = Q₀ + α₁ · Δt · R(Q₀)
Q₂ = Q₀ + α₂ · Δt · R(Q₁)
Q₃ = Q₀ + α₃ · Δt · R(Q₂)
Qⁿ⁺¹ = Q₀ + α₄ · Δt · R(Q₃)

α = [1/4, 1/3, 1/2, 1]
```

**边界条件**：

```
远场 (Farfield) — Riemann 不反射:
  根据来流 Mach, α, β 确定外边界 Q_∞
  在面处求解局部一维 Riemann 问题, 取特征波外传部分

壁面 (Wall) — 滑移:
  通量 = [0, p·n̂_x, p·n̂_y, p·n̂_z, 0]ᵀ
  速度法向分量强制为零
```

**批处理架构**：

```
线程映射:
  blockIdx.x: Mach × Alpha 条件索引 (每个 block 一个条件)
  threadIdx.x: 四面体索引 (每线程一个单元)
  每个 block 包含 ~500-1000 个线程 (单元数 ≤ 1024)

GPU kernel 时间循环:
  for (int iter = 0; iter < N_iter; ++iter) {
    __syncthreads()
    // 1. 计算面通量 (相邻单元相互通信用 shared memory)
    // 2. 残差累加
    // 3. RK4 子步
    // 4. 收敛检查 (reduce)
    if (残差 < 1e-8) break
  }

共享内存布局:
  Q_smem[256][5]        // 256 单元的守恒变量
  ∇Q_smem[256][5][3]    // 梯度

网格分区:
  单元数 > 1024 时, 多个 thread block 处理同一条件:
  blockIdx.x = MachAlpha_idx * N_blocks_per_case + block_in_case
  threadIdx.x = 单元在 block 内的局部索引
```

**收敛准则**：
```
残差 = √( Σ|ΔQ/Q|² / N_cells ) < 1e-8
单条件约 500-2000 时间步收敛, 随 Mach 增加变慢
```

### B.4 状态空间

与当前相同，M ≥ 3 时采用 FVM Euler，其余回退到阶段 A：

| 方法 | 马赫范围 | 状态点数 |
|------|---------|---------|
| FVM Euler (阶段 B) | 3-25 | ~200 (17×12) |
| 阶段 A (含黏性修正) | 0.5-3 | ~20 |
| **合计** | | **~220 状态点** |

每点约 10 万单元 × 1000 步 × 5 变量，单 GPU 约 0.5-2 秒。
批处理总耗时 ~2-5 分钟。

### B.5 交付物

| 组件 | 文件 | 行数估计 |
|------|------|---------|
| Delaunay 四面体网格生成 | `src/mesh_generator.cpp` + `.hpp` | 800 |
| 网格导出到 GPU | `src/mesh_generator.cpp` | 100 |
| FVM 核心 (HLLC+MUSCL+RK4) | `src/cfd_solver.cu` | 600 |
| 边界条件 | `src/cfd_solver.cu` | 150 |
| 批处理调度 | `src/cfd_solver.cu` | 100 |
| CPU 端 C++ 包装 | `include/cfd_solver/cfd_solver.hpp` | 100 |
| 集成到生成管线 | `src/aero_table_gen.cpp` (修改) | +50 |
| 测试验证 | `tests/test_cfd_mach10.cu` | 200 |
| **合计** | | **~2100 行** |

---

## 阶段 C — GPU NS + RANS + 热化学非平衡（零简化完整 NS 求解器）

### 总体目标

在 FVM Euler 骨架上逐层添加：黏性通量（NS） → 湍流模型（RANS） → 有限速率化学反应 → 热非平衡双温度 → 高阶格式。最终实现覆盖 M=0.5-25 的完整可压缩 NS 求解器，精度目标 ±1-3%。

### 核心原则

1. **零第三方依赖**：全部自研 C++/CUDA，不依赖任何外部 CFD 库
2. **逐层验证**：每新增一个物理模型，必须在已知基准算例上独立验证通过后，才可进入下一层
3. **无简化假设**：不允许用工程近似代替物理模型（例如不允许用 γ(T) 代替有限速率化学，不允许用壁函数代替积分到壁面）
4. **回归保护**：每个子阶段必须保证之前通过的所有测试仍然通过

### 总览管线

```
阶段 B 输出（欧拉解）
    │
C.0: 验证基础设施 ──→ 基准数据库（平板/圆锥/球体参考值）
    │
C.1: NS 黏性通量 ──→ 无滑移壁面 + 黏性应力 + 热传导 + Sutherland μ(T),k(T)
    │                 验证: 平板层流 BL (Cf, q_w vs Blasius/van Driest 层流)
    │
C.2: SA 湍流模型 ──→ ν̃ 输运 + 涡黏 + Boussinesq 修正
    │                 验证: 平板湍流 BL (Cf vs van Driest II 湍流)
    │
C.3: 数值重构升级 ──→ 黏性面重构 + Barth-Jespersen 限制器 + 可压修正
    │                 验证: 网格收敛性确认 (2 阶精度)
    │
C.4: Park 5-species ──→ 有限速率化学反应 + NR 温度求解器 + 混合输运
    │   化学非平衡     验证: 正激波组分 (vs NASA CEA)
    │
C.5: 双温度 ──────────→ T_v 振动温度 + Landau-Teller 松弛 + Park 控温
    │   热非平衡       验证: 球锥体 M=15 热流 (vs 文献)
    │
C.6: 壁面催化 ────────→ SiO₂ 有限催化 + 催化热流
    │                 验证: 催化/非催化热流比
    │
C.7: 棱柱 BL 网格 ────→ 表面推进层 + y+<1 + 棱柱-四面体过渡
    │                 验证: 网格收敛性 (粗糙/中等/精细)
    │
C.8: 高阶格式 ────────→ WENO-5 + AUSM+-up + LU-SGS 隐式
    │                 验证: 完整 HGV 管道
    ▼
完整 NS 求解器，覆盖 M=0.5-25，精度 ±1-3%
```

---

### C.0 — 验证基础设施（~150 行，~3 天）

**目标**: 建立标准验证算例基准数据库，使后续每个子阶段新增物理模型后，可在同一基准上量化精度变化。

**输入文件**: `tests/ref_data/` 目录存入以下基准数据的头文件或 CSV：

| 算例 | 流态 | 对比量 | 数据来源 |
|------|------|--------|---------|
| 平板层流 BL | M=5, Re/m=1e7, 等温壁 300K | Cf(x), q_w(x), δ(x) | Blasius + van Driest 层流变换 |
| 平板湍流 BL | M=5, Re/m=1e7, 绝热壁 | Cf(x), 速度型 u⁺(y⁺) | van Driest II + DNS 数据 |
| 正激波后组分 | M=10, p=1atm, T=300K | 摩尔分数 N₂/O₂/N/O/NO | NASA CEA 平衡计算 |
| 球头激波脱体 | M=10, R=0.1m, 等温壁 1500K | 激波距离 Δ, 驻点热流 | Fay-Riddell + 文献 |
| 半球柱体 | M=8, Re=1e6, 等温壁 | Cp(θ), St(θ) 沿表面分布 | NASA 兰利风洞数据 |

**实现**:

```
tests/ref_data/
├── flat_plate_laminar_ref.hpp      // Cf(x), q_w(x) 参考表
├── flat_plate_turbulent_ref.hpp    // Cf(x), u⁺(y⁺) 参考表
├── normal_shock_cea_ref.hpp        // Park 5-species 参考组分
├── sphere_heatflux_ref.hpp         // 球头驻点热流参考值
└── hemi_cylinder_ref.hpp           // Cp(θ), St(θ) 参考分布
```

**测试要求 `tests/test_cfd_ns.cpp`**:

```
TEST("C.0 reference data integrity")
    // 验证基准数据自身的物理合理性
    // 平板 Cf(x) 必须随 x 递减 (Re_x 增大)
    // 正激波后 T 必须上升
    // 所有数据 NaN/Inf 检查

TEST("C.0 baseline Euler on flat plate")
    // 在当前纯欧拉求解器上运行平板算例
    // 确认滑移壁面给出 Cf≈0, q_w≈0 (无黏验证)
    // 建立"零物理模型"基线，后续每步都可对比
```

**成功标准**: 所有参考数据加载正确，欧拉基线确认 Cf=0。

---

### C.1 — NS 黏性通量 + 无滑移壁面（~400 行，~10 天）

**目标**: 在现有 FVM 求解器中加入完整黏性通量，使控制方程从 Euler 升级为 NS。

**物理模型**:

```
∂Q/∂t + ∇·F_inv = ∇·F_vis

F_inv = [ρu, ρuu+pI, u(ρE+p)]ᵀ            (欧拉通量，已实现)
F_vis = [0, τ, τ·u + q]ᵀ                  (黏性通量，新增)

τ = μ(T)·[∇u + (∇u)ᵀ - 2/3(∇·u)I]        (牛顿流体应力张量)
q = -k(T)·∇T                               (Fourier 热传导)

μ(T) = μ_ref·(T/T_ref)^{3/2}·(T_ref+S)/(T+S)   (Sutherland，已有)
k(T) = μ(T)·Cp/Pr                               (Pr = 0.72，空气)
```

**守恒变量不变**: Q = [ρ, ρu, ρv, ρw, ρE]ᵀ（5 分量，同阶段 B）

**实现步骤**:

| # | 内容 | 文件 | 行数 | 详细说明 |
|---|------|------|------|---------|
| C.1.1 | 细胞中心梯度计算 | `cfd_solver.cu` | 100 | Green-Gauss 遍历四面体所有面：∇φ_i = (1/V_i)·Σ(φ_j + φ_i)/2 · n̂_f · A_f。支持原始变量 [u,v,w,T] 的梯度。需要新增 device kernel 或扩展现有 kernel 的梯度计算阶段 |
| C.1.2 | 面梯度插值 | `cfd_solver.cu` | 40 | 从两侧细胞梯度取 inverse-distance 加权平均：∇φ_f = w_i·∇φ_i + w_j·∇φ_j，其中 w = 1/|x_f - x_cell|。黏性通量需要面法向梯度 (∇φ·n̂)，需做正交修正：∇φ·n̂ = (φ_j-φ_i)/|d_ij| + [∇φ_avg - (∇φ_avg·ê)·ê]·n̂，其中 ê = d_ij/|d_ij| |
| C.1.3 | Sutherland k(T) + Pr 热传导 | `cfd_solver.cu` | 15 | k(T) = μ(T)·Cp/Pr，device 函数 |
| C.1.4 | 黏性通量计算 kernel | `cfd_solver.cu` | 100 | 在面循环中新增黏性分支：计算 τ_xx = 2μ·∂u/∂x - 2/3μ·(∇·u)，τ_xy = μ·(∂u/∂y+∂v/∂x)，q_x = -k·∂T/∂x。F_vis = [0, τ_x, τ_y, τ_z, τ·u + q]ᵀ·n̂·A |
| C.1.5 | 无滑移壁面 BC | `cfd_solver.cu` | 40 | 壁面通量不再用滑移（法向无通量），改为：u_w=0, v_w=0, w_w=0（速度无滑移），T_w = T_const（等温壁）或 ∂T/∂n=0（绝热壁）。壁面压力由内部外推（∂p/∂n=0）|
| C.1.6 | 黏性 CFL 条件 | `cfd_solver.cu` | 30 | dt_visc = CFL·Δx²·ρ / (2μ_eff)。总 dt = min(dt_inv, dt_visc)。黏性 CFL 通常比无黏严格 O(Re) 倍，需要隐式处理或显著降低 CFL |
| C.1.7 | 壁面热流输出 | `cfd_solver.cu` | 30 | 在 force_integration 中新增 q_w = -k·∂T/∂n 的积分，输出总加热率 Q (W) 和驻点热流 q_stag (W/m²) |

**CfdConfig 扩展**:

```cpp
struct CfdConfig {
    // ... 现有字段 ...
    bool viscous = false;       // 启用黏性通量 (C.1+)
    float wall_temperature = 300.0f;  // 等温壁温度 (K)
    bool adiabatic_wall = false;      // 绝热壁 (覆盖 wall_temperature)
    float prandtl = 0.72f;            // Pr 数（空气）
};
```

**CfdResult 扩展**:

```cpp
struct CfdResult {
    // ... 现有字段 ...
    float Q_wall;       // 总壁面热流 (W)
    float q_stag;       // 驻点热流 (W/m²)
};
```

**验证基准: 平板层流边界层 (M=5)**

算例设置:
```
计算域: 0 ≤ x ≤ 1m, 0 ≤ y ≤ 0.1m（二维平板，用单层四面体拉伸模拟二维）
入流: M=5, Re/m=1e7, T∞=300K, p∞=101325Pa
壁面: 等温 T_w=300K, 无滑移 u=v=w=0
网格: 壁面首层高度 y₁ = 1e-5m (y+ ≈ 1)，增长比 1.1，共 50 层
```

测试验证:
```
TEST("C.1 flat plate laminar Cf")
    // Cf_local(x) = τ_w / (0.5·ρ∞·U∞²)
    // 与 Blasius 解偏差 < 5% (考虑可压缩性后)
    // Blasius: Cf = 0.664 / sqrt(Re_x)
    // van Driest 层流: Cf_comp = Cf_incomp / sqrt(T_w/T_e)

TEST("C.1 flat plate laminar heat flux")
    // q_w(x) = -k·∂T/∂n|_wall
    // 与解析解偏差 < 10%
    // 层流: St = 0.332 / sqrt(Re_x) / Pr^{2/3}

TEST("C.1 velocity profile")
    // u(y) 沿壁面法向分布
    // 与 Blasius 变换后速度型对比

TEST("C.1 Euler regression — inviscid limit")
    // 设置 viscous=false 必须复现纯欧拉结果
    // Cf=0, q_w=0, CX 与欧拉一致
```

**成功标准**: 所有层流验证测试 PASS，欧拉回归测试 PASS。

**设计决策**:

- **为什么先做 NS 黏性通量再做湍流？** 因为 SA 模型和化学模型都依赖梯度算子（扩散项），必须先验证梯度 + 黏性通量的正确性
- **为什么从 Green-Gauss 梯度开始而不是 Least-Squares？** GG 更简单、在四面体网格上足够精确，LS 作为后续优化项
- **为什么保留滑移壁面作为 viscous=false 选项？** 回归测试需要，且对极高 Re 流动可以用 Euler 快速扫参

---

### C.2 — SA 一方程湍流模型（~500 行，~14 天）

**目标**: 在 NS 黏性通量基础上加入 Spalart-Allmaras 一方程 RANS 模型，计算湍流黏性系数 μ_t。

**物理模型**:

SA 输运方程（无量纲形式）:

```
∂ν̃/∂t + uⱼ·∂ν̃/∂xⱼ = c_b1·S̃·ν̃ - c_w1·f_w·(ν̃/d)²
  + (1/σ)·[∂/∂xⱼ((ν+ν̃)·∂ν̃/∂xⱼ) + c_b2·(∂ν̃/∂xⱼ)·(∂ν̃/∂xⱼ)]

其中:
  ν = μ_l/ρ                          (分子运动粘度)
  μ_t = ρ·ν̃·f_v1                     (涡黏系数)
  f_v1 = χ³/(χ³ + c_v1³),  χ = ν̃/ν
  S̃ = Ω + ν̃·f_v2/(κ²·d²)
  f_v2 = 1 - χ/(1+χ·f_v1)
  Ω = |∇×u|                          (涡量幅值)
  f_w = g·[(1+c_w3⁶)/(g⁶+c_w3⁶)]^{1/6}
  g = r + c_w2·(r⁶ - r),  r = ν̃/(S̃·κ²·d²)
```

SA 常数: c_b1=0.1355, c_b2=0.622, σ=2/3, κ=0.41, c_w1=c_b1/κ²+(1+c_b2)/σ, c_w2=0.3, c_w3=2, c_v1=7.1

Boussinesq 近似修正黏性通量:
```
μ_eff = μ_l + μ_t
k_eff = μ_l·Cp/Pr_l + μ_t·Cp/Pr_t    (Pr_l=0.72, Pr_t=0.9)
τ_ij,eff = μ_eff·(∂u_i/∂x_j + ∂u_j/∂x_i - 2/3·δ_ij·∂u_k/∂x_k)
q_eff = -k_eff·∇T
```

**实现步骤**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.2.1 | SA 输运方程离散 | 120 | 附加标量变量 ν̃，Q 扩展为 6 分量 [ρ,ρu,ρv,ρw,ρE,ρν̃]。对流项用 HLLC（退化到标量对流），扩散项用中心差分（同 C.1 黏性通量格式）。SA 方程与 NS 方程弱耦合（在一个迭代步内冻结 μ_t） |
| C.2.2 | SA 源项 | 80 | 生成项: c_b1·S̃·ν̃（壁面附近需要特殊处理防止非物理增长）。破坏项: c_w1·f_w·(ν̃/d)²（确保远场 ν̃ 衰减）。负 ν̃ 截断: ν̃ = max(ν̃, 0) |
| C.2.3 | 壁面/远场 BC | 30 | 壁面: ν̃_w = 0。远场: ν̃_∞ = 3·ν_∞（层流来流，湍流度 ~1% 假设），∂ν̃/∂n=0 |
| C.2.4 | 涡黏 + Boussinesq 修正 | 40 | f_v1(χ) 函数 → μ_t → μ_eff, k_eff, τ_eff, q_eff。修改 C.1.4 的黏性通量代码使 μ 和 k 变为有效值 |
| C.2.5 | SA-NS 耦合稳定性 | 60 | SA 方程在壁面附近高度刚性。处理方案: (a) 点隐式处理生成/破坏源项，(b) 对 ν̃ 方程单独 CFL 控制，(c) 初始 100 步用纯 Euler + SA 冻结启动 |
| C.2.6 | 壁面距离 d 计算 | 40 | 每个细胞中心到最近壁面的距离。CPU 预计算（遍历所有壁面三角面，取最小距离）。存储为 `d_wall[num_tets]`，上传 GPU 常量内存 |

**CfdConfig 扩展**:

```cpp
struct CfdConfig {
    // ... 现有 ...
    // C.1
    bool viscous = false;
    float wall_temperature = 300.0f;
    bool adiabatic_wall = false;
    float prandtl = 0.72f;
    // C.2
    bool turbulence = false;    // 启用 SA 湍流模型
    float prandtl_turb = 0.9f;  // 湍流 Pr 数
};
```

**验证基准: 平板湍流边界层 (M=5)**

算例设置:
```
与 C.1 相同平板，Re/m=1e7, M=5, T_w=300K
网格修正: y₁ = 1e-6m (y+ ≈ 0.1) 以满足 SA 积分到壁面
要求 y+ < 1（首层节点在黏性底层内）
```

测试验证:
```
TEST("C.2 flat plate turbulent Cf")
    // Cf_local(x) 与 van Driest II 湍流公式偏差 < 10%
    // van Driest II: Cf = 0.455 / [log₁₀(Re_x)]^{2.58} / F_c(M,T_w)

TEST("C.2 flat plate velocity profile")
    // u⁺ = u/u_τ vs y⁺ = y·u_τ/ν_w
    // 与 DNS 数据对比: 黏性底层 u⁺=y⁺ (y⁺<5), 对数律 u⁺=(1/κ)·ln(y⁺)+B (y⁺>30)
    // κ=0.41, B=5.0

TEST("C.2 turbulent heat flux")
    // St(x) 与 Reynolds 类比验证: St/Cf ≈ 1/(2·Pr^{2/3})

TEST("C.2 laminar regression")
    // turbulence=false 必须复现 C.1 层流结果

TEST("C.2 SA convergence")
    // 残差下降到 < 1e-6，ν̃ 场无负值
    // 壁面 ν̃_w = 0
```

**成功标准**: 所有湍流验证测试 PASS，层流回归测试 PASS，残差单调下降。

**设计决策**:

- **为什么选 SA 而非 SST k-ω？** SA 一方程，GPU 上每四面体只需多存 1 个变量（vs SST 的 2 个），实现和验证都更快。SST 在强分离流中更准，可在 C.8 后作为升级选项
- **为什么不用壁函数？** 壁函数在分离流和强压力梯度中精度不足，积分到壁面 (y+<1) 是"零简化"的要求
- **ν̃ 方程存为 ρν̃ 还是 ν̃？** 存 ρν̃（守恒变量），与现有 Q 数组格式统一

---

### C.3 — 黏性数值重构升级（~300 行，~7 天）

**目标**: 提升 C.1-C.2 中黏性通量的面重构精度，引入梯度限制器避免非物理振荡。

**必要性**: C.1 中的简单平均面梯度在高梯度区域（激波/边界层交界）会产生非物理通量。

**实现**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.3.1 | 黏性面重构升级 | 60 | 从"两侧简单平均"改为"inverse-distance 加权 + 正交修正"：∇φ_f = w_i∇φ_i + w_j∇φ_j，然后修正法向分量: ∇φ_f·n̂ = (φ_j-φ_i)/|d_ij| + [w_i∇φ_i + w_j∇φ_j - ((w_i∇φ_i + w_j∇φ_j)·ê)·ê]·n̂ |
| C.3.2 | Barth-Jespersen 梯度限制器 | 80 | 对原始变量 [ρ,u,v,w,p,ν̃] 的梯度施加限制，确保 MUSCL 重构不产生新的极值。ψ(r) = min(1, (φ_max-φ_i)/(φ_j-φ_i), (φ_min-φ_i)/(φ_k-φ_i)) |
| C.3.3 | 可压修正 | 40 | 亚音速区（M_local < 0.3）的黏性通量修正：低 Mach 预处理防止伪扩散 |
| C.3.4 | 隐式残差平均 | 60 | 加速收敛: R̃_i = R_i + ε·Σ(R̃_j - R̃_i)，ε=0.3-0.5，迭代 3-5 次。松弛因子 0.8 |

**测试验证**:

```
TEST("C.3 gradient reconstruction accuracy")
    // 在已知函数 φ(x,y,z)=x²+y²+z² 上测试梯度精度
    // 应达到 2 阶精度: ||∇φ_h - ∇φ_exact|| ∝ h²

TEST("C.3 limiting preserves monotonicity")
    // 在激波管问题 (Sod 问题) 上测试
    // 密度/压力无过冲 (overshoot < 1%)

TEST("C.3 regression — C.1 flat plate laminar")
    // 数值重构升级后平板层流结果不变
```

**成功标准**: 梯度 2 阶精度确认，激波管无过冲，回归测试全部通过。

---

### C.4 — Park 5-species 有限速率化学非平衡（~600 行，~21 天）

**目标**: 在 NS 黏性通量 + SA 湍流基础上加入有限速率化学反应，用 Park 5-species 模型计算空气离解和复合。

**这是 C 阶段中最复杂、计算量最大的子阶段。**

**模型选择: 5-species (N₂, O₂, N, O, NO)**

| Species | M_s (g/mol) | 作用 |
|---------|-------------|------|
| N₂ | 28.013 | 主要组分，高温离解 |
| O₂ | 31.999 | 低离解能，先离解 |
| N | 14.007 | 离解产物 |
| O | 15.999 | 离解产物 |
| NO | 30.006 | 中间产物，Zeldovich 机理 |

**控制方程**:

```
Q = [ρ₁, ρ₂, ρ₃, ρ₄, ρ₅, ρu, ρv, ρw, ρE]ᵀ    (9 分量)

∂ρ_s/∂t + ∇·(ρ_s u) = ∇·(ρ·D_s·∇Y_s) + ω̇_s     (组分输运，5 方程)
∂(ρu)/∂t + ∇·(ρuu + pI) = ∇·τ                    (动量，3 方程)
∂(ρE)/∂t + ∇·(u(ρE+p)) = ∇·(τ·u + q)            (能量，1 方程)
```

**实现步骤**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.4.1 | 数据结构升级 | 50 | Q 从 5 分量 → 9 分量（5 组分 + 3 动量 + 1 能量）。共享内存分配对应扩展，核函数参数全部更新 |
| C.4.2 | 混合气体状态方程 | 80 | p = Σρ_s·R_s·T，R_s = R_universal / M_s。C_p_s(T) 用 NASA 多项式系数（7 项，200-6000K）。混合 C_p = ΣY_s·C_p_s(T)。γ 不再为常数，γ = C_p/(C_p - R_mix) |
| C.4.3 | Newton-Raphson 温度求解器 | 120 | 从 ρE 反解 T: ρE = ½ρ|u|² + Σρ_s·h_s⁰ + Σρ_s·∫C_v_s(T)dT。NR 迭代: T^{n+1} = T^n - f(T^n)/f'(T^n)，其中 f(T) = ρE - ½ρ|u|² - Σρ_s·h_s⁰ - ρ·C_v_mix·T。容差 1e-6，最多 20 次迭代。失败时回退到二分法 |
| C.4.4 | 反应速率 Arrhenius | 100 | 17 个正逆反应 (Park 1990)。每个反应: k_f(T) = A·T^n·exp(-E_a/T)。逆反应速率 k_b = k_f/K_eq，K_eq = exp(-ΔG/RT) |
| C.4.5 | 反应源项 | 80 | ω̇_s = M_s·Σ(ν''_rs - ν'_rs)·(k_f_r·Π[ρ_i/M_i]^{ν'_ir} - k_b_r·Π[ρ_i/M_i]^{ν''_ir})。刚性问题：用点隐式处理源项 Jacobian |
| C.4.6 | 混合输运系数 | 60 | Wilke's 混合规则计算 μ_mix 和 k_mix。二元扩散系数 D_s (修正 Chapman-Enskog)。Lewis 数 Le = 1.4 近似（简化初期）|
| C.4.7 | 化学反应 CFL | 30 | 化学时间尺度 τ_chem = min(ρ_s/|ω̇_s|)。CFL_chem = Δt/τ_chem < 0.1。总 Δt = min(Δt_inv, Δt_visc, Δt_chem) |

**反应列表 (Park 1990, 17 reactions)**:

```
离解反应 (5):
  N₂ + M ⇌ N + N + M          (M = 任意碰撞伙伴)
  O₂ + M ⇌ O + O + M
  NO + M ⇌ N + O + M
  N₂ + O ⇌ NO + N
  NO + O ⇌ O₂ + N

交换反应 (2):
  N₂ + O ⇌ NO + N
  NO + O ⇌ O₂ + N

电离 (暂不包含，M>15 时扩展):
```

**CfdConfig 扩展**:

```cpp
struct CfdConfig {
    // ... 现有 + C.1 + C.2 ...
    // C.4
    bool chemistry = false;        // 启用有限速率化学
    int n_species = 5;             // 组分数量 (5 或扩展至 11)
};
```

**验证基准: 正激波后组分分布 (M=10)**

算例设置:
```
一维正激波: 入流 M=10, p=101325Pa, T=300K, 空气 (79% N₂, 21% O₂)
激波后 T ≈ 5000K, O₂ 完全离解, N₂ 部分离解
```

测试验证:
```
TEST("C.4 normal shock species — equilibrium limit")
    // 长时间积分 (t → ∞) 后组分应趋近平衡
    // 与 NASA CEA 平衡组分对比:
    //   N₂: ~65%, O₂: ~0%, N: ~5%, O: ~25%, NO: ~5%
    // 偏差 < 5%

TEST("C.4 normal shock T profile")
    // 激波后温度分布
    // 最大温度 T_max 与正激波关系计算值对比偏差 < 10%

TEST("C.4 species conservation")
    // ΣY_s = 1.0 (机器精度)
    // N 元素守恒: 2·Y_N₂ + Y_N + Y_NO = const
    // O 元素守恒: 2·Y_O₂ + Y_O + Y_NO = const

TEST("C.4 chemistry regression — frozen flow")
    // chemistry=false 必须复现 C.2 结果（冻结流，γ=1.4）

TEST("C.4 NR temperature solver")
    // 给定已知 ρE 和 ρ_s，NR 求解的 T 与直接计算偏差 < 1e-6 K
```

**成功标准**: 所有化学验证测试 PASS，组分守恒精确满足 (<1e-12)，冻结流回归通过。

---

### C.5 — 双温度热非平衡（~300 行，~10 天）

**目标**: 在 5-species 化学非平衡基础上加入振动温度 T_v，实现 T_t ≠ T_v 双温度模型。

**必要性**: M > 12 时激波层内 T_t > 10000K，振动自由度松弛时间与流动时间可比，热非平衡效应显著。

**控制方程扩展**:

```
Q = [ρ₁..ρ₅, ρu, ρv, ρw, ρE, ρe_v]ᵀ    (10 分量)

∂(ρe_v)/∂t + ∇·(ρe_v·u) = ∇·(k_v·∇T_v) + Q_TV - ω̇_chem_v
```

其中:
- e_v = ΣY_s·e_v_s(T_v): 混合气体振动能量
- e_v_s(T_v) = R_s·θ_v_s / [exp(θ_v_s/T_v) - 1]: 谐振子模型
- θ_v(N₂)=3393K, θ_v(O₂)=2270K, θ_v(NO)=2740K, θ_v(N)=0, θ_v(O)=0（原子无振动）
- Q_TV = Σρ_s·[e_v_s*(T) - e_v_s(T_v)] / τ_s_LT: Landau-Teller 松弛
- τ_s_LT = (ΣX_j) / (ΣX_j/τ_sj): Millikan-White 松弛时间 + Park 修正
- ω̇_chem_v: 化学反应引起的振动能量变化（离解偏好消耗高能分子）

**实现步骤**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.5.1 | 振动能量输运方程 | 80 | 对流项同组分输运格式，扩散项 k_v = ρ·D_v (振动扩散系数 ≈ 混合扩散) |
| C.5.2 | Landau-Teller 松弛 | 60 | Q_TV = Σρ_s·(e_v_s* - e_v_s)/τ_s。T-T_v 能量交换率。τ_s 由 Millikan-White 经验公式 + Park 下限修正 (τ_s ≥ 1/(σ_s·n·c̄)) |
| C.5.3 | Park 控温化学反应 | 50 | 反应速率用有效温度 T_a = T^q·T_v^{1-q}，q=0.5 (Park 建议)。离解反应受 T_v 控制，交换反应受 T_t 控制 |
| C.5.4 | 双温度 NR 求解器 | 60 | 从 ρE 和 ρe_v 联立求解 T 和 T_v。两步 NR: (1) 用 ρe_v 求 T_v，(2) 用 ρE - ρe_v 求 T_t |

**验证基准: 球锥体 M=15 再入**

算例设置:
```
球头半径 R=0.1m，半锥角 9°
M=15, 高度 h=60km (ρ=3.1e-4 kg/m³, T=247K)
等温壁 T_w=1500K
```

测试验证:
```
TEST("C.5 sphere-cone T-T_v profile")
    // 驻点线上 T_t 和 T_v 分布
    // T_v 应滞后于 T_t (松弛有限)
    // 与文献定性对比

TEST("C.5 sphere-cone surface heat flux")
    // 驻点热流 q_stag
    // 与 Fay-Riddell 公式对比偏差 < 15%

TEST("C.5 frozen vibration regression")
    // 设置 T_v=T（平衡）必须复现 C.4 结果
```

**成功标准**: 双温度热非平衡测试 PASS，平衡回归通过。

---

### C.6 — 壁面有限催化（~120 行，~3 天）

**目标**: 在化学非平衡壁面边界条件中加入有限速率催化复合反应。

**物理模型**:

SiO₂ 型防热瓦表面的 O/N 原子催化复合:
```
J_s = -γ_c(T_w) · ρ·Y_s · √(R_s·T_w / 2π)     (壁面法向组分通量)

γ_c,N(T_w) = 0.01 · exp(-1000/T_w)              (N 原子催化效率)
γ_c,O(T_w) = 0.01 · exp(-1500/T_w)              (O 原子催化效率)

催化热流: q_cat = ΣJ_s · h_s⁰                    (组分复合释放的化学能)
总壁面热流: q_total = q_fourier + q_cat
```

**实现**:

| # | 内容 | 行数 |
|---|------|------|
| C.6.1 | 催化效率计算 | 30 |
| C.6.2 | 组分壁面通量 J_s | 40 |
| C.6.3 | 催化热流 q_cat → CfdResult | 30 |

**测试验证**:

```
TEST("C.6 catalytic vs non-catalytic heat flux")
    // 催化壁热流 > 非催化壁热流
    // 催化加热比例 ≈ h_s⁰·Y_s / (C_p·ΔT) ≈ 10-30%

TEST("C.6 fully catalytic limit")
    // 设置 γ=1 应接近完全催化壁
    // 壁面 Y_s → 0 (原子全部复合)
```

---

### C.7 — 棱柱边界层网格生成（~600 行，~21 天）

**目标**: 从 STL 表面三角网格生成带棱柱边界层的混合网格，y+ < 1。

**必要性**: 当前纯四面体网格在壁面附近只有各向同性单元，无法高效解析附面层。棱柱层可用极少单元数（~20 层 × 增长比 1.2）覆盖 BL 的 ~4 个数量级速度变化。

**实现步骤**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.7.1 | 表面法向 + 曲率估计 | 60 | 每个表面节点计算加权平均法线（面积加权相邻三角面法线）。基于法线变化率估计曲率半径，限制棱柱层高度 < 0.5·R_curve |
| C.7.2 | 推进层算法 | 250 | 从每个表面三角面沿法向推进: 首层高度 h_1 根据目标 y+ 计算（y+=1 时 h₁ ≈ y+·ν_w/u_τ）。逐层推进: h_{i+1} = h_i·r_g，r_g=1.2。每层推进后做平滑（Laplacian 松弛 3-5 次）防止单元交叉 |
| C.7.3 | 棱柱-四面体过渡 | 100 | 棱柱层顶面作为新表面，用现有 Delaunay 生成器填充四面体核心。棱柱层顶面节点需做保形处理（保持 STL 几何特征）|
| C.7.4 | 质量检查 + 修复 | 100 | 负体积检测（任意单元 det(J) < 0 → 标记）。局部节点重定位修复负体积。最小正交角 > 20° |
| C.7.5 | 输出 TetMesh 格式 | 30 | 棱柱拆分为 3 个四面体/棱柱（保持网格连续性）。输出标准 TetMesh（节点 + 四面体 + 邻居）供 CfdSolver::load_mesh 加载 |

**网格参数**（CfdConfig 扩展）:

```cpp
struct CfdConfig {
    // ... 现有 ...
    // C.7
    float bl_first_height = 1e-6f;    // 首层棱柱高度 (m)
    float bl_growth_rate = 1.2f;      // 增长比
    int bl_layers = 30;               // 棱柱层数
};
```

**验证**:

```
TEST("C.7 BL mesh quality")
    // 最小正交角 > 20°
    // 最大长宽比 < 100
    // 无负体积
    // y+ < 1 覆盖率 > 99%

TEST("C.7 grid convergence — flat plate Cf")
    // 在 3 套网格上（粗糙 1K、中等 10K、精细 100K 单元）
    // Cf 随网格加密收敛到 van Driest II 参考值
    // 收敛阶 ≈ 1.5-2.0 (NS + 湍流)

TEST("C.7 prism vs tetrahedral-only")
    // 相同节点数的棱柱+四面体网格 vs 纯四面体
    // 棱柱网格在 BL 解析度上 > 纯四面体
```

---

### C.8 — 高阶格式 + 隐式时间推进（~450 行，~14 天）

**目标**: 在完成所有物理模型后提升数值精度和收敛速度。

**必要性**: 当前显式 RK4 + MUSCL 2 阶在含化学反应的刚性问题中需要极小的 CFL（通常 CFL < 0.01），且 2 阶精度对高超音速激波分辨率不足。

**实现**:

| # | 内容 | 行数 | 说明 |
|---|------|------|------|
| C.8.1 | WENO-5 重构 | 150 | 替代 MUSCL 2 阶。5 阶精度（光滑区），3 阶（激波附近）。基于特征变量的 WENO 重构（避免标量 WENO 在方程组中的振荡）。使用 5 个模板（3 个左偏 + 中央 + 3 个右偏）|
| C.8.2 | AUSM+-up 通量 | 120 | 替代 HLLC（可选）。低马赫数兼容性更好（HLLC 在 M<0.1 时耗散过大）。统一通量函数适用于全马赫数范围 (M=0-25)。保留 HLLC 作为选项（回归测试）|
| C.8.3 | LU-SGS 隐式时间推进 | 150 | 替代 RK4 显式。无条件稳定（线性化意义上），Δt 不受 CFL 限制（实际受线性化误差限制）。LUSGS 扫两遍（前向 + 后向），每遍 O(N_cells) 复杂度。化学源项 Jacobian 精确线性化 |

**CfdConfig 扩展**:

```cpp
struct CfdConfig {
    // ... 全部现有 ...
    // C.8
    int reconstruction_order = 2;   // 2=MUSCL, 5=WENO
    int riemann_solver = 0;         // 0=HLLC, 1=AUSM+
    int time_integration = 0;       // 0=RK4, 1=LU-SGS
};
```

**测试验证**:

```
TEST("C.8 WENO accuracy — isentropic vortex")
    // 等熵涡对流: 应达到 5 阶精度
    // ||ρ - ρ_exact|| ∝ h⁵ (光滑区)

TEST("C.8 AUSM+ vs HLLC — low Mach")
    // M=0.1 圆球绕流
    // AUSM+ 的耗散应显著低于 HLLC

TEST("C.8 LU-SGS convergence speed")
    // 同样算例, LU-SGS 达到收敛的迭代次数
    // 比 RK4 减少 10-100×
    // 残差下降速率对比

TEST("C.8 full regression — all previous cases")
    // 所有之前子阶段的测试用例必须全部通过
    // 这是进入最终集成前的最后关口
```

---

### 子阶段依赖关系图

```
C.0 ─────────────────────────────────────────────────────────────────
  │                                                                 
  ▼                                                                 
C.1 (独立)          C.7 (独立)                                       
  │                   │                                              
  ▼                   │                                              
C.2 (依赖 C.1)       │                                              
  │                   │                                              
  ▼                   │                                              
C.3 (依赖 C.1-C.2)   │                                              
  │                   │                                              
  ▼                   ▼                                              
C.4 (依赖 C.1-C.3) ──┤   (C.7 与 C.4 可部分并行: C.7.1-2 可与 C.1-3 并行)
  │                   │                                              
  ▼                   │                                              
C.5 (依赖 C.4)       │                                              
  │                   │                                              
  ▼                   │                                              
C.6 (依赖 C.4-C.5)   │                                              
  │                   │                                              
  ────────────────────┤                                              
  │                   │                                              
  ▼                   ▼                                              
C.8 (依赖全部) ───────┤                                              
  │                   │                                              
  ▼                   ▼                                              
完整 NS 求解器 ───────┤                                              
```

---

### 完整交付物清单

| 子阶段 | 文件 | 代码量 | 测试文件 | 测试数 |
|--------|------|--------|---------|--------|
| C.0 | `tests/ref_data/*.hpp` | 150 | `tests/test_cfd_ns.cpp` | 2 |
| C.1 | `src/aero_solver/cfd_solver.cu` (+350), `include/aero_solver/cfd_solver.hpp` (+20) | 400 | `tests/test_cfd_ns.cpp` | 4 |
| C.2 | `src/aero_solver/cfd_solver.cu` (+450), `cfd_solver.hpp` (+20) | 500 | `tests/test_cfd_ns.cpp` | 4 |
| C.3 | `src/aero_solver/cfd_solver.cu` (+250), `cfd_solver.hpp` (+10) | 300 | `tests/test_cfd_ns.cpp` | 3 |
| C.4 | `src/aero_solver/cfd_solver.cu` (+400), `include/aero_solver/chemistry.hpp` (+200) | 600 | `tests/test_cfd_ns.cpp` | 5 |
| C.5 | `src/aero_solver/cfd_solver.cu` (+200), `chemistry.hpp` (+100) | 300 | `tests/test_cfd_ns.cpp` | 3 |
| C.6 | `src/aero_solver/cfd_solver.cu` (+100) | 120 | `tests/test_cfd_ns.cpp` | 2 |
| C.7 | `src/aero_solver/mesh_generator.cpp` (+500), `mesh_generator.hpp` (+50) | 600 | `tests/test_mesh_generator.cpp` | 2 |
| C.8 | `src/aero_solver/cfd_solver.cu` (+400), `cfd_solver.hpp` (+20) | 450 | `tests/test_cfd_ns.cpp` | 4 |
| **合计** | | **~3420 行** | | **29 测试** |

---

### 最终验证：完整 HGV 气动表生成

所有子阶段完成后，对真实 HGV 几何（1936 三角面）生成 238 点气动表：

```
Mach: 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0
Alpha: -10, -5, 0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40
Beta: 0 (轴对称)
```

**最终精度验证**:

| 马赫范围 | 物理模型覆盖 | 目标精度 |
|---------|------------|---------|
| M < 3 | 阶段 A 工程方法 | ±10-15% |
| M 3-5 | NS + SA (C.1-C.3) | ±5-8% |
| M 5-12 | NS + SA + Park 5-species (C.4-C.6) | ±3-5% |
| M 12-25 | NS + SA + 5-species + T-T_v (C.5-C.8) | ±1-3% |

**交叉验证**: 最终 238 点表与阶段 B（纯 Euler FVM）在 M≥3 区域对比，CX/CD 应在 ~20-40% 范围内（黏性 + 化学反应对力的修正量级）。

---

### 测试总体要求

| 要求 | 说明 |
|------|------|
| 回归完整性 | 每子阶段完成后运行所有之前测试，确保零退化 |
| 数值精度 | 所有浮点比较用相对容差 (tol_rel=1e-5)，NaN/Inf 检查 |
| GPU 错误检查 | 每个 CUDA 调用后检查错误，kernel 启动后 cudaPeekAtLastError |
| 覆盖范围 | 边界条件（壁面/远场/对称）、极端参数（M=25, α=40°）、零物理模型退化 |
| 残差收敛 | 每个求解算例必须报告最终残差，残差 > 1e-4 时测试警告 |
