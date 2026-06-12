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

## 阶段 C — GPU RANS + 化学非平衡（远期，3-6 月）

### 目标

在 FVM Euler 基础上增加：SA 湍流模型 + Park 11 组分化学 + 有限催化壁面，精度 ±1-3%。

### C.1 SA 湍流模型

Spalart-Allmaras 一方程模型，附加方程求解湍流黏性系数 ν̃：

```
∂ν̃/∂t + u·∇ν̃ = c_b1·S̃·ν̃ - c_w1·f_w·(ν̃/d)²
  + 1/σ·[∇·((ν+ν̃)∇ν̃) + c_b2·(∇ν̃)·(∇ν̃)]
```

壁面: ν̃_wall = 0
远场: ν̃_∞ = 3·ν_∞ (层流)

### C.2 Park 11 组分化学模型

控制方程增加 11 个组分质量分数输运方程 + 有限速率反应源项：

```
Species: N₂, O₂, N, O, NO, N₂⁺, O₂⁺, NO⁺, e⁻, Ar, CO₂
Reactions: 20 个正逆反应 (Park, 1993)
源项: d(ρY_s)/dt = M_s · Σ R_rs · (ν''_rs - ν'_rs)
```

温度求解: 从总能量中减去化学能得到平动-转动温度 T:
```
ρE = ρ·C_v·T + Σ ρ_s·h_s⁰ + ½ρ|u|²
T 用 Newton-Raphson 迭代求解
```

### C.3 壁面有限催化

SiO₂ 涂层催化效率 γ_c(T_w):

```
γ_c(N)  = 0.01 · exp(-1000/T_w)
γ_c(O)  = 0.01 · exp(-1500/T_w)
壁面法向组分通量: J_s = -γ_c · ρ·Y_s · √(RT_w/2πM_s)
```

### C.4 交付物

| 组件 | 行数估计 |
|------|---------|
| SA 湍流模型 GPU | 400 |
| Park 11 species + 20 reactions | 500 |
| Newton-Raphson T 求解器 | 150 |
| 有限催化壁面 BC | 100 |
| **合计** | **~1150 行** |

---

## 项目里程碑

```
阶段 A ──────────────── 当前
  周 1:   van Driest II + 底阻 + 流线追踪   ✅ 编码
  周 2:   GPU kernel 集成 + 真实气体        ✅ 编码+单元测试
  周 3:   全速域标定 + 精度评估             ✅ 验证

阶段 B ──────────────── 下月
  月 1:   Delaunay 网格生成器               ✅ C++ CPU
  月 2:   FVM HLLC kernel                   ✅ CUDA
  月 3:   全表格生成 + 与阶段 A 交叉验证     ✅ 集成

阶段 C ──────────────── 远期
  按需启动
```

---

## 与现有架构的集成

阶段 A 不改变数据管线结构，只在 GPU kernel 计算的力模型中增加物理项：

```
STL → GPU: [Newtonian Cp] ──→ [Newtonian + Cf + Δp_VI + CD_base]
                                     │ γ_eff 修正
                                     ▼
                             计算 CL, CD, Cm
                                     │
                                     ▼
                             aero_table_gen.cpp (不变)
                                     │
                                     ▼
                             aerodynamics_table.csv (格式不变)
                                     │
                                     ▼
                             aerodynamics_model.hpp (不变)
```

阶段 B 增加一条新路径：
```
STL → 网格生成 (CPU) → FVM Euler (GPU) → 压积 → CL/CD/Cm → CSV
```

与阶段 A 共存，M ≥ 3 用 FVM，M < 3 用阶段 A，连续过渡。
