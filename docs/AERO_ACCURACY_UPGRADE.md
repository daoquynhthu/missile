# CFD Solver Rebuild Design

## 0. 目标与边界

本子工程的目标是建立面向高超声速飞行器和弹体的高保真气动/热环境求解器，而不是做演示性质或工程曲线拟合性质的 CFD。短期输出必须能进入 6-DOF 弹道仿真；长期目标是形成从可验证低阶基线到 DNS/高阶/热化学壁面模型的连续精度路线。

“第一性原理”在本项目中定义为：

- 离散方程必须来自明确的控制方程、通量、边界条件和材料/气体模型。
- 任何经验模型、湍流闭合、转捩模型、壁面催化模型都必须显式标注适用范围和不确定性，不允许伪装成真实物理解。
- 任何数值近似都必须有网格收敛、阶精度验证、守恒误差和物理量误差预算。
- 不接受“看起来合理”“量级差不多”作为最终正确性；这类测试只允许作为早期 smoke/sanity gate。

核心输出：

- 体轴力系数：`CX`, `CY`, `CZ`
- 风轴力系数：`CD`, `CL`
- 力矩系数：`Cl`, `Cm`, `Cn`
- 壁面热流：总热流、驻点热流、局部 `Cf/St` 探针
- 收敛历史、守恒误差、诊断场

非目标：

- 不追求第一版覆盖全物理，但架构不得封死 DNS、高阶格式、热化学、转捩和 GPU 生产路径。
- 不用壁函数替代积分到壁面作为最终方案。
- 不把“finite/positive/no NaN”当作物理正确。
- 不在旧 `cfd_solver.cu` 基础上继续堆补丁。

明确不再接受的长期路线：

- 以 SA/RANS 作为“最终真值”。
- 以低阶一阶/二阶耗散格式作为“顶尖求解器”终点。
- 以 CPU 参考实现作为生产性能路线。
- 在未知转捩、未知壁面催化、未知材料反应率时输出单一“精确热流值”。

当前清理后的原则：

- 旧 CFD 求解器、旧网格生成器、旧 CFD 测试已移除。
- 现有 Newtonian/engineering 气动估算保留，作为低保真 fallback 和交叉验证参考。
- `use_fvm=true` 暂时禁用，直到新 CFD solver 重新接入。

## 1. 成功标准

每一层物理模型必须同时满足四类门禁：

1. 数值门禁：残差下降、无 NaN/Inf、正密度/正压力、有限 CFL。
2. 守恒门禁：封闭控制体质量/动量/能量通量误差在阈值内。
3. 物理门禁：量级和趋势符合解析解、参考解或工程边界。
4. 回归门禁：禁用新物理时精确退化到上一层结果。

禁止的门禁：

- 只检查输出为正。
- 只检查残差有限。
- 只用单一算例证明稳定。
- 在断言里接受多个数量级错误。

## 2. 总体架构

新 CFD 子工程按模块拆分，不再把网格、通量、时间推进、边界条件、诊断和测试钩子全部塞进单个 CUDA 文件。

建议目录：

```text
include/aero_cfd/
  cfd_config.hpp
  cfd_state.hpp
  cfd_mesh.hpp
  cfd_result.hpp
  cfd_solver.hpp
  diagnostics.hpp

src/aero_cfd/
  cfd_solver.cu
  mesh_metrics.cpp
  boundary_conditions.cu
  flux_euler.cu
  flux_viscous.cu
  timestep.cu
  force_integrator.cu
  diagnostics.cpp
  gpu_kernels.cu
  high_order/
  transition/
  thermochemistry/

tests/cfd/
  test_cfd_mesh.cpp
  test_cfd_euler.cu
  test_cfd_viscous.cu
  test_cfd_diagnostics.cpp
```

模块职责：

| 模块 | 职责 | 禁止事项 |
|------|------|----------|
| `cfd_mesh` | 存储节点、单元、面、邻接、边界标签 | 不负责物理边界条件 |
| `mesh_metrics` | 面面积法向、单元体积、形心、壁距、质量指标 | 不修正坏网格后继续无声运行 |
| `boundary_conditions` | farfield、slip wall、no-slip wall、thermal wall、symmetry | 不在通量函数里隐式猜边界 |
| `flux_euler` | HLLC/AUSM 等无黏通量 | 不处理黏性项 |
| `flux_viscous` | 黏性应力、热传导、面梯度 | 不修改 cell-centered 重构梯度 |
| `timestep` | inviscid/viscous/source CFL | 不把 debug dt 和实际 dt 分开 |
| `force_integrator` | 壁面积分和系数归一化 | 不从非壁面取力 |
| `diagnostics` | residual/history/VTK/probes/failure dump | 不改变数值状态 |

执行策略：

- CPU 路径只允许作为参考实现、单元测试 oracle、小网格调试和可重复性验证。
- GPU 路径是生产路线，不是可选增强。核心 residual、重构、通量、时间推进、线性/隐式求解器、壁面量积分最终都必须有 GPU 实现。
- 不允许为了 CPU 简单性选择会阻碍 GPU memory layout、kernel fusion、coalescing、multi-GPU domain decomposition 的数据结构。
- 早期 CPU 实现存在的唯一理由是降低物理和数值公式验证成本；一旦公式门禁通过，必须迁移到 GPU 并用 CPU/GPU bitwise 或容差回归约束。

GPU 域分解策略：

- 网格必须支持 block/partition ownership，cell、face、boundary face 和 halo cell 分开存储。
- 单 GPU 内部采用 face-list residual assembly 作为第一版，后续根据 atomic contention 切换到 coloring 或 cell-based gather。
- 多 GPU 路线采用 partition-local owned cells + halo ghost cells；每个 pseudo-time step 交换 conservative state halo。
- 边界积分和残差归约必须支持分块归约，不能依赖单个全局 host loop。
- 所有 GPU kernel 必须有 CPU reference equivalence test；性能优化不能改变物理离散。
- 高阶/DNS 阶段的 GPU 数据结构必须能承载高阶 DOF，不允许只为一阶 tet FV 写死布局。

## 3. 数据模型

### 3.1 Mesh

第一版使用显式面表，不再只依赖 tet 的四个 neighbor。

```cpp
struct CfdNode {
    float x, y, z;
};

struct CfdCell {
    int node[4];
    int first_face;
    int face_count;
    float volume;
    float cx, cy, cz;
    float h_min;
    float wall_distance;
};

enum class BoundaryKind : int {
    Interior = 0,
    Farfield = 1,
    SlipWall = 2,
    NoSlipWall = 3,
    Symmetry = 4
};

struct CfdFace {
    int left_cell;
    int right_cell;
    int node[3];
    BoundaryKind boundary;
    float area;
    float nx, ny, nz;
    float cx, cy, cz;
};
```

约束：

- 所有面法向统一从 `left_cell` 指向 `right_cell` 或外部。
- `right_cell < 0` 才允许 `boundary != Interior`。
- 所有 wall face 必须有非零 `wall_distance`。
- 网格加载后必须输出质量统计：最小体积、最大 aspect ratio、最小 `h_min`、wall face 数量。

### 3.2 State

分阶段扩展状态量，不提前混入未实现变量。

Stage E 状态：

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E]
W = [rho, u, v, w, p, T, a]
```

Stage V 状态仍为 5 个守恒量，黏性参数由 `W` 和配置计算。

Stage RANS 才允许扩展：

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E, rho*nu_tilde]
```

约束：

- `NVAR` 只在真正实现对应方程时增加。
- 所有 loops 使用 `NVAR` 或明确命名的 `NS_VAR_COUNT`，禁止混用 `NVAR=6` 但只更新 5 项。
- `cons_to_prim` 必须返回状态有效性，不允许静默 clamp 后继续当作物理解。

## 4. 数值方法路线

### Stage E0: 标量与网格基础

目标：验证网格、面法向、体积、边界标签、GPU 数据上传。

测试：

- 单 tet 体积和面法向解析校验。
- 结构化 cube wall/farfield face 数量固定。
- flat plate wall face 面积和总面积匹配几何值。
- 坏网格负体积必须 hard fail。

### Stage E1: Euler 一阶有限体积

方程：三维可压 Euler。

空间离散：

- cell-centered finite volume
- HLLC 第一版；AUSM+ 作为后续可选
- 一阶 piecewise constant

时间推进：

- 显式 pseudo-time
- 第一版使用 Forward Euler 或 SSPRK3，不用 RK4 掩盖正性问题
- 全局 dt，`dt = CFL * min(h / (|u| + a))`

边界：

- farfield：基于来流状态的超声速入口/出口第一版可简化为固定 ghost，但必须记录适用条件。
- slip wall：零法向速度，压力外推，质量通量严格为 0。

门禁：

- uniform freestream 保持精确不变。
- supersonic wedge/sphere/cube 的压力 drag 量级合理。
- 结构化对称 cube 在 α=β=0 时 `CY/CZ` 近机器零。
- `viscous=false` 后续必须回归到此层。

### Stage E2: Euler 二阶重构

空间离散：

- Green-Gauss 或 least-squares 梯度，二者必须可切换。
- Barth-Jespersen 或 Venkatakrishnan limiter。
- 重构后 `rho/p` 必须正性保护。

门禁：

- 一阶结果可复现。
- 二阶不能破坏 uniform freestream。
- 激波算例不得产生负压力。

### Stage V1: Laminar Navier-Stokes

物理：

- Sutherland viscosity。
- Fourier heat conduction。
- 常 Prandtl。
- no-slip wall。
- isothermal wall 和 adiabatic wall 都必须实现。

黏性通量：

- 面梯度采用“平均梯度 + 正交修正”。
- wall face 使用真实 wall primitive：
  - no-slip: `u_w=v_w=w_w=0`
  - isothermal: `T_w = wall_temperature / T_inf`
  - adiabatic: `dT/dn = 0`
  - pressure: `dp/dn = 0`
- wall density 由 `rho_w = gamma * p_w / T_w` 得到。

黏性时间步：

```text
dt_inv  = CFL_i * h / (|u| + a)
dt_visc = CFL_v * rho * h^2 * Re / max(mu_eff, eps)
dt      = min(dt_inv, dt_visc)
```

第一版使用全局 dt。局部时间步只能在 Euler 和 laminar NS 都稳定后再启用。

门禁：

- flat plate laminar `Cf_avg` 与 Blasius/可压修正同量级，初期允许 50%，后续收紧。
- `St` 与参考同量级。
- `Q_wall` 符号符合冷热壁设置。
- no-slip wall 速度探针接近 0。
- 禁用 viscous 后严格回归 Euler。

### Stage V2: Diagnostics

诊断不是 debug printf 的集合，而是求解器接口的一部分。

必需输出：

- residual history：L2 residual、mass residual、energy residual。
- dt history：min/max dt、dt limiter source。
- state bounds：rho/p/T/mu min/max。
- failure snapshot：首个坏 cell、face、boundary、QL/QR/flux/dt/R。
- VTK output：cell-centered rho/p/T/Mach/residual。
- wall probes：Cf/St/q_wall along x。

约束：

- diagnostics 只读，不改变数值结果。
- `OFF` 模式不得输出。
- `BASIC` 模式足够定位 NaN 首发位置。

### Stage R1: Spalart-Allmaras RANS

只有在 laminar NS 可信后进入。SA 是工程湍流闭合，不是转捩预测模型，也不是最终第一性原理真值。

状态扩展：

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E, rho*nu_tilde]
```

要求：

- SA 方程单独 residual。
- wall `nu_tilde=0`。
- freestream `nu_tilde` 明确由湍流度/黏度比设置。
- source term 点隐式或受限显式处理。
- negative `nu_tilde` 模型按 SA 标准处理，不允许简单 clamp 后当作成功。

门禁：

- turbulence=false 回归 laminar NS。
- zero `nu_tilde` 回归 laminar NS。
- turbulent flat plate `Cf` 与参考同量级。
- 不允许宣称 SA 预测了自然转捩位置。

### Stage T: 转捩与稳定性

转捩必须作为独立物理问题处理，不能由 SA 模型顺带承担。

路线：

1. 线性稳定性/LST 或 e^N 方法，用于可解释的边界层转捩估计。
2. γ-Reθ 或同类转捩模型，仅作为工程预测层，必须和 LST/DNS/实验对比。
3. DNS/WRLES 解析局部转捩机制，用于校准和验证，不作为全弹全包线日常生产计算。
4. 粗糙度、来流湍强、壁温、真实气体效应对转捩位置的敏感性必须进入误差预算。

门禁：

- 平板 Tollmien-Schlichting 或 Mack mode 增长率与参考解对比。
- 转捩位置不得只由经验常数拟合得到。
- 对同一算例必须报告 transition onset 的不确定区间，而不是单点假精确值。

### Stage HO/DNS: 高阶格式与 DNS 级验证

DNS 级精度需要两个条件同时成立：高阶低耗散格式和足够细网格。单独提高格式阶数或单独加密网格都不等于 DNS。

高阶路线：

- p=2/3/4 的 DG、FR/CPR 或高阶有限体积/WENO/TENO 路线择一实现，并保留低阶 FV 作为鲁棒基线。
- 几何必须支持高阶曲面表示；线性三角/四面体边界不能作为 DNS 几何真值。
- 时间推进必须支持高阶 SSPRK/ADER 或隐式高阶方法。
- shock capturing 必须使用可解释的传感器和局部耗散，不允许全场过耗散。

DNS 路线：

- 对低 Reynolds 或局部小域先做 DNS 验证，而不是直接承诺全飞行器全包线 DNS。
- 网格必须解析 Kolmogorov/壁面尺度；近壁要求以 `y+ < 1` 为下限，DNS 还需要满足切向和流向尺度。
- 必须做 p-refinement、h-refinement、domain-size 和 time-step convergence。
- 所有 DNS 结果必须同时报告计算域、边界条件、扰动谱、分辨率指标和耗散谱。

门禁：

- method-of-manufactured-solutions 阶精度验证。
- 等熵涡、Taylor-Green vortex、shock/vortex interaction 等高阶基准。
- DNS/WRLES 小域算例与公开参考数据或自洽谱指标对比。

### Stage H: 高温气体扩展

只在 laminar/RANS 基线、诊断和高阶基础可控后规划实施。高超声速定量热流不能只靠常 gamma + Sutherland + Fourier 得到最终精度。

顺序：

1. 变 `gamma`/热容表，仅作为诊断对照，不替代真实模型。
2. Park 5-species finite-rate chemistry。
3. 双温度模型。
4. 壁面催化。

壁面热流精度约束：

- 定量热流必须区分数值误差、气体模型误差、壁面催化/材料不确定性和网格误差。
- 壁面催化模型必须支持非催化、有限速率催化、完全催化等边界，并输出敏感性区间。
- 材料表面状态未知时，禁止输出单一“绝对精确”热流；必须输出条件化结果和不确定性。
- 辐射、烧蚀、粗糙度和表面反应率如果不建模，必须在结果元数据中声明。

该阶段不允许提前污染 E/V/RANS 基线。

## 5. 边界条件规范

所有边界条件必须显式返回 ghost primitive 和边界通量策略。

```cpp
struct BoundaryState {
    Primitive WL;
    Primitive WR;
    bool direct_flux;
    Flux direct;
};
```

slip wall：

- 反射法向速度。
- HLLC 或直接压力通量均可，但必须通过同一算例验证质量通量为 0。

no-slip viscous wall：

- 无黏部分使用直接压力通量，质量/能量无穿透。
- 黏性部分使用 wall primitive 计算剪切和热流。
- 不能用 HLLC 在 no-slip wall 上构造非物理质量通量。

farfield：

- 超声速入口：全量 freestream。
- 超声速出口：内部外推。
- 亚声速入口/出口后续单独实现，不在第一版偷用同一逻辑。

## 6. 网格策略

第一版只支持两类验证网格：

- structured cube：对称性和 wall classification 验证。
- structured flat plate：laminar boundary layer 验证。

Delaunay tetra mesh 不作为求解器稳定性门禁，直到网格质量检查和边界重建完成。

结构化 cube 必须使用 hex-cull 标记法：

1. 生成 hex grid。
2. 标记 inside-body hex。
3. 保留 outside hex 并分解为 tet。
4. face map 连接邻居。
5. 边界 face 若相邻被剔除 hex，则为 wall；若在外盒边界，则为 farfield。

禁止用“节点是否落在 ±1”或“面质心接近 ±1”判定 cube wall。

flat plate：

- wall at `z=0`。
- x/y/z 坐标必须可预测。
- wall face 总面积必须等于 `Lx * Ly`。
- 首层高度、增长比、总高度进入测试输出。

## 7. 稳定性策略

第一原则：先找实现错误，再讨论隐式架构。

必须先满足：

- uniform freestream exact preservation。
- boundary flux mass consistency。
- positive density/pressure before and after update。
- dt limiter source 可解释。
- bad state 不得被静默修复为“成功收敛”。

正性策略：

- Stage E1 可以使用保守变量回退，但必须计数并导致测试失败。
- Stage E2 起需要重构正性限制。
- Stage V1 起需要区分 convective failure 与 viscous failure。

显式推进限制：

- 高 Re 细壁面层会 stiff，显式全局 dt 很小是预期。
- 如果 V1 在正确边界和正确 dt 下仍不可接受，再评估 local time stepping、LU-SGS 或 implicit residual smoothing。
- 不允许用过大的数值耗散掩盖边界条件错误。

## 8. 测试体系

测试分层命名：

```text
CFD-MESH-*
CFD-EULER-*
CFD-VISC-*
CFD-DIAG-*
CFD-RANS-*
```

最低断言：

- 所有输出 finite。
- 残差下降达到指定倍数。
- 关键物理量在上下界内。
- 禁用新特性时回归上一阶段。

示例门禁：

| Test | Gate |
|------|------|
| `CFD-MESH-1` | all volumes > 0 |
| `CFD-MESH-2` | cube wall face count exact |
| `CFD-EULER-1` | freestream unchanged to 1e-7 |
| `CFD-EULER-2` | slip wall mass flux < 1e-7 |
| `CFD-EULER-3` | symmetric cube `CY/CZ < 1e-6` |
| `CFD-VISC-1` | no-slip wall velocity probe < tolerance |
| `CFD-VISC-2` | flat plate `0.5 < CX/Cf_ref < 2.0` initial |
| `CFD-VISC-3` | isothermal hot/cold wall heat flux sign |
| `CFD-DIAG-1` | injected bad state produces failure snapshot |

## 9. 接入策略

新 CFD solver 接入气动表生成前必须满足：

- Euler stage 通过全部门禁。
- Laminar NS 至少通过 flat plate sanity。
- `generate_aero_table(... use_fvm=true)` 有明确 feature gate。
- 输出 CSV 标明 fidelity source：engineering/newtonian/cfd。

接入后不能覆盖全部条件。第一版只允许在受支持条件范围内启用：

- Mach 范围。
- alpha/beta 范围。
- 网格类型。
- viscous/RANS 状态。

不支持条件必须 fallback 或 hard fail，不能静默输出 CFD 结果。

## 10. 文档与状态文件

- `docs/AERO_ACCURACY_UPGRADE.md`：设计文档，只描述目标、架构、验收标准。
- `docs/PLAN.md`：执行计划，勾选任务。
- `docs/progress.md`：纯追加进度日志。
- `docs/ISSUES.md`：活跃阻塞问题。

任何实现阶段完成后必须同步：

1. 编译结果。
2. 测试结果。
3. 残留风险。
4. 下一步计划。
