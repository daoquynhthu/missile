# AeroSim 修复清单

按依赖关系从根本问题到边缘问题排列。优先级数字不代表工作量，代表**必须先修这个才能修其他的**。

---

## Level 1 — 根基问题（必须先修，否则后续修复无效）

### L1-0 GPU AeroSolver 纠正（气动数据来源的根本问题）
**文件**: `src/aero_solver/aero_solver.cu`, `include/aero_solver/aero_solver.hpp`  
**现状**: 整个 `aerodynamics_table.csv` 的数据是无效的（见 PIPELINE.md 问题4）。

具体缺陷:
1. **牛顿撞击法在 Mach<5 下无效**: `aero_solver.cu:34-88` 只适用于高超声速，但 CSV 表 Mach 起点 0.5
2. **力矩参考点错误**: `aero_solver.cu:85` `moment_term = cross(tri.center, force_term)` — 参考点是 (0,0,0) 即 STL 几何原点，而不是质心
3. **对称体在 β=0 时 CY/Cn 不为零**: `aero_solver.cu` 核函数对左右半模的对称求和有问题
4. **Cp 公式不正确**: `Cp = 2.0f * dot_val * dot_val` 是简单的修正牛顿公式，未考虑马赫数效应和比热比

**修复要求**:
1. 明确 AeroCalc/AeroSolver 的适用范围（仅 Mach>5），低马赫数使用解析法或工程估算
2. 力矩参考点改为可配置的质心位置，而不是固定在几何原点
3. 修复对称体的 CY/Cn 求和 — 在核函数中检查，或确认 STL 模型对称性
4. 重新审视 Cp 公式，至少添加马赫数相关的修正项
5. **数据验证**: 用已知基准（如锥体/球体的理论 Cp）验证单个三角形面元的力计算正确性

### L1-1 末端比例导引(PN)重写
**文件**: `include/gnc/guidance.hpp` lines 363-517  
**现状**: 一稿草稿，代码中十几条自我质疑注释（"Wait...", "Sign?", "Let's guess alpha"），重力补偿硬编码9.81，AoA映射靠猜比例  
**修复要求**:
1. 删除所有草稿注释
2. 用实际 EGM2008 重力矢量替换 `-9.81 * up`
3. 使用标准 PN 加速度公式 `a_cmd = N * Vc × ω_los`，明确坐标符号
4. 加速度→姿态指令的映射使用 `L = M * (a_cmd - g)` 再通过气动系数 `Cl_alpha` 反算 AoA，或基于动压做限幅
5. 添加与 GLIDE 阶段的平滑交班逻辑

### L1-A Dart 气动表数据填充
**文件**: `data/dart/rm_dart_aero_table.csv`, `scripts/dart/generate_dart_aero_table.py`  
**现状**: CSV 表只有 2 条数据（Mach=0.1, Alpha=±3°），31 格 Alpha 和 6 格 Mach 网格全部缺失。`DartAeroTable` 的双线性插值退化为常量。

**修复要求**:
1. 重新运行 `generate_dart_aero_table.py` 生成完整网格
2. 或确认 CSV 文件为何只有 2 行（可能是生成脚本未完整执行）

### L1-B SubsonicSolver 收敛伪造
**文件**: `src/subsonic_solver/subsonic_solver.cu:497`  
**现状**: GPU 核函数中强行写 `result.converged = 1; result.residual = entry.residual_target * 0.2;`，不是真实的收敛判定。

**修复要求**:
1. `converged` 改为基于耦合迭代前后差的真实判断
2. 或添加注释声明该求解器为"非迭代工程估算"，不声称收敛

### L1-2 气动表插值实现
**文件**: `include/aerodynamics_model.hpp` line 314  
**现状**: 显式 `// TODO: Interpolate`，实际永远取 `cd0_table[0]`  
**修复要求**:
1. 在 `compute_coeffs_analytical()` 中实现基于 Mach 的 1D 插值（`cd0_table`, `cl_alpha_table`, `xcp_table`）
2. 如果本函数只用解析模型，应该也支持从 `cl_table_2d`/`cd_table_2d`/`cm_table_2d` 做 2D 查表
3. 或者删掉 `missile_config.cpp` 中从未被消费的表格生成代码

### L1-3 配置生成的数据从未被消费
**文件**: `src/missile_config.cpp` lines 141-200  
**现状**: `cl_table_2d`/`cd_table_2d`/`cm_table_2d` 生成了完整的 Mach×Alpha 气动数据，但 `AerodynamicsModel` 既不读这些 vector，也不用它们做插值  
**修复要求**:
- **方案A**: 让 `AerodynamicsModel` 直接使用这些内存表做插值（删掉 CSV 依赖）
- **方案B**: 删掉生成代码，全部走 CSV 文件（确保 CSV 文件存在并路径正确）
- **决策**: 取决于是否想在编译时确定气动数据

---

## Level 2 — 仿真正确性问题

### L2-1 滑翔段"海豚跳"振荡
**文件**: `include/gnc/guidance.hpp` lines 235-360  
**现状**: HANDOFF.md 确认存在剧烈高度振荡，PID 增益需要精细调节  
**修复要求**:
1. 添加高度误差的微分项（当前只有比例项 `kp_alt`）
2. 限制最大 AoA 指令变化率（rate limiter）
3. 引入能量管理调度：根据剩余能量动态调整参考高度剖面

### L2-2 COAST→GLIDE 相位切换 AOA 跳变
**文件**: `include/gnc/guidance.hpp` lines 210-213  
**问题**: 当 `vertical_vel < 0 && alt < 80000` 时，AOA 从 0 度瞬跳 20 度  
**修复要求**:
1. 添加过渡期平滑（例如 5 秒内线性 ramp 到目标 AoA）
2. 或者基于动压/高度做渐进式调度

### L2-3 BOOST→COAST 滞回阈值不合理
**文件**: `include/gnc/guidance.hpp` line 550  
**问题**: `vertical_vel > -10.0` 作为 COAST 判定条件。导弹如有显著下沉率（例如 -8 m/s）仍被归为 COAST  
**修复要求**:
1. 将阈值缩小到 `-1.0 m/s` 或基于高度判定
2. 或者用高度+速度矢量方向综合判定

### L2-4 所有路径硬编码
**文件**: 
- `src/main.cpp:162` → `"e:/missile/data/EGM2008.gfc"`
- `src/missile_config.cpp:125` → `"e:/missile/aerodynamics_table.csv"`
- `src/missile_config.cpp:127` → `"e:/missile/hgv_model_optimized.stl"`
- `src/rm_dart_sim.cpp:44-45` → `"e:/missile/data/..."`

**修复要求**:
1. 统一到基于可执行文件相对路径或 CMake 定义的资源路径宏
2. 或添加命令行参数覆盖
3. 验证所有文件确实存在于仓库中（例如 `aerodynamics_table.csv` 是否存在？）

### L2-5 RM Dart使用恒定重力和恒定大气
**文件**: `src/rm_dart_sim.cpp` lines 185, 206  
**现状**: 全程使用发射点的重力矢量和大气密度  
**修复要求**:
1. 每步计算当前高度的 `AtmosphereModel::calculate_ussa76(alt)`
2. 使用 `GravityModel::calculate_acceleration()` 替代固定重力
3. 注意性能（dart 仿真很短，影响可忽略）

---

## Level 3 — 简化与死代码清除

### L3-1 9-PID Autopilot 简化
**文件**: `include/gnc/autopilot.hpp`  
**现状**: 3 种执行器 × 3 轴 = 9 个 PID，互相独立，配置重复  
**修复要求**:
- **最小方案**: 合并为 3 个 PID（pitch/yaw/roll），输出分发到各执行器
- **推荐方案**: 改为 `ActuatorAllocator` 模式：PID 输出目标力矩，按优先级分配到 TVC/RCS/Aero
- 删除未使用的 `set_aero_pids()` 方法

### L3-2 GPU AeroSolver 确认用途
**文件**: `src/aero_solver/aero_solver.cu`, `include/aero_solver/aero_solver.hpp`  
**现状**: 实现完整但主程序设置 `use_cuda_solver = false`，从未被使用  
**修复要求**:
- **方案A**: 如果GPU求解器有独立工具价值（`AeroCalc.exe`），保留但明确文档
- **方案B**: 从主仿真代码中删除所有对它的引用，避免误导

### L3-3 重力场单点CUDA函数清理
**文件**: `include/gravity_model.hpp` line 27  
**现状**: `calculate_acceleration_cuda()` 标注 "Deprecated" 但仍在头文件暴露  
**修复要求**: 删除声明和实现，或改为内联调用 batch 版本

### L3-4 SolidMotor::get_inertia() 死函数
**文件**: `include/propulsion_model.hpp` lines 151-176  
**现状**: 完整的质量-惯量估算函数，但主程序从没调过  
**修复要求**: 删除或用起来

### L3-5 `m_propellant_remaining` 未使用
**文件**: `include/propulsion_model.hpp` line 300  
**修复要求**: 删除

### L3-6 推力日志占位符
**文件**: `src/main.cpp` line 316  
**现状**: `double thrust = 0.0; // Placeholder`  
**修复要求**: 从 `SolidMotor::compute()` 返回值中取实际推力，或删掉CSV中的推力列

### L3-7 CMake DLL拷贝重复
**文件**: `CMakeLists.txt` lines 120-149  
**现状**: 7 个 `add_custom_command` 做同一件事  
**修复要求**: 用函数封装或 loop 生成

### L3-8 RM Dart批查表成员变量可简化
**文件**: `include/rm_dart_config.hpp` lines 68-78  
**现状**: `lookup_err_1..5`, `lookup_dp_1..6` 等 20 个独立 double  
**修复要求**: 改为 `std::vector<double>` 或 `std::array<double, N>`

---

## Level 4 — 代码质量改进

### L4-1 惯性矩阵求逆优化
**文件**: `src/dynamics_6dof.cu` line 52  
**现状**: 每步 `inertia.inertia.inverse()` 求 3×3 逆矩阵  
**修复要求**: 预计算并存储 `inertia.inv_inertia`，或只在质量变化时更新

### L4-2 PID添加微分滤波器
**文件**: `include/utils/pid.hpp`  
**现状**: `update_with_rate()` 直接 `-Kd * rate_pv`，无低通滤波  
**修复要求**: 添加一阶低通 `d_term += (raw_d - d_term) * dt / tau`

### L4-3 AeroSolver STL解析缺乏格式校验
**文件**: `src/aero_solver/aero_solver.cu` lines 211-262  
**现状**: 假设一定是二进制 STL，遇到文本格式静默解析出乱码  
**修复要求**: 检查文件前 5 字节是否为 `"solid"`，分别走文本/二进制路径

### L4-4 `compute_coeffs_analytical()` 力系数符号审查
**文件**: `include/aerodynamics_model.hpp` lines 302-336  
**现状**: Body frame 力系数推导使用混合坐标系假设，需要与 `dynamics_6dof.cu` 中力的使用方式对齐验证

### L4-5 aerodynamic 调试计数器残留
**文件**: `include/aerodynamics_model.hpp` lines 383-384  
**现状**: `static int call_count` 即使禁用了打印也在每次气动调用时递增  
**修复要求**: 用 `#ifdef DEBUG` 包裹或删除

---

## Level 5 — 扩展功能（原始 PLAN.md 中未完成的）

### L5-1 气动热（Fay-Riddell 驻点热流）
**文件**: `PLAN.md` 提及，未开始  
**必要性**: 高，因为拉起机动受热流约束

### L5-2 蒙特卡洛全面打靶
**文件**: `PLAN.md` 提及  
**现状**: 仅 dart 子系统的 GPU 蒙特卡洛完成，HGV 主系统多弹道是 deterministic dispersion

### L5-3 自适应步长积分器验证
**文件**: `include/dynamics_6dof.hpp` lines 142-299  
**现状**: DP54 已实现但有强制容错 `forced acceptance` 逻辑，未与 RK4 做精度对比验证

### L5-4 能量守恒检验
**文件**: `PLAN.md` section 4  
**现状**: 缺失，无积分误差监控

---

## 建议执行顺序

```
Phase 0 (气动根基 — 必须先确认气动系数可靠)
  L1-0  GPU AeroSolver 纠正 (力矩参考点 + 对称性 + Cp 公式)
  确认 aerodynamics_table.csv 数据有效 (CD~0.05@Mach0.5α0°, 对称体CY=Cn=0@β=0)
  或直接删掉 CSV 表，改用 missile_config.cpp 生成的解析数据（打通死代码）
  L1-A  Dart 气动表数据填充（补全 6Mach×31Alpha）
  L1-B  SubsonicSolver 收敛伪造标记明确

Phase 1 (制导正确性)
  L1-1   末端 PN 重写
  L1-2   气动表插值实现（解析法至少用 1D 插值）
  L1-3   配置数据消费打通（删死代码或用起来）
  L2-4   路径硬编码 → 相对路径

Phase 2 (飞行稳定性)
  L2-1   海豚跳振荡
  L2-2   AOA跳变平滑
  L2-3   滞回阈值
  L5-3   自适应积分器验证
  L5-4   能量守恒检验

Phase 3 (精度提升)
  L2-5   Dart环境模型实时化
  L4-1   惯量求逆优化
  L4-4   力系数符号审查

Phase 4 (清理简化)
  L3-1   9-PID简化
  L3-2~L3-8 死代码删除
  L4-2   PID滤波器
  L4-3   STL格式校验
  L4-5   调试计数器

Phase 5 (扩展)
  L5-1   气动热
  L5-2   全系统蒙特卡洛
```
