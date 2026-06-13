# Stage C — GPU NS + RANS + 热化学非平衡

## C.0 — 验证基础设施 [2026-06-14]
- 创建 `tests/ref_data/` 目录，含 5 个参考数据头文件
  - `flat_plate_laminar_ref.hpp`: Blasius + van Driess 可压缩修正
  - `flat_plate_turbulent_ref.hpp`: van Driest II 湍流 Cf
  - `normal_shock_cea_ref.hpp`: Rankine-Hugoniot + Park 5-species 平衡组分
  - `sphere_heatflux_ref.hpp`: Fay-Riddell 驻点热流
  - `hemi_cylinder_ref.hpp`: 修正 Newtonian Cp(θ)
- 创建 `tests/test_cfd_ns.cpp`，含 9 个测试
  - 5 项参考数据完整性检查 (层流/湍流 Cf、正激波、球头热流、半柱 Cp)
  - 2 项欧拉基线测试 (滑移壁面 α=0 和 α=10°)
  - 1 项 FVM 与牛顿面板法交叉验证
- 添加 CMake 目标 `TestCfdNs`，含 DLL POST_BUILD
- **全部 9/9 C.0 测试通过**

## 计划

| 子阶段 | 状态 |
|--------|------|
| C.0 验证基础设施 | ✅ 完成 |
| C.1 NS 黏性通量 + 无滑移壁面 | ⏳ 下一步 |
| C.2 SA 一方程湍流模型 | 待开始 |
| C.3 黏性数值重构升级 | 待开始 |
| C.4 Park 5-species 化学非平衡 | 待开始 |
| C.5 双温度热非平衡 | 待开始 |
| C.6 壁面有限催化 | 待开始 |
| C.7 棱柱边界层网格 | 待开始 |
| C.8 高阶格式 + 隐式推进 | 待开始 |
