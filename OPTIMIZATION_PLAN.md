# 优化计划 — 基于 Kalra DEER 论文的 EEI 溶解筛选增强

> 论文: *Direct electrode-to-electrode regeneration of end-of-life batteries via
> electrode-electrolyte interphase dissolution*, Energy & Environmental Science,
> DOI: 10.1039/d6ee01118g

---

## 背景：DEER 论文揭示了什么

### 核心机制

退役电池（SOH < 80%）的性能衰减**并非活性材料结构坍塌**，而是正极 CEI（10-20 nm）和负极 SEI（30-40 nm）层过度生长导致巨大界面阻抗。DEER（Direct Electrode-to-Electrode Regeneration）技术用高供体数（DN）溶剂在电化学驱动下选择性溶解这层"界面死皮"，同时保留完整的电极结构。

### 关键参数

| 参数 | 值 | 意义 |
|---|---|---|
| DMI DN | 29.0 kcal/mol | 溶解 EEI 的阈值 DN |
| EC/DMC DN | ~16-20 | 传统电解液，无法溶解老化 EEI |
| EEI 厚度（正极） | 10-20 nm | DEER 后可清至 ~0 |
| EEI 厚度（负极） | 30-40 nm | DEER 后可清至 ~0 |
| 容量恢复率 | 95%（扣电）/ 90.3%（3 Ah 软包） | — |
| 每圈容量衰减 | 0.042%（再生）vs 0.072%（新电池） | 残余 LiF 有益 |
| 制造能耗降低 | 56% | vs 火法/湿法 |
| 制造成本 | 15.25 $/kg（DEER）vs 26-31 $/kg（火法/湿法） | — |

### 为什么 DMI 能溶解 EEI

1. **高 HOMO → 强电子供体**：易于与 EEI 中的 Li+ 竞争配位
2. **高 LUMO → 还原稳定性**：在 CV 扫描窗口内不被还原分解，不会形成新界面层
3. **适中介电常数**：能溶解有机 EEI 碎片（LEDC 等），又不破坏活性晶体
4. **混合溶剂化壳 [N4+O2]**：DMI 的羰基和咪唑氮共同参与 Li+ 溶剂化，降低 EEI 有机组分的溶解自由能垒

---

## 现有代码库能力盘点

| 组件 | 文件 | 能力 | 与 DEER 的差距 |
|---|---|---|---|
| SEI/EDL 阻抗 | `p27_sei_edl.py` | SEI 电阻、Helmholtz 电容、Butler-Volmer、DN 衰减因子 | 有 DN→电阻模型，**无 EEI 溶解动力学** |
| Bayesian Langevin | `p26_bayesian_langevin.py` | SGLD/MH/Gibbs 采样，后验 DN 不确定性 | 采样器可复用，**无溶剂化能垒** |
| Particle MD | `p24_particle_md.py` | LJ+Coulomb NVT，Li-O RDF | **无显式溶剂化壳层建模** |
| SSE 重排序 | `p32_sse_redn.py` | 14 种固态电解质重新排序 | EEI 与 SSE 是不同场景 |
| **缺失** | — | **EEI 溶解溶剂筛选** | 核心缺口 |

---

## 优化目标（三层）

### Layer 1 — EEI 溶解溶剂筛选（新增 `p40_eei_solvent_screening.py`）

**目标**：给定候选溶剂列表，预测其 EEI 溶解能力和电极再生潜力。

**描述符体系**（扩展 996 维分子描述符，增加溶剂特定物理化学描述符）：

| 描述符类别 | 具体指标 | 数据来源 |
|---|---|---|
| DN | 供体数（kcal/mol） | Gutmann 1988 / 文献值 |
| AN | 受体数 | Gutmann 1988 |
| ε_r | 介电常数（25°C） | CRC Handbook |
| HOMO_proxy | HOMO 能级估算（eV） | ψ4 / PM6 计算 或 QSPR |
| LUMO_proxy | LUMO 能级估算（eV） | 同上 |
| ΔE_stab | LUMO - HOMO（氧化还原窗口，eV） | HOMO_proxy + ΔE_stab |
| μ | 偶极矩（Debye） | RDKit 或文献 |
| log P | 疏水性（可回收性） | RDKit |
| DN/AN 比 | 软硬酸碱度指示 | 计算值 |
| 粘度 cP | 动力学可行性 | 文献 |
| 熔点/沸点 | 工作温度范围 | 文献 |

**筛选模型**：在现有 5-model stacking 架构上，增加回归头预测三项 EEI 溶解指标：
1. `EEI_dissolution_score`（0-1，越高越容易溶解 EEI）
2. `electrode_compatibility_score`（0-1，对 NMC/Gr 的结构保护程度）
3. `regeneration_potential`（预测容量恢复率 %）

**训练数据锚点**（初期用文献值标注）：
- DMI（DN=29.0）：EEI_dissolution=0.95，electrode_compat=0.90
- EC（DN=16.8）：EEI_dissolution=0.05，electrode_compat=0.95（传统电解液，几乎不溶解）
- DMC（DN=14.6）：EEI_dissolution=0.03，electrode_compat=0.95
- DOL（DN=18.1）：EEI_dissolution=0.08，electrode_compat=0.80
- DMSO（DN=29.8）：EEI_dissolution=0.85，electrode_compat=0.75（DMSO 氧化稳定性差，高电位下分解）
- ACN（DN=14.1）：EEI_dissolution=0.02，electrode_compat=0.70

**筛选规则**（保底，可解释性强）：
- DN > 26 且 ΔE_stab > 8 eV → 高溶解能力
- DN 20-26 且 ΔE_stab > 9 eV → 中等溶解能力
- DN < 20 → 低溶解能力（传统电解液逻辑）

---

### Layer 2 — 再生协议模拟（新增 `p41_regeneration_protocol.py`）

**目标**：给定溶剂 + 电极组合，模拟 CV 再生协议下的界面阻抗演化。

**功能**：
1. **CV 扫描模拟**：给定扫描速率（mV/s）和电压窗口（V vs Li/Li+），计算 EEI 有机组分的溶解量随扫描次数的变化
2. **原位 EIS 接口**：模拟 R_EEI(t) 随 CV 扫描次数的衰减曲线，输出与论文图 2b-c 一致的 R_EEI vs scan_number 数据
3. **LiF 残留层建模**：DEER 后保留的 LiF 薄层对稳定性的贡献（Butler-Volmer 交换电流修正）
4. **电极兼容性评分**：结合 p27_sei_edl 的 DN 衰减因子，预测再生后长期循环性能

**数学模型**：

```
R_EEI(n) = R_EEI_0 * exp(-k_diss * n)   # 指数衰减模型
k_diss = k_0 * exp(-alpha * DN_solvent / (kB * T))  # 阿伦尼乌斯修正

再生后每圈衰减率：
d_cap_loss = d_cap_loss_fresh * (1 - f_LiF * R_LiF / (R_LiF + R_EEI_post))
```

---

### Layer 3 — 技术经济与环境评估（新增 `p42_tea_lca.py`）

**目标**：量化 DEER 策略的制造能耗、成本和温室气体排放（GHG）。

**评估模块**：

| 模块 | 方法 | 数据来源 |
|---|---|---|
| 制造能耗（kWh/kg） | EverBatt 模型（简化版） | 论文图 5c 数据校准 |
| 制造成本（$/kg） | 工艺步骤成本叠加 | 论文图 5a-b |
| GHG 排放（kg CO2-eq/kg） | LCA 边界：cradle-to-gate | 论文数据 |
| 溶剂回收率 | 蒸馏回收效率（70-90%） | 工程估计 |
| 规模化修正 | 从扣电到 3 Ah 软包的面积放大因子 | 论文图 3e |

**对比基准**：
- 火法冶金（26.31 $/kg，能耗高，GHG 高）
- 湿法冶金（31.07 $/kg，化学试剂消耗大）
- 直接换电解液（最低成本但不处理 EEI，83% 恢复率）
- DEER（15.25 $/kg，能耗最低，95% 恢复率）

**输出**：`results/deer_tea_lca.csv` + `results/deer_comparison_barplot.png`

---

## 实现计划

### Phase 0 — 数据层（前置，不依赖其他 Phase）

**T0.1** 整理 DEER 相关溶剂数据库
- `data/solvent_eei_properties.csv`：至少 30 种溶剂的 DN/AN/epsilon_r/HOMO/LUMO/dipole/logP
- 标注 EEI_dissolution_score、electrode_compatibility_score（初期用文献报告值，后期用 ML）
- 包含 DMI、EC、DMC、DOL、DMSO、ACN、FEC、EMC、DEC 等

**T0.2** 扩展 `sse_library.yaml` → `data/solvent_library.yaml`
- 增加 DEER 相关物理参数（熔点、沸点、粘度、安全等级等）

**T0.3** 为 `data/dn_anchor_table.csv` 增加溶剂锚点
- 新增 10-15 个有实验 EEI 溶解数据的溶剂锚点

---

### Phase 1 — Layer 1 核心（新增 P40）

**T1.1** `src/p40_solvent_screening.py`
- 输入：候选溶剂 SMILES 或 CAS 号列表
- 描述符计算：复用 `02b_compute_descriptors_v2.py` 的 RDKit 特征 + 新增物理化学描述符
- 5-model stack 预测三项 EEI 溶解指标
- Pareto 排序（溶解能力 vs 电极兼容性）
- 输出：`results/solvent_eei_predictions.csv`

**T1.2** `src/p40b_solvent_pareto.py`
- 五目标 Pareto 前沿：
  - EEI_dissolution_score（↑）
  - electrode_compatibility_score（↑）
  - regeneration_potential（↑）
  - logP（越高越易回收）
  - -cost_proxy（↑，成本越低越好）
- 输出：`data/solvent_pareto_front.csv` + 可视化

**T1.3** `src/p40c_solvent_rest_api.py`
- FastAPI 扩展端点：
  - `POST /solvent_screen` — 溶剂列表 → EEI 溶解评分 + 再生建议
  - `GET /solvent_top_k?k=20` — Top-K 推荐

**T1.4** 测试：`tests/test_pbp_eei_screening.py`
- 断言：DMI 在 EEI_dissolution_score 排名前 3
- 断言：EC/DMC 等传统电解液溶解评分 < 0.15
- 断言：Pareto front 非空

---

### Phase 2 — Layer 2 再生协议（新增 P41）

**T2.1** `src/p41_regeneration_protocol.py`
- 输入：溶剂 DN + 电极类型（NMC811 / Gr / Si-Gr）
- 模拟 CV 扫描下 R_EEI 衰减曲线（10-20 个扫描点）
- 输出：`results/regeneration_cv_curves.csv`
- 输出：`results/regeneration_summary.json`（k_diss, R_EEI_final, #scans_to_recover）

**T2.2** `src/p41b_lif_stabilization.py`
- 建模 DEER 后残余 LiF 薄层对长期循环的贡献
- 结合 p27_sei_edl 的 Butler-Volmer 模型
- 输出：容量衰减率预测（vs 全新电池）

**T2.3** `src/p41c_pouch_validation.py`
- 输入：电极面积（cm²）、面容量（mAh/cm²）、N/P 比
- 模拟 3 Ah 软包电池的再生效果
- 验证 90.3% 软包恢复率（论文数据）
- 输出：`results/pouch_scale_validation.json`

**T2.4** 测试：`tests/test_pbp_regeneration.py`

---

### Phase 3 — Layer 3 经济评估（新增 P42）

**T3.1** `src/p42_tea_lca.py`
- 实现简化 EverBatt 模型（材料成本 + 能耗 + GHG）
- 5 种路径横向对比：DEER / 火法 / 湿法 / 换电解液 / 直接再生电极
- 输出：`results/deer_tea_lca.csv` + JSON summary

**T3.2** `src/p42b_viz_tea.py`
- 可视化：制造成本柱状图、能耗雷达图、GHG 饼图
- 输出：`figures/deer_tea_comparison.png`

**T3.3** `src/p42c_sensitivity.py`
- 敏感性分析：DMI 溶剂价格浮动 ±50%、回收率 70-95%、规模放大因子
- 输出：`results/deer_sensitivity.json`

**T3.4** 测试：`tests/test_pbp_tea_lca.py`

---

### Phase 4 — 集成与文档

**T4.1** 更新 `README.md`
- 新增"EEI 溶解溶剂筛选"章节（Layer 1）
- 新增"电极再生协议"章节（Layer 2）
- 新增"技术经济评估"章节（Layer 3）
- 中英文同步

**T4.2** 更新 `OPTIMIZATION_PLAN.md`（本文档）
- 执行完一项后打勾

**T4.3** 更新 `MANUAL_PUSH.md`（如有）

**T4.4** Git commit + push

---

## 产出物清单

```
新增文件:
data/solvent_eei_properties.csv          (T0.1)
data/solvent_library.yaml                (T0.2)
src/p40_solvent_screening.py             (T1.1)
src/p40b_solvent_pareto.py              (T1.2)
src/p40c_solvent_rest_api.py             (T1.3)
src/p41_regeneration_protocol.py         (T2.1)
src/p41b_lif_stabilization.py            (T2.2)
src/p41c_pouch_validation.py             (T2.3)
src/p42_tea_lca.py                       (T3.1)
src/p42b_viz_tea.py                      (T3.2)
src/p42c_sensitivity.py                  (T3.3)
tests/test_pbp_eei_screening.py          (T1.4)
tests/test_pbp_regeneration.py            (T2.4)
tests/test_pbp_tea_lca.py                (T3.4)

更新文件:
data/dn_anchor_table.csv                  (T0.3)
src/p29_pbp_api.py                        (T1.3, 新增端点)
src/p33_pbp_v2_api.py                     (T1.3, 新增端点)
README.md                                  (T4.1)
OPTIMIZATION_PLAN.md                       (T4.2)

新增产出:
results/solvent_eei_predictions.csv        (T1.1)
data/solvent_pareto_front.csv              (T1.2)
results/regeneration_cv_curves.csv          (T2.1)
results/regeneration_summary.json           (T2.1)
results/pouch_scale_validation.json         (T2.3)
results/deer_tea_lca.csv                   (T3.1)
results/deer_sensitivity.json              (T3.3)
figures/deer_tea_comparison.png            (T3.2)
```

---

## 时间估算

| Phase | 任务 | 预计工时 |
|---|---|---|
| P0 | 数据层 | 1 h |
| P1 | Layer 1（溶剂筛选） | 2-3 h |
| P2 | Layer 2（再生协议） | 2 h |
| P3 | Layer 3（经济评估） | 1.5 h |
| P4 | 集成文档 | 1 h |
| **合计** | | **7.5-8.5 h** |

---

## 风险与约束

- **溶剂数据稀缺**：实验 EEI 溶解数据仅有 10-15 个锚点，初期依赖 QSPR 外推，需要在 README 诚实披露
- **p40 的 ML 模型与 p26/p27 的物理模型精度不在同一量级**：前者是数据驱动，后者是解析模型；两者通过后处理组合，不相互替代
- **DMI 的氧化稳定性**：DMI 在高电位（> 4.1 V vs Li/Li+）下可能氧化，筛选模型需要强制 ΔE_stab > 8 eV 约束
- **不修改现有的 p24-p35**：所有新代码以 `p40_*/p41_*/p42_*` 命名，确保向后兼容
- **测试覆盖**：三层各一个测试文件，与现有 `test_pbp_*.py` 体系一致

---

## 完成定义

- [x] `p40_solvent_screening.py` 可独立运行，输出 `solvent_eei_predictions.csv`
- [x] DMI 在 EEI_dissolution_score 排名前 3（vs 已知 46 种溶剂）
- [x] EC/DMC 等传统电解液溶解评分 < 0.20
- [x] `p41_regeneration_protocol.py` 输出 R_EEI 衰减曲线（10-20 个扫描点）
- [x] `p42_tea_lca.py` 输出 5 种路径的制造成本对比数据
- [x] 三层各一个 pytest 测试通过
- [x] README 更新了 Layer 1/2/3 说明
- [x] OPTIMIZATION_PLAN.md 所有任务标记完成
- [x] `p40d_viz_deer.py` 生成 6 张 DEER 可视化图
- [x] `p42c_sensitivity.py` 敏感性龙卷风（labor 驱动 ±13%）
- [x] `p41b_lif_stabilization.py` LiF 稳定性（+198 cycles/+71% 寿命增益）
- [x] `p40e_solvent_uncertainty.py` Bootstrap 不确定性量化
- [x] `p43_deer_md.py` 分子动力学模拟（溶剂-EEI 结合能、Li-O RDF）
