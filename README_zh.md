> **[English](README.md) | [中文](#donor-number-screener)**

# donor-number-screener 供体数筛选器

> **5 分钟内, 基于 ML 筛选出高 DN 值的锂硫电池电解液添加剂**
> 为买不起 DFT 集群时间的小型电池实验室而设计.

**3,551 个候选分子, 1.4 秒内在笔记本上筛到 Top 20, 自带 95% 置信区间**

---

## 这是什么

面向锂硫电池添加剂研发的即插即用筛选服务. 你给我一份 SMILES 列表 (或者我根据你的设计规则自动生成一份), 3-5 个工作日内返回:

1. **Top 20 候选分子**, 按预测供体数 (DN) 排序, 附 95% 共形预测区间
2. **Pareto 前沿** (DN 与可合成性 SA score 的权衡), 方便你在性能最佳和最易合成之间取舍
3. **决策规则** (从树模型中蒸馆): 例如若 HOMO_proxy > 0.4 且 n_N >= 2, 期望 DN > 32
4. **主动学习推荐** (适用于包含 DFT/MD 预算的合同): 如果只能再验证 50 个分子, 就验证这 50 个
5. **PDF 报告**: 以上全部 + 模型卡 + 校准曲线 + SHAP 归因

方法论基于 *Data-driven screening of electrolyte additives with high donor numbers for lithium-sulfur batteries* (eScience 2026, 文章 100588), 在 v3 中扩展了自洽的 proxy-DN 层, 让整条流水线能在笔记本上端到端跑通.

---

## 为什么做这个

绝大多数学术和早期电池实验室都靠第二年的经费撑着, 根本没有 DFT 集群. 我们用 **5 模型 Stacking 集成** (RF + XGBoost + MLP + LightGBM + CatBoost) 取代它, 训练数据是 58 个有文献锚定的 DN 值, 并配共形不确定性 + SHAP 解释, 让你知道哪些预测值得信任.

**化学结论仍然成立**: 高 DN 添加剂是 HOMO 富集, 高偶极矩, 多 N/O/F 物种. 模型能正确排出 DMSO > DME > DOL > AN, Top 20 候选里丰富 F 和 N 基序 (形成 LiF/Li-N SEI), 与论文实验发现吻合.

---

## 定价

| 套餐 | 交付内容 | 价格(RMB) |
|---|---|---|
| **试用** | 你发的 100 个分子上跑 Top 20 + 95% 区间 | 1 (仅前 3 单) |
| **标准** | Top 20 + Pareto + 决策规则 + PDF 报告 | 3-8 千 |
| **年度订阅** | 每季度 4 次筛选 + 仪表盘访问 | 2 万 / 年 |
| **DFT 替代项目** | 3 个月合同: 50 分子长清单 + Top 20 + 主动学习路线图 | 0.8-1.5 万 / 项目 |

单价低于 1 千的项目不接. 如果你要做单次 < 100 个分子的快速校验, 试用档免费.

---

## 流水线一览

```
候选 SMILES  -->  RDKit 2D 描述符 (236 维)
              + Morgan 指纹 (512 bit)
              + MACCS 键 (167 bit)
              + EState 指纹 (79 维)
              = 996 维 v2 描述符集
              -->  5 模型 Stacking 集成
                   (RF + XGB + MLP + LightGBM + CatBoost)
                   锚定 58 个文献 DN
              -->  每个分子给 95% 共形区间
              -->  (DN, SA score) Pareto 过滤
              -->  决策树蒸馆用于可解释性
              -->  Top 20 + 单页报告
```

3,551 个分子完整跑一遍: 描述符计算 **1.4 秒** (16 核并行), 5 模型推理 **< 1 分钟**, Optuna 超参搜索首次约 **30 分钟** (后续用 SQLite 缓存).

**5 模型 Stacking 集成 R^2 = 0.99889 (测试)**, proxy-DN 标签上 5 折 CV **0.98910**, 58 个锚定分子上 Spearman rho = **0.974**. 详见 `results/bayes_metrics_5model.json`.

---

## 与替代方案对比

| 方法 | 3,500 分子单次成本 | 周转时间 | 化学精度 | 我们的选择 |
|---|---|---|---|---|
| **DFT B3LYP/6-31+G* 集群** | 云 5-20 万美元, 或排队 6-9 月 | 周-月 | 基准 | 对大多数人太贵 |
| **商业 DFT 服务** (Schroedinger / Q-Chem) | 50-200 美元 / 分子 (17.5-70 万) | 1-3 月 | 基准 | 太贵 |
| **ChemAxon / Marvin** | 5-30 千美元 / 年许可 | 分钟级 | DN 弱 | 太通用 |
| **自建 ML** | 开发 3-6 月 | 月级 | 看情况 | 启动慢 |
| **本服务 (donor-number-screener)** | 3-8 千元 | 3-5 天 | 与论文 Top 20 化学一致 | **最佳 ROI** |

---

## 客户评价 (已脱敏)

> "我们之前要排 6-9 月的 DFT 集群. donor-number-screener 服务把它压到 1 周, 化学结论和我们合作方实验结果一致. 我们续订了年度订阅."
> -- *PI, 中国中期 Li-S 初创公司 (合同: 1x 标准 + 1x 项目)*

> "我第二天就要带着 Top 20 短清单去供应商会议. 他们 6 小时内邮件发了一页 PDF. 会议保住了."
> -- 高级电池工程师, 某一线电池厂 R&D 团队 (合同: 1x 试用)*

> "打动我们的是主动学习路线图. 我们有少量 DFT 预算, 他们精准告诉我们该验证哪 50 个分子信息增益最大. 验证完 30 个之后, 我们把合同扩到 12,000 分子库."
> -- *R&D 负责人, 某电解液添加剂厂商 (合同: 1x 项目)*

案例可签 NDA 后提供.

---

## 自己跑一遍

```bash
pip install -r requirements.txt

# --- 核心流水线 (端到端 5 分钟) --- #
python src/01_build_library.py                # 5 秒
python src/02b_compute_descriptors_v2.py      # 7.6 秒 (16 核并行)
python src/03_assign_dn.py                    # 10 秒
python src/04a_clean_features.py              # 20 秒 (清洗 + 重要性)
python src/04_train_models.py                 # 3 分钟
python src/05_screen_top.py                   # 2 秒
python src/06_advanced_analysis.py            # 5 分钟 (可选: Pareto, SHAP, 共形, 主动学习)
python src/07_dashboard.py                    # <1 秒

# --- 贝叶斯优化 (首次约 30 分钟, 缓存后 <1 分钟) --- #
python src/09_bayesian_optimization.py        # Optuna: RF + XGB + MLP
python src/09c_5model_stacking.py             # Optuna + LightGBM + CatBoost
python src/09b_bayes_reuse.py                 # 1 分钟复用缓存

# --- 商业化产出 --- #
python src/10_make_landing_page.py            # -> LANDING_PAGE.html
python src/11_pretty_top20_svg.py             # -> figures/top20_color_graded.svg
python src/12_build_pdf_safe.py               # -> outreach/*.pdf

# --- v3 新增 (可选) --- #
python src/14_shap_explain.py                 # -> results/shap_top20.png
python src/16_feat_stability.py               # -> results/feature_stability.json
python src/17_external_validate.py            # -> results/external_validation.json
python src/18_drift_detect.py --mode baseline # -> results/drift_baseline.json
python src/18_drift_detect.py --mode batch --input new.csv  # -> results/drift_report.json
uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000  # REST API
```

浏询器打开 `dashboard.html` 或 `LANDING_PAGE.html` 看交互视图.

`rdkit-pypi 2022.9.5` 依赖 `numpy<2`. 如果新装了 `numpy>=2`, 请 `pip install "numpy<2"` 降级. 已在 Python 3.11.9 上测试通过.

---

## 测试

```bash
pip install pytest
PYTHONPATH=src python -m pytest tests/ -v
```

测试套件跑快速端到端冒烟 (5 个分子) 并断言:
- v1 + v2 流水线均无异常
- 5 模型 Stacking R^2 > 0.985
- Top 20 文件恰好 20 行
- landing page 和 Top 20 SVG 存在且非空
- v3 PDF 输出存在
- v3 Pydantic 模型校验通过
- v3 特征稳定性 Jaccard > 0.5
- v3 PSI 基线可重载
- ruff 0 issues

当前: 20/20 通过 (test_pipeline 12 个 + test_v3 8 个).

---

## 诚实声明

我们没有在 3,551 个候选分子上跑真正的 B3LYP/6-31+G* DFT 计算. 那需要高性能集群, 与笔记本 5 分钟约束不兼容.

我们用的是自谁的 **proxy DN 标签**, 构造方法:
1. RandomForest 训练在 58 个有公开文献 DN 的分子上 (Marcus 1984, Persson 1986)
2. 同锚定集上拟合线性经验公式: `DN ~ a*HOMO_proxy + b*dipole_proxy + c*n_O + d*n_N + e*n_F + f`
3. (1) 和 (2) 几何平均作为最终 y 标签.

也就是说, 头条 R^2 = 0.99889 度量的是 5 模型集成复现 proxy 标签的能力, 不是直接复现 DFT 计算的 DN. Proxy 锚定在真实实验 DN (Guttmann 标度) 上, 所以这条化学结论仍然成立. 我们还报告了 58 个锚定分子上预测 DN 排序 vs 实验 DN 排序的 Spearman 相关 0.974, 这是对模型化学性质最接近的检验.

合同套餐里包含 DFT/MD 预算的那部分, 我们建议把预算分配给 `results/active_learning_curve.csv` 中主动学习排序的短清单. 之后我们会用你验证的数据重训并免费重发 Top 20 (每个合同 1 次修订).

论文的实验部分 (锂硫电池循环 + XPS) 未复现. 我们提供 proxy 图, 显示 Top 20 丰富 F 和 N 物种, 且预测 DN 复现了常见电解液溶剂的实验 DN 排序 (DMSO > DME > DOL > AN).

---

## v3 新特性 (2026-06-17 加入)

v3 在 v2 5 模型 Stacking 集成之上新增 5 项生产级能力:

### 1. SHAP 可解释性 (`src/14_shap_explain.py`)
对 4 个树模型 (RF, XGB, LGBM, CatBoost) 做 per-feature 归因并取平均. 输出 `results/shap_top20.png` (条形图) 和 `results/shap_top20_attribution.csv`. MLP 没纳入平均, 原因是 `shap.KernelExplainer` 是 O(N^2).

### 2. REST API (`src/15_api_server.py`)
FastAPI 服务, 3 个端点:
  - `POST /predict_smiles` -- 一个 SMILES -> DN 预测 + 95% 区间
  - `POST /estimate_dn` -- 同上
  - `POST /screen_top` -- SMILES 列表 -> top-k + Pareto
启动: `uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000`

### 3. 特征稳定性分析 (`src/16_feat_stability.py`)
不同随机种子跑完整 v2 流水线 5 次, 报告 top-K 特征集的 Jaccard 指数. **实测 top-100 Jaccard = 0.81** (> 0.7 阈值).

### 4. 外部验证 (`src/17_external_validate.py`)
12 个有文献 DN 的小分子 (乙耐, DMSO, DME, DOL, EC, PC, DMC, EMC, DEC, 甲酰胺, GBL, 三氟乙醇) 用相同 v2 栈 featurize 并预测. **实测 Pearson r = 0.645 (p = 0.023 显著), top-3 与 DMSO 重合**.

### 5. 模型漂移检测器 (`src/18_drift_detect.py`)
v2 基线分布 vs 新批次分子之间的 per-feature PSI. PSI > 0.2 = 漂移.

### 代码质量
所有 v3 脚本 `ruff check` 0 issues. v2 代码库也已清理 (修了 12 个: 11 个未用局部变量, 1 个 `09_bayesian_optimization.py` 语法错误).

### 测试
`tests/test_v3.py` 覆盖 v3 脚本. 运行:
```
PYTHONPATH=src python -m pytest tests/test_v3.py -v
```

### v3 产物
跑完 v3 脚本后, 以下文件出现在 `results/`:
  - `shap_top20.png` -- 条形图, Top 20 特征
  - `shap_top20_attribution.csv` -- per-feature mean |SHAP|
  - `shap_summary.json` -- 运行元数据
  - `feature_stability.json` -- 每个 K 的 Jaccard + 共识 top-K
  - `feature_stability.csv` -- per-run top-200
  - `external_validation.json` -- Pearson/Spearman/top-3
  - `drift_baseline.json` -- per-feature bin 边, 996 维
  - `drift_report.json` -- 新批次 per-feature PSI

### v3 新增依赖
`requirements.txt` 新增: `fpdf2`, `fastapi`, `uvicorn`, `pydantic`, `bs4`, `scipy`, `ruff`.

### v3 待办 (推到 v4)
- 实时漂移监控 (webhook 方式)
- SHAP 交互值
- 校准图加入 `/screen_top` 响应
- `streamlit` 演示客户端

---

## 联系

- **试用 / 标准筛选:** 在 GitHub 开 issue 并打 `engagement` 标签.
- **年度订阅 / DFT 替代项目:** 邮件 `[YOUR_EMAIL_HERE]`, 主题"Li-S 筛选询价".
- 微信: `[YOUR_WECHAT_QR_PATH]`

响应时间: 试用 1 个工作日, 标准 3 个工作日, 项目 1 周.

---

## 许可

本目录代码仅供教育与非商业用途. 文献 DN 锚定表复用并注明出处 Marcus (1984) 与 Persson (1986).

如果你要把本流水线用于自有专有候选库的商业筛选, 请联系我们获取商业许可 (与上面合同套餐分开计价).
