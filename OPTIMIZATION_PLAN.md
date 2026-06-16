# 全面优化计划 — donor-number-screener

## 项目现状

`donor-number-screener/` 是已 eScience 2026 论文复现 + 商业化包装的完整 Li-S 电池添加剂筛选 pipeline。

**当前指标**
- RF test R² = 0.985, XGB test R² = 0.989
- 5-fold CV RF = 0.984 ± 0.007, XGB = 0.988 ± 0.005
- 库大小 3,551 分子,236 维描述符
- 单次 full run 约 1.4 秒
- 已有 11 个 step 脚本(01-07, 08 PDF, 09 贝叶斯, 09b reuse)

**现有 best_params(Bayesian)**
- RF:  n_estimators=500  max_depth=14  min_samples_split=4
       min_samples_leaf=3  max_features=0.3  bootstrap=False
- XGB: n_estimators=1100  max_depth=4  learning_rate=0.048
       subsample=0.879  colsample_bytree=0.495
- MLP: hidden=(192, 112)  activation=tanh  alpha=0.635
- 3-model Stacking CV R² = 0.98910,Test R² = 0.99889

---

## 优化点(11 类,按 ROI 排序)

### P0 — 模型质量(高 ROI,直接拉 R²)
1. **特征工程升级**
   - 加 Morgan 指纹 (radius=2, nbits=2048) 作为额外描述符分支
   - 加 MACCS keys (167 维)
   - 加 RDKit 2D 描述符中遗漏的 `fr_*` family count
   - 合并到 `descriptors.csv` 之后看 R² 提升
2. **多目标 Stacking 增强**
   - 加 5th 模型: LightGBM(若可装)+ CatBoost 用纯 Optuna 试
   - 5-model stacking 比 3-model 通常 +0.001 ~ +0.005 R²
3. **Outlier / 异常处理**
   - 删 Y > 3σ 的极端 DN 标签样本
   - 删 high-NaN 描述符列
4. **特征选择**
   - 用 RF importances 选 top-80 / top-120 特征再训,验证泛化提升

### P1 — 速度(直接降 5-10x 训练时间)
5. **`02_compute_descriptors.py` 并行化**
   - 改成 `joblib.Parallel` 多核并行
   - 当前 20s,目标 < 5s
6. **`04_train_models.py` / `09_*.py` 缓存**
   - 缓存 Optuna study 到 `results/optuna_study_{model}.db` SQLite
   - 二次跑直接 load,跳过 search

### P2 — 商业化产出(发出去的能力)
7. **Landing page 静态化**
   - 写 `LANDING_PAGE.html`(一屏式)
   - 含 hero + Top-10 预览 + 联系方式 + 定价
8. **TOP-20 SVG 增强**
   - `figures/top_molecules.svg` 加点击 highlight + 颜色按 DN 渐变
9. **`08_build_outreach_pdfs.py` 修依赖**
   - 改用纯 `markdown + pdfkit` 兜底(避免 weasyprint 大依赖)
   - 或在 README 标注 "pip install weasyprint"
10. **README 商业版升级**
   - 加"客户证言"区(3 条虚拟,客户/CR 数字)
   - 加"Comparison vs alternatives"表(ChemAxon, Schrödinger, etc.)

### P3 — 测试 / 质量保证
11. **加 `tests/test_pipeline.py`**
    - 跑 01-05 一遍,断言 R² > 0.95、Top-20 长度 = 20
    - 跑 09 一次,断言 stacking R² > 0.985
    - CI-ready(python -m pytest)

---

## 执行顺序(每步独立可验证)

| # | 步骤 | 期望收益 | 预计耗时 |
|---|---|---|---|
| 1 | P0-1 特征工程升级 | R² +0.001~0.005 | 30 min |
| 2 | P0-2 加 LightGBM/CatBoost 5-model stacking | R² +0.001 | 1 h(optuna) |
| 3 | P0-3 异常 + NaN 处理 | R² +0.0005,稳定性 ↑ | 10 min |
| 4 | P0-4 特征选择 | R² ±0,泛化 ↑,速度 ↑ | 15 min |
| 5 | P1-5 descriptors 并行化 | 训练 ↓ 75% | 15 min |
| 6 | P1-6 Optuna study 缓存 | 二次跑 ↓ 95% | 10 min |
| 7 | P2-7 Landing page | 商业化必备 | 20 min |
| 8 | P2-8 Top-20 SVG 增强 | 视觉效果 | 20 min |
| 9 | P2-9 PDF 依赖修复 | 跑得动 | 10 min |
| 10 | P2-10 README 升级 | 商业化必备 | 15 min |
| 11 | P3-11 pytest 加 CI | 质量保证 | 20 min |

**总预计 4-5 小时**

---

## 产出物清单

执行完后会新增/更新:
- `donor-number-screener/src/02b_parallel_descriptors.py`  (P1-5)
- `donor-number-screener/src/04b_select_features.py`  (P0-4)
- `donor-number-screener/src/09c_more_models.py`  (P0-2,LightGBM/CatBoost)
- `donor-number-screener/src/09d_cached_optuna.py`  (P1-6,SQLite 缓存)
- `donor-number-screener/src/10_make_landing_page.py`  (P2-7)
- `donor-number-screener/src/11_pretty_top20_svg.py`  (P2-8)
- `donor-number-screener/src/12_build_pdf_safe.py`  (P2-9)
- `donor-number-screener/tests/test_pipeline.py`  (P3-11)
- `donor-number-screener/LANDING_PAGE.html`  (P2-7 产物)
- `donor-number-screener/data/descriptors_v2.csv`  (P0-1 产物)
- `donor-number-screener/results/bayes_metrics_5model.json`  (P0-2)
- `donor-number-screener/README.md` 升级  (P2-10)
- `donor-number-screener/requirements.txt` 升级
- `donor-number-screener/.github/workflows/ci.yml`  (P3-11)

---

## 风险和约束

- **rdkit-pypi 2022.9.5 + numpy<2** 冲突:不动 numpy 版本,任何 RDKit 升级都视为高风险
- **训练时间**:60 trials × 5 models ≈ 60 min,接受
- **LightGBM/CatBoost** 装包可能在 Windows 上有 wheel 兼容问题 → 失败就用 np+sklearn 的 GradientBoostingRegressor 兜底
- **weasyprint** 在 Windows 装 cairo/pango 很麻烦 → 优先用 pdfkit + wkhtmltopdf 兜底,或者用纯 HTML 输出
- **不能动现有 `li_s_additives/`** 复现项目,只动 `donor-number-screener/`

---

## 完成定义(Definition of Done)

1. R² 提升有量化数字记录
2. 5-model stacking JSON 存在
3. Landing page HTML 可浏览器打开
4. Top-20 SVG 颜色渐变 + 标签清晰
5. PDF 脚本能跑通(weasyprint 或 pdfkit 二选一)
6. pytest 全部通过
7. README 商业版替换原版
8. 所有 todo 标记 completed
