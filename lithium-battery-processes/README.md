# Lithium-Battery Process Documentation Mirror

Bilingual (EN / ZH) mirror of public lithium-battery **manufacturing process**
documentation. The companion repo `donor-screener-pbp/` covers materials
physics; this subdir covers the production line — electrode manufacturing,
cell assembly, formation, and aging.

## What is here

| Path | Contents |
|---|---|
| `manifest.json` | Hand-curated source list (URL, license, sha256, local mirror path) |
| `index.csv` | Flat bilingual table — one row per source, 13 columns |
| `sources/<category>/*.html` or `*.pdf` | The actual mirrored files |
| `data/process_steps.yaml` | 14-step cell-manufacturing process map (CATL-aligned) |
| `data/parameter_ranges.csv` | Bilingual process parameter ranges with citations |
| `src/p36_fetch_process_docs.py` | Offline-safe fetcher (urllib, sha256 verified) |
| `src/utils_lb.py` | Small helper module (shared with the fetcher) |
| `tests/test_fetch_process_docs.py` | Unit tests for fetcher, manifest, sha256 |

## Categories

- **process_summary**: high-level process overviews (Volta Foundation, BatteryDesign.net)
- **electrode_manufacturing**: slurry / coating / drying / calendering papers
- **cell_assembly**: stacking / winding / packaging reviews
- **formation**: SEI formation cycling protocols
- **aging_calendar**: aging and capacity-grading references

## 锂离子电池工艺资料镜像

本目录为公开的锂电池**生产工艺**文档的中英双语镜像。配套仓库
`donor-screener-pbp/` 覆盖材料物理，本目录覆盖产线工艺：极片制造、
电芯组装、化成分容与老化。

| 路径 | 内容 |
|---|---|
| `manifest.json` | 手工整理的来源清单（URL、协议、sha256、本地镜像路径） |
| `index.csv` | 中英双语平表，每行一个来源，13 列 |
| `sources/<分类>/*.html` 或 `*.pdf` | 实际镜像文件 |
| `data/process_steps.yaml` | 14 步电芯制造工艺地图（宁德产线对标） |
| `data/parameter_ranges.csv` | 中英双语工艺参数区间与参考文献 |
| `src/p36_fetch_process_docs.py` | 离线安全抓取器（urllib，sha256 校验） |
| `src/utils_lb.py` | 抓取器共享的小工具模块 |
| `tests/test_fetch_process_docs.py` | 抓取器、清单、sha256 单元测试 |

## 分类

- **process_summary 工艺综述**：高层工艺概览（Volta Foundation, BatteryDesign.net）
- **electrode_manufacturing 极片制造**：浆料 / 涂布 / 干燥 / 辊压论文
- **cell_assembly 电芯组装**：叠片 / 卷绕 / 封装综述
- **formation 化成**：SEI 化成循环工艺
- **aging_calendar 老化分容**：老化与容量分选参考

## Usage

### Run the fetcher (online, populates sources/)

```bash
python src/p36_fetch_process_docs.py
```

### Verify integrity (offline, default in CI)

```bash
python src/p36_fetch_process_docs.py --offline
```

### Run the tests

```bash
python -m pytest tests/test_fetch_process_docs.py -v
```

### Adding a new source

1. Add an entry to `manifest.json` (all 8 required keys).
2. Run `python src/p36_fetch_process_docs.py` once to download.
3. Commit the new file under `sources/` plus the updated `manifest.json`
   and `index.csv`.

## Citations

- Volta Foundation, *Battery Manufacturing Basics from CATL's Cell Production Line (Part 1 & 2)*, 2024
- BatteryDesign.net, *Viscosity / Cell Assembly / Formation & Aging*, 2024
- B. Westphal et al., *Understanding the effect of coating and drying on Li-ion electrode performance*, University of Warwick, 2021
- Reynolds et al., *Impact of Formulation and Slurry Properties on Li-ion Electrode Manufacturing*, Batteries & Supercaps, 2024
- Liu et al., *Review of electrode manufacturing in Li-ion batteries*, OSTI, 2021
- S. J. An et al., *Fast formation cycling for lithium ion batteries*, J. Power Sources, 2017
- Bar-Tal et al., *Modeling Battery Formation*, J. Electrochem. Soc., 2023
- Dose et al., *Influence of Formation Current Density on Transport Properties of Model-Type SEIs*, Batteries & Supercaps, 2019
- Reichel et al., *Effectiveness of Formation Protocols for Aqueous Processed NMC622||Gr Full Cells*, J. Electrochem. Soc., 2024

## License notes

- Open Access PDFs (Westphal 2021, OSTI 2021, etc.) are mirrored under their
  original CC-BY / public-domain terms.
- Web articles (Volta Foundation, BatteryDesign.net) are mirrored as HTML for
  fair-use reference; please cite the original URL.
- All numbers in `data/parameter_ranges.csv` are summaries of the cited
  papers; please cite the originals for any production use.
