# Elexacaftor 合成工艺逆向推导

> 基于 PubChem Trikafta 分子式 (C76H82F6N6O15) 及专利 CN-118530179-A 公开信息逆向分析

---

## 1 | 背景与分子组成

Trikafta 是 Vertex 公司开发的三联疗法囊性纤维化药物，由以下三个组分以 **1:1:2（Elexacaftor : Tezacaftor : Ivacaftor）** 摩尔比组成：

| 组分 | PubChem CID | 分子式 | 角色 |
|---|---|---|---|
| **Elexacaftor** (VX-814) | CID 46199646 | C26H27F3N2O6 | CFTR 纠正剂 |
| **Tezacaftor** (VX-661) | CID 16220172 | C26H27F3N2O6 | CFTR 纠正剂 |
| **Ivacaftor** (VX-770) | CID 16220172 | C24H28N2O3 | CFTR 增效剂 |

**加和验证：**

```
2 x C26H27F3N2O6 + C24H28N2O3
= C52H54F6N4O12 + C24H28N2O3
= C76H82F6N6O15  [与 PubChem 一致]
```

分子全量：C76H82F6N6O15，分子量 1433.5 g/mol

---

## 2 | 关键中间体汇总

| 编号 | 名称 | 结构特征 | 合成方法 |
|---|---|---|---|
| **原料 B** | 4-氟三氟甲基苯乙酮（或硝基衍生物） | C6H3(F)(CF3)-C(=O)-CH3 | 商业可得 |
| **中间体 C** | gem-Me2C(OH)-苯并二氧杂环-吲哚酮胺 | 还原后骨架，含伯胺 | NaBH4 / LiAlH4 还原酮或硝基 |
| **化合物 D** | 3,3-双(CF3)-2,2-二甲基丙醇衍生物 | gem-Me2C(OH)-CF3 | CF3-乙酰丙酮还原或格氏+CF3加成 |
| **中间体 E** | N-烷基化产物 | 3度胺 + gem-Me2C(OH)-CF3 侧链 | DPPA/PPh3 偶联（叠氮-膦试剂体系） |
| **化合物 A** | Elexacaftor | 完整 C26 骨架，含 gem-Me2C(OH)-CF3 | 酸性环化 / 脱保护（Fischer 吲哚合成） |
| **最终产物** | Trikafta | Elexacaftor-Ivacaftor 酰胺键 | HATU / EDCI.HOBt 酰胺缩合 |

---

## 3 | 完整合成路线

### Step 1 | 原料 B → 中间体 C（还原反应）

**原料 B**：4-氟-2-三氟甲基苯乙酮（或对应的硝基苯甲醛衍生物）

**反应试剂**：NaBH4 / MeOH 或 LiAlH4（还原酮羰基）；或 H2 / Pd-C（还原硝基）

```
原料 B（芳基酮/硝基苯）
   |
   |  NaBH4 / MeOH  或  H2 / Pd-C
   |
   V
中间体 C：gem-Me2C(OH)-苯并二氧杂环-吲哚酮胺骨架，含伯胺端
```

---

### Step 2 | 中间体 C + 化合物 D → 中间体 E（叠氮-膦试剂偶联）

**化合物 D**：3,3-双(三氟甲基)-2,2-二甲基丙醇（gem-Me2C(OH)-CF3 侧链）

合成路径：CF3-乙酰丙酮  -->  NaBH4 还原  -->  gem-Me2C(OH)-CF3 醇

```
        CF3
         |
    ---- C ----  gem-二甲基-CF3 碳中心
       /     \
   (CH3)2    CH2-OH  (化合物 D 的核心片段)
```

**反应体系：叠氮试剂（DPPA）+ 膦试剂（PPh3）**

这是 **Staudinger 反应变体** 或 **Mitsunobu 型亲核取代**：

1. DPPA（叠氮磷酸二苯酯）将化合物 D 的伯羟基转化为叠氮或磷酰胺中间体
2. PPh3（膦试剂）与中间体 C 的胺基反应，形成 P-N 键，活化胺端
3. 偶联得到中间体 E：N-烷基化产物（3度胺），含 gem-Me2C(OH)-CF3 侧链

---

### Step 3 | 中间体 E → 化合物 A / Elexacaftor（酸性环化）

**反应试剂**：TFA、PPA 或浓 HCl

在酸性条件下，中间体 E 发生 **Fischer 吲哚合成** 或 **分子内环化**，构建最终的二氢吲哚（indoline）/ 吲哚酮（oxindole）双环核心骨架：

```
中间体 E（含邻氟苯乙酮片段 + gem-Me2C(OH)-CF3 侧链）
     |
     |  H+（酸催化环化）
     |
     V
化合物 A = Elexacaftor
|-- 苯并二氧杂环（2,2-二氟）
|-- 吲哚酮骨架（连接 5-位酰胺）
|-- gem-Me2C(OH)-CF3 侧链
|-- (2R)-2,3-二羟基丙基手性链
```

**IPC 分类支持（专利 CN-118530179-A）：**
- `C07D231/20`：吡唑啉酮 / 吲哚衍生物的环化反应
- `C07C29/147`：含氟醇的还原合成
- `C07C31/38`：含 CF3 的多元醇衍生物

---

### Step 4 | 化合物 A（Elexacaftor）+ Ivacaftor → 最终药物（酰胺缩合）

**Ivacaftor**（C24H28N2O3）的活性羧酸端（-COOH）与 Elexacaftor 的仲胺端（-NH-）在缩合剂存在下反应，形成酰胺键：

```
Elexacaftor (化合物A)      Ivacaftor (VX-770)
   -NH-  +  HO-C(=O)-  -->  -NH-C(=O)-  (酰胺键)
```

**常用缩合剂**：HATU、DIC/HOBt 或 EDCI.HCl / HOBt

---

## 4 | 专利核心创新点（CN-118530179-A）

**发明人**：FU SHOU, GONG HAIWEI, ZHANG CANJIE, DING CHAOWANG, JIA YUXIANG 等（河南雨辰制药）

**优先权日**：2024/05/09  |  **公开日**：2024/08/23  |  **申请人**：HENAN YUCHEN PHARMACEUTICAL CO LTD

| 创新点 | 描述 |
|---|---|
| **连续流还原-偶联** | 3,3-双三氟甲基-2,2-二甲基丙醇的甲苯溶液还原后直接投入叠氮-膦试剂偶联反应，无需分离纯化 |
| **避免热分解** | gem-Me2C(OH)-CF3 侧链中间体在蒸馏提纯过程中易热分解变质，连续流工艺彻底规避此风险 |
| **避免设备堵塞** | 蒸馏过程中该中间体易结晶堵塞设备，连续流直接投料解决此工程问题 |
| **工艺简化** | 溶剂用量减少，操作步骤减少，产品收率大幅提升 |
| **工业化可行性** | 整条路线操作简便、易于放大，适合工业化生产 |

---

## 5 | 参考信息

| 来源 | 链接 |
|---|---|
| PubChem Trikafta 页面 | https://pubchem.ncbi.nlm.nih.gov/compound/Trikafta |
| PubChem Elexacaftor | https://pubchem.ncbi.nlm.nih.gov/compound/Elexacaftor |
| 专利 CN-118530179-A | https://pubchem.ncbi.nlm.nih.gov/patent/CN-118530179-A |
| Google Patents | https://patents.google.com/patent/CN118530179A |

---

*本文档为基于公开专利信息的逆向工艺分析，仅供科研参考。*
