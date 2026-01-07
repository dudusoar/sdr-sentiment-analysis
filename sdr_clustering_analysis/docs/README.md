# YouTube送货机器人评论聚类分析系统（sdr_clustering_analysis）

## 项目简介
本项目旨在对YouTube平台上与送货机器人相关的评论进行自动化聚类分析，结合情感标签，实现评论内容的结构化理解和可视化。系统采用SBERT文本嵌入与K-means聚类，支持多K值分析，并输出详细的聚类结果、关键词、情感分布和可追溯的索引。

## 主要功能
- SBERT文本嵌入与特征提取
- K-means聚类（支持多K值）
- 聚类结果与情感标签分布分析
- 关键词提取与聚类profile导出（Excel）
- 可视化（聚类分布、情感分布、轮廓系数等）
- 输出文件主键与原始数据index一致，便于溯源

## 目录结构
```
sdr_clustering_analysis/
├── main.py                # 主程序入口，聚类分析全流程
├── config.py              # 配置文件，数据与参数管理
├── framework.txt          # 项目框架与设计说明
├── data/                  # 原始与中间数据
│   └── combined_comments.xlsx  # 原始评论数据
├── src/                   # 核心代码模块
│   ├── data_loader.py     # 数据加载
│   ├── text_preprocessor.py # 文本预处理
│   ├── feature_extractor.py # 特征提取
│   ├── clustering.py      # 聚类算法
│   ├── evaluation.py      # 结果评估与可视化
│   └── utils.py           # 工具函数
├── results/               # 聚类分析输出
│   └── kX_results/        # 每个K值的聚类结果（含profile、可视化等）
└── docs/                  # 项目文档（本文件）
```

## 聚类分析流程
1. **数据加载与预处理**：加载原始评论，清洗文本，确保index唯一。
2. **特征提取**：SBERT生成文本嵌入。
3. **聚类分析**：对不同K值进行K-means聚类。
4. **结果评估与导出**：
   - 生成每个cluster的关键词、样本评论、情感分布
   - 导出profile为Excel（每个cluster一个文件，sheet1为keywords，sheet2为comments）
   - 生成聚类分配csv，主键为index
   - 输出多种可视化图片
5. **可追溯性**：所有输出文件均以原始数据index为主键，便于后续分析和溯源。

## 输出文件结构与一致性说明
- 每个K值下的`cluster_profiles`目录，包含每个cluster的Excel文件（如`cluster_0_profile.xlsx`）：
  - Sheet1: `keywords`，为该cluster的关键词（按TF-IDF排序）。
  - Sheet2: `comments`，为该cluster下的所有评论文本。
- `cluster_assignments_kX.csv`（如k3），主键为原始数据的`index`列，便于与原始数据一一对应和追溯。
- 所有聚类相关输出文件均以原始数据的`index`列为唯一主键，确保可追溯性和一致性。

## 维护与扩展建议
- 保持index列在原始数据中的唯一性，所有分析与输出均以index为主键。
- 新增聚类算法或分析模块时，务必保证输出文件结构与主键一致性。
- 文档与代码同步维护，便于团队协作和论文复现。
- 如需扩展到其他平台评论分析，可复用本项目的聚类与评估框架。 