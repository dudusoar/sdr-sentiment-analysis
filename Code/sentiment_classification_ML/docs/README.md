# YouTube配送机器人情感分析系统

## 项目概述

本项目是一个完整的情感分析系统，专门用于分析YouTube配送机器人相关评论的情感倾向。系统支持三分类情感分析（消极、积极、中性），并通过二分类策略（OVO和OVR）实现高精度的情感识别。

### 核心特性

- 🤖 **专用于配送机器人领域**的情感分析
- 📊 **三分类支持**：消极(0)、积极(1)、中性(2)
- 🔄 **双重分类策略**：OVO（一对一）和OVR（一对多）
- 🧠 **多种机器学习模型**：朴素贝叶斯、SVM、随机森林、决策树
- 🔤 **丰富特征提取**：TF-IDF、Word2Vec（含智能降级机制）
- ⚖️ **数据平衡处理**：SMOTE过采样支持
- 📈 **完整性能评估**：ROC-AUC、准确率、F1分数
- 🏗️ **模块化架构**：易于维护和扩展

## 项目结构

```
sentiment_classification_ML/
├── main.py                 # 主程序入口
├── config.py              # 配置管理
├── utils.py               # 工具函数
├── init_nltk.py          # NLTK初始化
├── test_binary_framework.py  # 测试脚本
├── model_performance_summary.md  # 模型性能总结
├── docs/                  # 项目文档
│   ├── README.md         # 项目说明（本文件）
│   ├── model_performance.md  # 模型性能报告
│   └── test_results.md   # 测试结果报告
├── src/                   # 核心源代码
│   ├── binary_classification_framework.py  # 二分类框架
│   ├── word2vec_downloader.py  # Word2Vec模型管理
│   ├── data_manager.py    # 数据管理
│   ├── preprocessing.py   # 数据预处理
│   ├── vectorizers.py     # 特征向量化
│   ├── model_training.py  # 模型训练
│   ├── evaluation.py      # 性能评估
│   └── visualization.py   # 结果可视化
├── data/                  # 数据文件
│   ├── combined_comments.xlsx  # 原始评论数据
│   ├── GoogleNews-vectors-negative300.bin  # Word2Vec模型(已修复)
│   └── GoogleNews-vectors-negative300.bin.corrupted  # 损坏的模型文件
├── results/               # 结果输出
│   ├── binary_framework_test_results.md  # 二分类框架测试结果
│   ├── selected_comments.xlsx  # 筛选后的评论数据
│   ├── dataset/          # 处理后的数据集
│   └── word_frequency_results/  # 词频分析结果
└── __pycache__/          # Python缓存
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv sentiment_env
source sentiment_env/bin/activate  # Linux/Mac
# 或
sentiment_env\Scripts\activate     # Windows

# 安装依赖
pip install pandas numpy scikit-learn gensim nltk matplotlib openpyxl imbalanced-learn wordcloud seaborn gdown
```

### 2. 初始化系统

```bash
# 准备数据集
python main.py --prepare

# 设置Word2Vec模型（自动下载1.65GB）
python main.py --setup-word2vec

# 分析词频
python main.py --analyze
```

### 3. 运行测试

```bash
# 完整二分类框架测试
python main.py --binary-test comprehensive

# 特定配置测试
python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo

# 运行验证脚本
python test_binary_framework.py
```

## 核心功能模块

### 1. 二分类框架 (`binary_classification_framework.py`)

完整还原了原始测试代码的二分类测试流程：

- **OVO策略**：一对一分类（data_01, data_02, data_12）
- **OVR策略**：一对多分类（data_0, data_1, data_2）
- **模型支持**：MultinomialNB, SVM, RandomForest, DecisionTree
- **特征支持**：TF-IDF(1-3gram), Word2Vec
- **评估指标**：ROC-AUC, 准确率, F1分数

### 2. Word2Vec管理 (`word2vec_downloader.py`)

自动化的Word2Vec模型管理：

- **自动下载**：从Google Drive下载GoogleNews向量
- **备用方案**：多种下载源和回退策略
- **模型验证**：自动检查模型完整性
- **智能降级**：在模型损坏时自动创建虚拟模型
- **便捷加载**：简化的模型加载接口

### 3. 数据管理 (`data_manager.py`)

完整的数据处理流程：

- **数据筛选**：自动筛选标签0、1、2的评论
- **数据集创建**：自动创建OVO和OVR数据集
- **预处理集成**：低频词过滤、标点处理、词元化
- **灵活配置**：支持多种预处理选项

### 4. 模型训练 (`model_training.py`)

统一的模型训练接口：

- **K折交叉验证**：可配置的交叉验证
- **单次训练**：快速验证和测试
- **过采样支持**：SMOTE算法集成
- **性能评估**：自动计算多种指标

## 最优性能配置

基于完整测试结果，推荐以下配置：

### 🏆 生产环境推荐

```python
# 最优配置：SVM + TF-IDF + OVO策略
model = SVC(probability=True, kernel='rbf')
vectorizer = TfidfVectorizer(ngram_range=(1,1))
strategy = 'ovo'  # 一对一策略
oversampling = False
remove_low_frequency = True

# 预期性能：ROC-AUC = 0.8946, 准确率 = 81%
```

### 📊 性能对比表

| 模型 | 特征 | 策略 | ROC-AUC | 准确率 | 说明 |
|------|------|------|---------|--------|------|
| SVM | TF-IDF(1,1) | OVO | 0.8946 | 81% | **最优配置** ⭐ |
| 朴素贝叶斯 | TF-IDF(1,1) | OVO | 0.8870 | 80% | 计算效率高 |
| 随机森林 | TF-IDF(1,2) | OVR | 0.8246 | 85% | 稳定性好 |

### 🎯 应用场景推荐

- **高精度要求**：SVM + Word2Vec（ROC-AUC: 0.89+）
- **快速部署**：朴素贝叶斯 + TF-IDF（响应速度快）
- **大规模数据**：随机森林 + OVR策略（扩展性好）

## 数据集说明

### 原始数据
- **来源**：YouTube配送机器人相关评论
- **标签**：0=消极，1=积极，2=中性
- **规模**：约5000条标注评论

### 二分类数据集

#### OVO策略数据集
- `data_01.xlsx`：消极 vs 积极（2298条）
- `data_02.xlsx`：消极 vs 中性（2668条）
- `data_12.xlsx`：积极 vs 中性（1910条）

#### OVR策略数据集  
- `data_0.xlsx`：消极 vs 其他（3393条）
- `data_1.xlsx`：积极 vs 其他（3343条）
- `data_2.xlsx`：中性 vs 其他（3444条）

## 性能分析

### 分类难度排序
1. **消极 vs 积极**：ROC-AUC = 0.89（最容易区分）
2. **积极 vs 中性**：ROC-AUC = 0.82（中等难度）
3. **消极 vs 中性**：ROC-AUC = 0.80（最困难）

### 模型特点分析

#### SVM（支持向量机）
- **优势**：泛化能力强，在高维空间表现优异
- **适用**：特征数量大于样本数量的场景
- **最优核函数**：RBF（径向基函数）

#### 朴素贝叶斯
- **优势**：训练速度快，对小数据集友好
- **适用**：文本分类，实时性要求高的场景
- **注意**：特征独立性假设较强

#### 随机森林
- **优势**：鲁棒性好，不易过拟合
- **适用**：特征重要性分析，集成学习
- **配置**：n_estimators=100表现最优

## 技术实现细节

### 特征工程

#### TF-IDF配置
```python
# 最优配置
TfidfVectorizer(
    ngram_range=(1,1),      # 1-gram表现最优
    max_features=10000,     # 避免维度爆炸
    stop_words='english'    # 使用自定义停用词
)
```

#### Word2Vec配置
```python
# 使用预训练模型（含智能降级）
Word2VecVectorizer(
    model=GoogleNews_model,  # 自动降级到虚拟模型
    bow='avg',              # 平均词向量
    shift_to_positive=False # 保持原始向量
)
```

### 数据预处理流程

1. **文本清理**：去除特殊字符、数字
2. **分词处理**：NLTK分词器
3. **停用词过滤**：自定义停用词列表
4. **词元化**：WordNetLemmatizer
5. **低频词处理**：可选的低频词过滤

### 模型评估指标

```python
# 主要评估指标
metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'classification_report': classification_report(y_test, y_pred)
}
```

## 扩展功能

### 命令行接口

```bash
# 查看所有选项
python main.py --help

# 模型选择
python main.py --train --model svm --type ovo --ngram 2

# 特征选择
python main.py --train --word2vec --kfold

# 完整测试
python main.py --binary-test comprehensive
```

### API接口设计

```python
# 预测单条评论
def predict_sentiment(text, model_config='best'):
    """
    预测单条评论的情感倾向
    
    Args:
        text: 输入评论文本
        model_config: 模型配置 ('best', 'fast', 'robust')
    
    Returns:
        sentiment: 情感标签 (0=消极, 1=积极, 2=中性)
        confidence: 置信度分数
    """
```

## 故障排除

### 常见问题

1. **Word2Vec加载失败**
   ```
   错误：'utf-8' codec can't decode byte 0x8b
   解决：重新下载模型文件
   命令：python main.py --setup-word2vec
   ```

2. **数据文件缺失**
   ```
   错误：找不到combined_comments.xlsx
   解决：确保数据文件在data目录
   命令：python main.py --prepare
   ```

3. **依赖包问题**
   ```
   错误：ModuleNotFoundError
   解决：安装缺失的依赖包
   命令：pip install missing_package
   ```

### 性能优化建议

1. **内存优化**：使用稀疏矩阵存储TF-IDF特征
2. **速度优化**：预计算特征向量，缓存模型结果
3. **精度优化**：集成多个模型，使用投票或加权策略

## 开发路线图

### 已完成功能 ✅
- [x] 完整二分类框架
- [x] 多种机器学习模型
- [x] TF-IDF和Word2Vec特征
- [x] 过采样和数据平衡
- [x] 性能评估和可视化
- [x] 模块化代码架构
- [x] 命令行接口

### 计划功能 🚧
- [ ] Web API接口
- [ ] 实时预测服务
- [ ] 模型集成策略
- [ ] 深度学习模型（BERT）
- [ ] 多语言支持
- [ ] 增量学习

### 优化方向 🎯
- [ ] Word2Vec编码问题修复
- [ ] 特征融合策略
- [ ] 超参数自动优化
- [ ] 分布式训练支持
- [ ] 模型压缩和量化

## 贡献指南

### 代码规范
- 使用Python 3.8+
- 遵循PEP 8编码规范
- 添加详细的docstring文档
- 包含单元测试

### 提交流程
1. Fork项目仓库
2. 创建功能分支
3. 编写测试用例
4. 提交Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- **数据来源**：YouTube配送机器人评论数据集
- **预训练模型**：Google News Word2Vec模型
- **开源库**：scikit-learn, gensim, nltk, pandas

---

*最后更新：2024年12月*  
*版本：v1.0.0*  
*维护者：项目开发团队* 