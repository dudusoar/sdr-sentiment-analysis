# YouTube-SC 虚拟环境使用指南

## 环境概览
- **Python版本**: 3.11.12
- **虚拟环境路径**: `.venv/`
- **依赖管理工具**: uv
- **安装包数量**: 138个
- **创建时间**: 2026-01-06

## 激活虚拟环境

### Windows (PowerShell/CMD)
```cmd
.venv\Scripts\activate
```

### Windows (Git Bash)
```bash
source .venv/Scripts/activate
```

### Linux/Mac
```bash
source .venv/bin/activate
```

## 验证环境
激活环境后，运行以下命令验证：
```bash
python -c "import pandas; print(f'pandas版本: {pandas.__version__}')"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import sklearn; print(f'scikit-learn版本: {sklearn.__version__}')"
```

## 运行项目模块

### 1. 聚类分析
```bash
cd Code/sdr_clustering_analysis
python main.py
```

### 2. 机器学习情感分类
```bash
cd Code/sentiment_classification_ML
python main.py --help
```

### 3. BERT情感分类
```bash
cd Code/sentiment_classification_Bert/code
python main.py --help
```

### 4. 主题建模
```bash
cd Code/topic_modeling
python topic_modeling_analysis.py
```

## 管理依赖

### 查看已安装包
```bash
uv pip list
```

### 更新包
```bash
uv update
```

### 添加新包
```bash
uv add 包名
```

### 从requirements.txt安装
```bash
uv pip install -r .claude/requirements.txt
```

## 关键包版本
- pandas: 2.3.3
- numpy: 2.4.0
- scikit-learn: 1.8.0
- torch: 2.9.1+cpu
- transformers: 4.57.3
- gensim: 4.4.0
- nltk: 3.9.2
- matplotlib: 3.10.8
- seaborn: 0.13.2
- wordcloud: 1.9.5

## 注意事项
1. 首次使用需要激活虚拟环境
2. 所有项目代码应在激活环境后运行
3. 如需重新创建环境，可删除`.venv`目录后运行`.claude/setup-environment.bat`
4. 如有编码问题（GBK错误），可在Python脚本开头添加：
   ```python
   import sys
   import io
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
   ```

## 故障排除
1. **激活失败**: 检查`.venv`目录是否存在
2. **导入错误**: 确保虚拟环境已激活
3. **包缺失**: 重新运行`uv pip install -r .claude/requirements.txt`
4. **编码问题**: 使用UTF-8编码保存文件，或在脚本中设置编码

## 开发建议
1. 在虚拟环境中使用Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. 安装开发工具:
   ```bash
   uv add --dev black flake8 pytest
   ```
3. 定期更新包:
   ```bash
   uv update --outdated
   uv update
   ```

---
*使用uv虚拟环境管理，创建于2026-01-06*