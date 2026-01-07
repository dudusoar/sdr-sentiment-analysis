# coding: utf-8
'''
filename: visualization.py
function: 结果可视化，词频统计和词云图绘制
'''

from collections import Counter
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
import numpy as np
from config import WORD_FREQ_DIR
from utils import ensure_directory, save_excel

def word_count(docs, file_name, threshold=0):
    '''
    计算词频并保存结果
    
    Args:
        docs: 单词列表的列表
        file_name: 保存的文件名
        threshold: 词频阈值，只保留词频大于threshold的词
    
    Returns:
        word_frequency_df: 包含词频统计的DataFrame
    '''
    ensure_directory(WORD_FREQ_DIR)
    file_path = os.path.join(WORD_FREQ_DIR, file_name)
    
    # 统计词频
    words_list = [word for row in docs for word in row]
    word_counts = Counter(words_list)
    word_counts_dict = dict(word_counts)
    sorted_word_counts = sorted(word_counts_dict.items(), key=lambda x: x[1], reverse=True)

    # 转化为dataframe
    word_frequency_df = pd.DataFrame(sorted_word_counts, columns=['word', 'frequency'])
    
    # 筛选词频大于阈值的词
    index = word_frequency_df['frequency'] > threshold
    word_frequency_df = word_frequency_df[index]
    
    # 保存结果
    save_excel(word_frequency_df, file_path)
    print(f'词频统计已保存到: {file_path}')

    return word_frequency_df

def generate_word_cloud(words, picture_name):
    '''
    生成词云图并保存
    
    Args:
        words: 词频字典，键为词，值为频率
        picture_name: 保存的图片名称
    
    Returns:
        None
    '''
    ensure_directory(WORD_FREQ_DIR)
    picture_path = os.path.join(WORD_FREQ_DIR, picture_name)
    
    # 创建WordCloud对象
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    # 根据词频生成词云
    wordcloud.generate_from_frequencies(frequencies=words)
    # 保存词云
    wordcloud.to_file(picture_path)
    print(f'词云图已保存到: {picture_path}')

    # 使用matplotlib来显示词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(picture_name.replace('.png', ''))
    plt.show()

def analyze_word_frequency(comments, output_dir=WORD_FREQ_DIR):
    '''
    分析词频并生成词云图
    
    Args:
        comments: 包含评论数据的DataFrame
        output_dir: 输出目录
    
    Returns:
        None
    '''
    ensure_directory(output_dir)
    
    # 总的词频统计
    word_frequency_df = word_count(comments['f_word_list'], file_name='word_frequency.xlsx', threshold=0)
    
    # 保存词频大于1的词
    word_frequency_df_small = word_count(comments['f_word_list'], file_name='word_frequency_small.xlsx', threshold=1)
    
    # 按标签统计词频
    for label in [0, 1, 2]:
        index = comments['label1'] == label
        word_frequency_df_label = word_count(
            comments['f_word_list'][index], 
            file_name=f'word_frequency_{label}.xlsx', 
            threshold=0
        )
    
    # 生成词云图
    # 总词频对应的词云
    words = word_frequency_df.set_index('word').to_dict()['frequency']
    generate_word_cloud(words, picture_name='wordcloud.png')
    
    # 词频大于1的词云
    words_small = word_frequency_df_small.set_index('word').to_dict()['frequency']
    generate_word_cloud(words_small, picture_name='wordcloud_small.png')
    
    # 按标签生成词云
    for label in [0, 1, 2]:
        index = comments['label1'] == label
        word_frequency_df_label = word_count(
            comments['f_word_list'][index], 
            file_name=f'word_frequency_{label}.xlsx', 
            threshold=0
        )
        words_label = word_frequency_df_label.set_index('word').to_dict()['frequency']
        generate_word_cloud(words_label, picture_name=f'wordcloud_{label}.png')

def plot_model_performance(results, model_names, metrics=None):
    '''
    绘制模型性能比较图
    
    Args:
        results: 包含各个模型评估结果的列表
        model_names: 模型名称列表
        metrics: 需要绘制的指标列表，默认为['accuracy', 'precision', 'recall', 'f1-score']
    
    Returns:
        None
    '''
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # 提取结果
    metric_values = {metric: [] for metric in metrics}
    
    for result in results:
        for metric in metrics:
            if metric == 'accuracy':
                metric_values[metric].append(result['accuracy'])
            else:
                # 计算各个类别的平均值
                values = [result[str(label)][metric] for label in [0, 1, 2] if str(label) in result]
                metric_values[metric].append(np.mean(values))
    
    # 绘图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].bar(model_names, metric_values[metric])
        axes[i].set_title(f'Average {metric.capitalize()}')
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel(metric.capitalize())
        
        # 添加数值标签
        for j, v in enumerate(metric_values[metric]):
            axes[i].text(j, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.show()