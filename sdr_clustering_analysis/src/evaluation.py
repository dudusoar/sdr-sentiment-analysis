# sdr_clustering_analysis/src/evaluation.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from pathlib import Path
import json

from config import (
    CLUSTER_PROFILES_DIR,
    TEXT_COLUMN,
    MANUAL_SENTIMENT_COLUMN,
    # Add if you plan to use LDA topics here
    # LDA_TOPIC_COLUMN, # Example: 'dominant_topic'
    RESULTS_DIR
)
from src.utils import save_csv # Assuming save_csv is in utils.py

def get_top_keywords_for_clusters(texts_series, cluster_labels, n_keywords=10):
    """
    为每个簇提取TF-IDF最高的关键词。
    Args:
        texts_series (pd.Series): 包含原始文本（或预处理后用于TF-IDF的文本）的Series。
        cluster_labels (np.array or pd.Series): 每条文本对应的簇标签。
        n_keywords (int): 每个簇提取的关键词数量。
    Returns:
        dict: 字典，键是簇标签，值是该簇的关键词列表。
    """
    if texts_series is None or cluster_labels is None:
        print("错误: 文本数据或簇标签为空。")
        return {}

    df_for_tfidf = pd.DataFrame({
        'text': texts_series,
        'cluster': cluster_labels
    })

    top_keywords = {}
    unique_clusters = sorted(df_for_tfidf['cluster'].unique())

    for cluster_id in unique_clusters:
        if cluster_id == -1: # Skip noise points often labeled as -1 by some algorithms
            top_keywords[cluster_id] = ["NOISE_POINTS"]
            continue

        cluster_texts = df_for_tfidf[df_for_tfidf['cluster'] == cluster_id]['text'].tolist()
        if not cluster_texts:
            top_keywords[cluster_id] = ["EMPTY_CLUSTER"]
            continue

        try:
            # It's better to fit TF-IDF on all texts for a global IDF,
            # but for simplicity here, we fit per cluster.
            # For more robust keywords, consider fitting TF-IDF on the entire corpus
            # and then calculating mean TF-IDF scores per cluster.
            # Alternative approach: Fit on all, then for each cluster, sum tf-idf vectors and get top words.
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across all documents in the cluster
            summed_tfidf = tfidf_matrix.sum(axis=0)
            # Convert to a 1D array
            summed_tfidf_array = np.asarray(summed_tfidf).flatten()
            
            # Get indices of top N keywords
            top_indices = summed_tfidf_array.argsort()[-n_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            top_keywords[cluster_id] = keywords
        except ValueError as e: # Happens if vocabulary is empty (e.g. all stop words)
            print(f"警告: 无法为簇 {cluster_id} 生成关键词: {e}")
            top_keywords[cluster_id] = ["ERROR_GENERATING_KEYWORDS"]
            
    print("\n各簇的代表性关键词:")
    for cid, words in top_keywords.items():
        print(f"  簇 {cid}: {', '.join(words)}")
    return top_keywords

def get_sample_comments_for_clusters(df_with_clusters,
                                     original_text_column=TEXT_COLUMN, # From config
                                     cluster_label_column='cluster_label',
                                     n_samples=5):
    """
    为每个簇抽取一些样本评论。
    Args:
        df_with_clusters (pd.DataFrame): 包含原始文本和簇标签的DataFrame。
        original_text_column (str): DataFrame中包含原始评论文本的列名。
        cluster_label_column (str): DataFrame中包含簇标签的列名。
        n_samples (int): 每个簇抽取的样本数量。
    Returns:
        dict: 字典，键是簇标签，值是该簇的样本评论列表。
    """
    if df_with_clusters is None or original_text_column not in df_with_clusters.columns or \
       cluster_label_column not in df_with_clusters.columns:
        print("错误: DataFrame无效或缺少必要的列。")
        return {}

    sample_comments = {}
    unique_clusters = sorted(df_with_clusters[cluster_label_column].unique())

    print(f"\n各簇的样本评论 (每个簇最多显示{n_samples}条):")
    for cluster_id in unique_clusters:
        cluster_df = df_with_clusters[df_with_clusters[cluster_label_column] == cluster_id]
        samples = cluster_df[original_text_column].sample(min(n_samples, len(cluster_df)), random_state=42).tolist()
        sample_comments[cluster_id] = samples
        print(f"  --- 簇 {cluster_id} (共 {len(cluster_df)} 条) ---")
        for i, comment in enumerate(samples):
            print(f"    Sample {i+1}: {comment[:150] + '...' if len(comment) > 150 else comment}") # Truncate long comments
    return sample_comments

def analyze_sentiment_distribution_in_clusters(df_with_clusters,
                                               cluster_label_column='cluster_label',
                                               sentiment_column=MANUAL_SENTIMENT_COLUMN):
    """分析手动情感标签在每个簇中的分布，只关注标签0、1、2"""
    if df_with_clusters is None or cluster_label_column not in df_with_clusters.columns or \
       sentiment_column not in df_with_clusters.columns:
        print("错误: DataFrame无效或缺少簇标签列或情感标签列。")
        if df_with_clusters is not None:
            print(f"DataFrame columns: {df_with_clusters.columns.tolist()}")
        print(f"Expected cluster column: {cluster_label_column}, Expected sentiment column: {sentiment_column}")
        return pd.DataFrame()

    print(f"\n分析情感标签 ('{sentiment_column}') 在各簇 ('{cluster_label_column}') 中的分布:")
    
    # 只保留情感标签0、1、2的数据
    df_filtered = df_with_clusters[df_with_clusters[sentiment_column].isin([0, 1, 2])].copy()
    df_filtered[sentiment_column] = pd.Categorical(df_filtered[sentiment_column])

    crosstab_abs = pd.crosstab(df_filtered[cluster_label_column], df_filtered[sentiment_column], dropna=False)
    crosstab_norm = pd.crosstab(df_filtered[cluster_label_column], df_filtered[sentiment_column], normalize='index', dropna=False)
    
    print("\n情感标签分布 (绝对数量):")
    print(crosstab_abs)
    print("\n情感标签分布 (簇内百分比):")
    print(crosstab_norm.round(4) * 100)
    
    return crosstab_abs, crosstab_norm

# Placeholder for LDA topic distribution - you'll need to adapt this
# if you have LDA topic assignments for each comment in your DataFrame.
# def analyze_lda_topic_distribution_in_clusters(df_with_clusters,
#                                                cluster_label_column='cluster_label',
#                                                lda_topic_column=LDA_TOPIC_COLUMN): # From config
#     """
#     分析LDA主题在每个簇中的分布。
#     Args:
#         df_with_clusters (pd.DataFrame): 包含簇标签和LDA主题标签的DataFrame。
#         cluster_label_column (str): DataFrame中包含簇标签的列名。
#         lda_topic_column (str): DataFrame中包含主要LDA主题的列名。
#     Returns:
#         pd.DataFrame: 交叉表，显示LDA主题在各簇的分布。
#     """
#     if df_with_clusters is None or cluster_label_column not in df_with_clusters.columns or \
#        lda_topic_column not in df_with_clusters.columns:
#         print("错误: DataFrame无效或缺少簇标签列或LDA主题列。")
#         return pd.DataFrame()

#     print(f"\n分析LDA主题 ('{lda_topic_column}') 在各簇 ('{cluster_label_column}') 中的分布:")
#     df_with_clusters[lda_topic_column] = pd.Categorical(df_with_clusters[lda_topic_column])
#     crosstab_lda_abs = pd.crosstab(df_with_clusters[cluster_label_column], df_with_clusters[lda_topic_column], dropna=False)
#     crosstab_lda_norm = pd.crosstab(df_with_clusters[cluster_label_column], df_with_clusters[lda_topic_column], normalize='index', dropna=False)

#     print("\nLDA主题分布 (绝对数量):")
#     print(crosstab_lda_abs)
#     print("\nLDA主题分布 (簇内百分比):")
#     print(crosstab_lda_norm.round(4) * 100)
    
#     return crosstab_lda_abs, crosstab_lda_norm

def save_cluster_profiles(cluster_keywords, cluster_samples, base_dir=CLUSTER_PROFILES_DIR):
    """将每个簇的关键词和样本评论保存到单独的文件中，并导出为Excel（每个cluster一个Excel文件，两个sheet）。"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建目录: {base_dir}")

    for cluster_id in cluster_keywords.keys():
        # 1. 关键词和评论写入txt（兼容旧用法）
        filepath = os.path.join(base_dir, f"cluster_{cluster_id}_profile.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"--- Cluster {cluster_id} Profile ---\n\n")
            f.write("Top Keywords:\n")
            if cluster_id in cluster_keywords and cluster_keywords[cluster_id]:
                f.write(", ".join(cluster_keywords[cluster_id]))
            else:
                f.write("N/A")
            f.write("\n\n")
            f.write("Sample Comments:\n")
            if cluster_id in cluster_samples and cluster_samples[cluster_id]:
                for i, comment in enumerate(cluster_samples[cluster_id]):
                    f.write(f"  Sample {i+1}: {comment}\n")
            else:
                f.write("N/A")
            f.write("\n")
        print(f"簇 {cluster_id} 的描述文件已保存到: {filepath}")

        # 2. 关键词和所有评论写入Excel
        excel_path = os.path.join(base_dir, f"cluster_{cluster_id}_profile.xlsx")
        # sheet1: keywords
        df_keywords = pd.DataFrame({'keywords': cluster_keywords[cluster_id] if cluster_id in cluster_keywords else []})
        # sheet2: all comments
        df_comments = pd.DataFrame({'comments': cluster_samples[cluster_id] if cluster_id in cluster_samples else []})
        with pd.ExcelWriter(excel_path) as writer:
            df_keywords.to_excel(writer, sheet_name='keywords', index=False)
            df_comments.to_excel(writer, sheet_name='comments', index=False)
        print(f"簇 {cluster_id} 的Excel描述文件已保存到: {excel_path}")

def plot_cluster_size_distribution(cluster_sizes, save_path=None):
    """绘制聚类大小分布图"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cluster_sizes)), list(cluster_sizes.values()))
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Comments')
    plt.title('Distribution of Comments Across Clusters')
    plt.xticks(range(len(cluster_sizes)), list(cluster_sizes.keys()))
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_sentiment_distribution(cluster_sentiment_dist, save_path=None):
    """只显示情感标签0,1,2，并将x轴标签显示为整数"""
    # 只保留0,1,2
    valid_labels = [0, 1, 2]
    cols = [c for c in cluster_sentiment_dist.columns if c in valid_labels or (isinstance(c, float) and int(c) in valid_labels)]
    sentiment_dist = cluster_sentiment_dist[cols]
    sentiment_dist.columns = [int(c) for c in sentiment_dist.columns]
    plt.figure(figsize=(10, 6))
    sns.heatmap(sentiment_dist, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Sentiment Distribution (0: Negative, 1: Positive, 2: Neutral)')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Cluster')
    plt.xticks(ticks=range(len(sentiment_dist.columns)), labels=sentiment_dist.columns)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_silhouette_analysis(X, labels, save_path=None):
    """绘制轮廓系数分析图"""
    from sklearn.metrics import silhouette_samples
    import numpy as np
    
    # 计算每个样本的轮廓系数
    silhouette_vals = silhouette_samples(X, labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = len(cluster_silhouette_vals)
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
    plt.title("Silhouette Analysis")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_cluster_visualizations(df_with_clusters, embeddings, cluster_label_column='cluster_label', vis_dir=None):
    """保存所有聚类可视化结果到指定目录"""
    if vis_dir is None:
        vis_dir = os.path.join(RESULTS_DIR, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # 1. 聚类大小分布
    cluster_sizes = df_with_clusters[cluster_label_column].value_counts().to_dict()
    plot_cluster_size_distribution(
        cluster_sizes,
        save_path=os.path.join(vis_dir, 'cluster_size_distribution.png')
    )
    # 2. 情感分布热力图
    if MANUAL_SENTIMENT_COLUMN in df_with_clusters.columns:
        sentiment_dist = pd.crosstab(
            df_with_clusters[cluster_label_column],
            df_with_clusters[MANUAL_SENTIMENT_COLUMN],
            normalize='index'
        )
        plot_sentiment_distribution(
            sentiment_dist,
            save_path=os.path.join(vis_dir, 'sentiment_distribution.png')
        )
    # 3. 轮廓系数分析
    plot_silhouette_analysis(
        embeddings,
        df_with_clusters[cluster_label_column],
        save_path=os.path.join(vis_dir, 'silhouette_analysis.png')
    )

def extract_keywords_for_clusters(df_with_clusters, embeddings, cluster_label_column='cluster_label', n_keywords=10):
    """为每个簇提取关键词"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # 获取每个簇的文本
    cluster_texts = {}
    for cluster_id in df_with_clusters[cluster_label_column].unique():
        cluster_mask = df_with_clusters[cluster_label_column] == cluster_id
        cluster_texts[cluster_id] = df_with_clusters.loc[cluster_mask, TEXT_COLUMN].tolist()
    
    # 为每个簇提取关键词
    cluster_keywords = {}
    for cluster_id, texts in cluster_texts.items():
        if not texts:  # 跳过空簇
            continue
            
        # 计算TF-IDF
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 计算每个词的平均TF-IDF分数
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 获取top N关键词
        top_indices = np.argsort(avg_tfidf)[-n_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        cluster_keywords[cluster_id] = top_keywords
    
    return cluster_keywords

if __name__ == '__main__':
    print("测试 evaluation.py...")
    # 创建一个模拟的DataFrame，包含文本、簇标签和情感标签
    data = {
        TEXT_COLUMN: [
            "great delivery robot, very fast and efficient sdr", # cluster 0, positive
            "amazing service, the robot is cute and helpful",    # cluster 0, positive
            "sdr technology is the future, love it",             # cluster 0, positive
            "bad experience, the robot got stuck in snow",       # cluster 1, negative
            "terrible sdr, food was cold and delivery late",     # cluster 1, negative
            "this robot is slow and blocks the sidewalk sometimes", # cluster 1, negative/neutral
            "i dont know what to think about these robots",      # cluster 2, neutral
            "neutral comment about sidewalk delivery robots",    # cluster 2, neutral
            "just a robot doing its job on the pavement",        # cluster 2, neutral
            "noise point one not in any cluster",                # cluster -1 (noise)
        ],
        'cluster_label': [0, 0, 0, 1, 1, 1, 2, 2, 2, -1], # 模拟聚类结果
        MANUAL_SENTIMENT_COLUMN: [1, 1, 1, 0, 0, 0, 2, 2, 2, 0] # 模拟手动情感标签
        # Add 'dominant_topic': [0, 1, 0, 2, 2, 1, 0, 1, 2, 0] for LDA testing
    }
    test_df = pd.DataFrame(data)

    # 1. 测试关键词提取
    print("\n--- 测试关键词提取 ---")
    keywords = get_top_keywords_for_clusters(test_df[TEXT_COLUMN], test_df['cluster_label'], n_keywords=3)
    # (实际输出会依赖TF-IDF结果，这里只是调用)

    # 2. 测试样本评论提取
    print("\n--- 测试样本评论提取 ---")
    samples = get_sample_comments_for_clusters(test_df, original_text_column=TEXT_COLUMN, cluster_label_column='cluster_label', n_samples=2)
    # (输出会打印到控制台)

    # 3. 测试情感分布分析
    if MANUAL_SENTIMENT_COLUMN in test_df.columns:
        print("\n--- 测试情感分布分析 ---")
        ct_abs, ct_norm = analyze_sentiment_distribution_in_clusters(test_df, cluster_label_column='cluster_label', sentiment_column=MANUAL_SENTIMENT_COLUMN)
        # (输出会打印到控制台)
    else:
        print(f"\n跳过情感分布分析，因为 '{MANUAL_SENTIMENT_COLUMN}' 列不在DataFrame中。")

    # 4. (可选) 测试LDA主题分布分析 - 需要你有LDA主题列
    # if LDA_TOPIC_COLUMN in test_df.columns:
    #     print("\n--- 测试LDA主题分布分析 ---")
    #     ct_lda_abs, ct_lda_norm = analyze_lda_topic_distribution_in_clusters(test_df, cluster_label_column='cluster_label', lda_topic_column=LDA_TOPIC_COLUMN)
    # else:
    #     print(f"\n跳过LDA主题分布分析，因为 '{LDA_TOPIC_COLUMN}' 列不在DataFrame中或未在config中定义。")

    # 5. 测试保存簇描述文件
    print("\n--- 测试保存簇描述文件 ---")
    save_cluster_profiles(keywords, samples, base_dir=os.path.join(RESULTS_DIR, "test_cluster_profiles"))
    # (会创建 test_cluster_profiles 文件夹并保存文件)

    # 6. 测试保存聚类可视化结果
    print("\n--- 测试保存聚类可视化结果 ---")
    save_cluster_visualizations(test_df, test_df[TEXT_COLUMN].values, cluster_label_column='cluster_label')
    # (会创建 visualizations 文件夹并保存可视化结果)

    print("\nevaluation.py 测试完成。")