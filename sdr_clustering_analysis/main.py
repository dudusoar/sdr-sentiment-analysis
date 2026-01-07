# sdr_clustering_analysis/main.py
import pandas as pd
import os
import argparse
import time

# 项目内部模块导入
from config import (
    RAW_COMMENTS_FILE, TEXT_COLUMN, ID_COLUMN, MANUAL_SENTIMENT_COLUMN,
    EMBEDDINGS_FILE, PREPROCESSED_COMMENTS_FILE,
    SBERT_MODEL_NAME,
    KMEANS_N_CLUSTERS_RANGE, KMEANS_SELECTED_K,
    # HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, # Uncomment if using HDBSCAN
    CLUSTER_ASSIGNMENT_FILE, CLUSTER_EVALUATION_FILE,
    CLUSTER_SENTIMENT_CROSSTAB_FILE, CLUSTER_LDA_CROSSTAB_FILE, # Ensure LDA column is set if used
    create_project_directories
)
from src.data_loader import load_raw_comments
from src.text_preprocessor import preprocess_dataframe # Using the DataFrame version
from src.feature_extractor import get_sentence_embeddings
from src.clustering import Clusterer, evaluate_clustering
from src.evaluation import (
    get_top_keywords_for_clusters,
    get_sample_comments_for_clusters,
    analyze_sentiment_distribution_in_clusters,
    # analyze_lda_topic_distribution_in_clusters, # Uncomment if using LDA
    save_cluster_profiles
)
from src.utils import save_csv, load_pickle, save_pickle

def run_clustering_pipeline(
        raw_file_path=RAW_COMMENTS_FILE,
        text_col=TEXT_COLUMN,
        id_col=ID_COLUMN, # Can be None if not available/needed for final output join
        sentiment_col=MANUAL_SENTIMENT_COLUMN, # Can be None
        # lda_topic_col=None, # Pass this if you have LDA topics, e.g., from config.LDA_TOPIC_COLUMN
        
        use_preprocessing=False, # Set to True if you want to run text_preprocessor
        preprocessed_cache_path=PREPROCESSED_COMMENTS_FILE,
        
        sbert_model=SBERT_MODEL_NAME,
        use_embedding_cache=True,
        embedding_cache_path=EMBEDDINGS_FILE,
        
        clustering_algorithm='kmeans', # 'kmeans' or 'hdbscan'
        kmeans_k_range=KMEANS_N_CLUSTERS_RANGE,
        force_k=KMEANS_SELECTED_K, # Use a specific K for KMeans, otherwise find optimal
        
        n_keywords_per_cluster=10,
        n_samples_per_cluster=5
    ):
    """
    执行完整的聚类分析流程。
    """
    start_time = time.time()
    print("开始执行SDR评论聚类分析流程...")
    create_project_directories() # 确保结果目录存在

    # 1. 加载数据
    print("\n--- 步骤1: 加载原始数据 ---")
    raw_df = load_raw_comments(filepath=raw_file_path)
    if raw_df is None or raw_df.empty:
        print("数据加载失败或数据为空，流程终止。")
        return
    
    # 确保后续使用的ID列是 'unique_comment_id' (由data_loader保证创建)
    current_id_col = 'unique_comment_id'


    # 2. (可选) 文本预处理
    # 对于SBERT，通常不需要复杂的预处理，但如果需要，可以在这里进行
    texts_to_embed = raw_df[text_col] # Default to original text column
    
    if use_preprocessing:
        print(f"\n--- 步骤2: 文本预处理 (use_preprocessing=True) ---")
        if use_embedding_cache and os.path.exists(preprocessed_cache_path):
            print(f"尝试从缓存加载预处理文本: {preprocessed_cache_path}")
            processed_df = load_pickle(preprocessed_cache_path)
            if processed_df is not None and 'preprocessed_text' in processed_df.columns and \
               len(processed_df) == len(raw_df):
                texts_to_embed = processed_df['preprocessed_text']
                print("成功从缓存加载预处理文本。")
            else:
                print("预处理文本缓存无效，将重新处理。")
                processed_df = preprocess_dataframe(raw_df.copy(), text_column=text_col, new_column_name='preprocessed_text')
                texts_to_embed = processed_df['preprocessed_text']
                save_pickle(processed_df[['preprocessed_text']], preprocessed_cache_path) # Save only relevant part
        else:
            processed_df = preprocess_dataframe(raw_df.copy(), text_column=text_col, new_column_name='preprocessed_text')
            texts_to_embed = processed_df['preprocessed_text']
            if use_embedding_cache: # implies saving preprocessed if embeddings will be cached
                 save_pickle(processed_df[['preprocessed_text']], preprocessed_cache_path)
    else:
        print(f"\n--- 步骤2: 文本预处理已跳过 (use_preprocessing=False) ---")
        # texts_to_embed is already raw_df[text_col]

    if texts_to_embed.empty:
        print("错误: 预处理后/选择的文本数据为空，流程终止。")
        return

    # 3. 特征提取 (Sentence Embeddings)
    print("\n--- 步骤3: 特征提取 (Sentence Embeddings) ---")
    embeddings = get_sentence_embeddings(
        texts_series=texts_to_embed,
        use_cache=use_embedding_cache,
        cache_filepath=embedding_cache_path,
        sbert_model_name=sbert_model
    )
    if embeddings is None or embeddings.shape[0] == 0:
        print("句子嵌入提取失败或返回空，流程终止。")
        return
    if embeddings.shape[0] != len(raw_df):
        print(f"错误: 嵌入向量数量 ({embeddings.shape[0]}) 与原始评论数量 ({len(raw_df)}) 不匹配。")
        print("这可能由文本预处理或嵌入过程中的错误导致。流程终止。")
        return

    # 4. 聚类
    print(f"\n--- 步骤4: 执行 {clustering_algorithm.upper()} 聚类 ---")
    cluster_analyzer = Clusterer()
    cluster_labels = None
    model_details = None # To store the trained clustering model if needed

    if clustering_algorithm.lower() == 'kmeans':
        if force_k:
            print(f"使用指定的K值进行K-Means聚类: K={force_k}")
            selected_k = force_k
        else:
            print(f"通过评估指标寻找K-Means的最佳K值 (范围: {list(kmeans_k_range)})")
            k_eval_df = cluster_analyzer.find_optimal_k_kmeans(embeddings, k_range=kmeans_k_range)
            print("\nK值评估 (Inertia & Silhouette Score):")
            print(k_eval_df)
            # 简单的K选择逻辑：选择轮廓系数最高的K
            if not k_eval_df.empty:
                # Ensure Silhouette_Score is numeric and handle potential NaNs by filling with a very low value
                k_eval_df['Silhouette_Score_numeric'] = pd.to_numeric(k_eval_df['Silhouette_Score'], errors='coerce').fillna(-2)
                best_k_row = k_eval_df.loc[k_eval_df['Silhouette_Score_numeric'].idxmax()]
                selected_k = int(best_k_row['K'])
                print(f"根据最高轮廓系数 ({best_k_row['Silhouette_Score_numeric']:.4f}) 自动选择的K值: {selected_k}")
            else:
                print(f"无法自动选择K值，将使用范围中的第一个值: {min(kmeans_k_range)}")
                selected_k = min(kmeans_k_range)
        
        model_details, cluster_labels = cluster_analyzer.kmeans_cluster(embeddings, n_clusters=selected_k)

    # elif clustering_algorithm.lower() == 'hdbscan': # Uncomment and implement if using HDBSCAN
    #     model_details, cluster_labels = cluster_analyzer.hdbscan_cluster(
    #         embeddings,
    #         min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    #         min_samples=HDBSCAN_MIN_SAMPLES
    #     )
    else:
        print(f"错误: 不支持的聚类算法 '{clustering_algorithm}'。请选择 'kmeans'。") # Add hdbscan when ready
        return

    if cluster_labels is None:
        print("聚类失败，流程终止。")
        return

    # 将簇标签添加到原始DataFrame
    # Ensuring correct alignment if any rows were dropped during preprocessing/embedding
    results_df = raw_df.copy() # Start with a fresh copy of the original df
    results_df['cluster_label'] = cluster_labels
    
    # 保存簇分配结果
    if current_id_col in results_df.columns and text_col in results_df.columns:
        save_csv(results_df[[current_id_col, text_col, 'cluster_label']], CLUSTER_ASSIGNMENT_FILE)
    else: # Fallback if columns are missing, save what's available
        save_csv(results_df[['cluster_label']], CLUSTER_ASSIGNMENT_FILE)


    # 5. 聚类评估与分析
    print("\n--- 步骤5: 聚类评估与分析 ---")
    # 5a. 内部评估指标
    print("\n--- 5a. 内部聚类评估指标 ---")
    clustering_metrics = evaluate_clustering(embeddings, cluster_labels)
    with open(CLUSTER_EVALUATION_FILE, 'w', encoding='utf-8') as f:
        f.write(f"聚类算法: {clustering_algorithm.upper()}\n")
        if clustering_algorithm.lower() == 'kmeans':
            f.write(f"选择的K值: {selected_k}\n")
        f.write("\n评估指标:\n")
        for metric, value in clustering_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    print(f"聚类评估指标已保存到: {CLUSTER_EVALUATION_FILE}")

    # 5b. 簇描述：关键词和样本评论
    print("\n--- 5b. 生成簇描述文件 ---")
    # 使用原始文本或预处理文本进行关键词提取，取决于哪个更有意义
    # SBERT通常基于原始文本，但TF-IDF关键词可能从略微清理的文本中获益
    texts_for_keywords = texts_to_embed # Or raw_df[text_col] if preferred
    
    cluster_keywords = get_top_keywords_for_clusters(
        texts_series=texts_for_keywords,
        cluster_labels=cluster_labels,
        n_keywords=n_keywords_per_cluster
    )
    cluster_samples = get_sample_comments_for_clusters(
        df_with_clusters=results_df, # results_df now contains original text and cluster_label
        original_text_column=text_col,
        cluster_label_column='cluster_label',
        n_samples=n_samples_per_cluster
    )
    save_cluster_profiles(cluster_keywords, cluster_samples) # Saves to CLUSTER_PROFILES_DIR

    # 5c. 聚类与手动情感标签交叉分析
    if sentiment_col and sentiment_col in results_df.columns:
        print(f"\n--- 5c. 分析聚类与 '{sentiment_col}' (手动情感标签) 的关系 ---")
        senti_ct_abs, senti_ct_norm = analyze_sentiment_distribution_in_clusters(
            df_with_clusters=results_df,
            cluster_label_column='cluster_label',
            sentiment_column=sentiment_col
        )
        save_csv(senti_ct_abs, CLUSTER_SENTIMENT_CROSSTAB_FILE.replace(".csv", "_absolute.csv"), index=True)
        save_csv(senti_ct_norm, CLUSTER_SENTIMENT_CROSSTAB_FILE.replace(".csv", "_normalized.csv"), index=True)
    else:
        print(f"\n提示: 未提供有效的手动情感标签列 ('{sentiment_col}')，跳过聚类与情感交叉分析。")

    # 5d. (可选) 聚类与LDA主题交叉分析
    # if lda_topic_col and lda_topic_col in results_df.columns:
    #     print(f"\n--- 5d. 分析聚类与 '{lda_topic_col}' (LDA主题) 的关系 ---")
    #     lda_ct_abs, lda_ct_norm = analyze_lda_topic_distribution_in_clusters(
    #         df_with_clusters=results_df,
    #         cluster_label_column='cluster_label',
    #         lda_topic_column=lda_topic_col
    #     )
    #     save_csv(lda_ct_abs, CLUSTER_LDA_CROSSTAB_FILE.replace(".csv", "_absolute.csv"), index=True)
    #     save_csv(lda_ct_norm, CLUSTER_LDA_CROSSTAB_FILE.replace(".csv", "_normalized.csv"), index=True)
    # else:
    #     print(f"\n提示: 未提供有效的LDA主题列 ('{lda_topic_col}')，跳过聚类与LDA交叉分析。")


    end_time = time.time()
    print(f"\nSDR评论聚类分析流程完成。总耗时: {end_time - start_time:.2f} 秒。")
    print(f"所有结果已保存到目录: {os.path.abspath(RESULTS_DIR)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SDR YouTube评论聚类分析工具")
    parser.add_argument(
        '--raw_file', type=str, default=RAW_COMMENTS_FILE,
        help="原始评论数据文件的路径 (.xlsx 或 .csv)"
    )
    parser.add_argument(
        '--text_column', type=str, default=TEXT_COLUMN,
        help="包含评论文本的列名"
    )
    parser.add_argument(
        '--id_column', type=str, default=ID_COLUMN,
        help="评论ID的列名 (可选)"
    )
    parser.add_argument(
        '--sentiment_column', type=str, default=MANUAL_SENTIMENT_COLUMN,
        help="手动情感标签的列名 (可选, 用于对比分析)"
    )
    # parser.add_argument( # Uncomment if you implement LDA topic integration
    #     '--lda_topic_column', type=str, default=None, # Example: config.LDA_TOPIC_COLUMN
    #     help="LDA主题标签的列名 (可选, 用于对比分析)"
    # )
    parser.add_argument(
        '--use_preprocessing', action='store_true',
        help="是否对文本进行预处理 (默认为False, SBERT通常不需要复杂预处理)"
    )
    parser.add_argument(
        '--sbert_model', type=str, default=SBERT_MODEL_NAME,
        help="SentenceTransformer模型名称"
    )
    parser.add_argument(
        '--no_embedding_cache', action='store_false', dest='use_embedding_cache',
        help="不使用或保存句子嵌入缓存"
    )
    parser.add_argument(
        '--algorithm', type=str, default='kmeans', choices=['kmeans'], # Add 'hdbscan' when ready
        help="要使用的聚类算法"
    )
    parser.add_argument(
        '--k', type=int, default=KMEANS_SELECTED_K, # Default from config, could be None
        help="为K-Means指定K值。如果未提供，将尝试自动寻找最佳K。"
    )
    parser.add_argument(
        '--k_min', type=int, default=min(KMEANS_N_CLUSTERS_RANGE) if KMEANS_N_CLUSTERS_RANGE else 2,
        help="自动寻找K值时的最小K (仅当 --k 未指定时)"
    )
    parser.add_argument(
        '--k_max', type=int, default=max(KMEANS_N_CLUSTERS_RANGE) if KMEANS_N_CLUSTERS_RANGE else 7,
        help="自动寻找K值时的最大K (仅当 --k 未指定时)"
    )

    args = parser.parse_args()

    # 更新KMeans K值范围如果通过命令行指定了范围且未指定固定K
    k_range_to_use = range(args.k_min, args.k_max + 1)
    if args.k: # User specified a K
        force_k_value = args.k
    else: # User did not specify a K, so we search in the range
        force_k_value = None # This will trigger find_optimal_k_kmeans
        print(f"将搜索K值范围: {list(k_range_to_use)}")
        # Update config-like variable for the pipeline based on CLI args
        # This is a bit of a workaround as config isn't dynamically updated
        # but pipeline function takes these as direct args.
    
    run_clustering_pipeline(
        raw_file_path=args.raw_file,
        text_col=args.text_column,
        id_col=args.id_column,
        sentiment_col=args.sentiment_column,
        # lda_topic_col=args.lda_topic_column, # Uncomment for LDA
        use_preprocessing=args.use_preprocessing,
        sbert_model=args.sbert_model,
        use_embedding_cache=args.use_embedding_cache,
        clustering_algorithm=args.algorithm,
        kmeans_k_range=k_range_to_use, # Pass the potentially updated range
        force_k=force_k_value # Pass the specific K or None
    )