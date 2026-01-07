# sdr_clustering_analysis/src/clustering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.cluster import HDBSCAN # Uncomment if you plan to implement HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt # For Elbow method visualization
import os

from config import KMEANS_RANDOM_STATE, KMEANS_N_CLUSTERS_RANGE, RESULTS_DIR
# from config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES # Uncomment for HDBSCAN

class Clusterer:
    def __init__(self, random_state=KMEANS_RANDOM_STATE):
        self.random_state = random_state
        self.kmeans_models_ = {} # To store trained KMeans models for different K
        self.inertia_ = {} # To store inertia for different K for Elbow method

    def _plot_elbow_method(self, k_range, inertia_values, filepath=None):
        """Helper function to plot the Elbow method graph."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia_values, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('Elbow Method for Optimal K')
        plt.xticks(k_range)
        plt.grid(True)
        if filepath:
            plt.savefig(filepath)
            print(f"Elbow method plot saved to: {filepath}")
        plt.show()

    def find_optimal_k_kmeans(self, embeddings, k_range=KMEANS_N_CLUSTERS_RANGE, plot_elbow=True):
        """
        Tries different K values for KMeans and calculates inertia and silhouette scores.
        Plots the Elbow method graph.
        Args:
            embeddings (numpy.ndarray): The input sentence embeddings.
            k_range (range): Range of K values to test (e.g., range(2, 11)).
            plot_elbow (bool): Whether to plot the Elbow method graph.
        Returns:
            pd.DataFrame: DataFrame with K, Inertia, and Silhouette Score for each K.
        """
        if embeddings is None or embeddings.shape[0] < max(k_range): # Ensure enough samples
            print("错误: 嵌入数据不足或K值范围过大。")
            return pd.DataFrame()

        results = []
        self.inertia_ = {}
        print(f"正在为K-Means寻找最优K值，测试范围: {list(k_range)}")

        for k in k_range:
            if embeddings.shape[0] < k:
                print(f"警告: 样本数量 ({embeddings.shape[0]}) 小于K值 ({k})，跳过K={k}。")
                continue
            
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings)
            
            inertia = kmeans.inertia_
            self.inertia_[k] = inertia
            
            # Silhouette score requires at least 2 labels and more samples than clusters
            if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < embeddings.shape[0] :
                silhouette = silhouette_score(embeddings, cluster_labels)
            else:
                silhouette = -1 # Invalid or not calculable
            
            print(f"  K={k}: Inertia={inertia:.2f}, Silhouette Score={silhouette:.4f}")
            results.append({'K': k, 'Inertia': inertia, 'Silhouette_Score': silhouette})
            self.kmeans_models_[k] = kmeans # Store the trained model

        results_df = pd.DataFrame(results)

        if plot_elbow and self.inertia_:
            elbow_plot_path = os.path.join(RESULTS_DIR, 'elbow_method_plot.png')
            self._plot_elbow_method(list(self.inertia_.keys()), list(self.inertia_.values()), filepath=elbow_plot_path)
            
        return results_df

    def kmeans_cluster(self, embeddings, n_clusters):
        """
        Performs K-Means clustering for a given number of clusters.
        Args:
            embeddings (numpy.ndarray): The input sentence embeddings.
            n_clusters (int): The desired number of clusters.
        Returns:
            tuple: (KMeans model, cluster_labels (numpy.ndarray)) or (None, None) if error.
        """
        if embeddings is None or embeddings.shape[0] < n_clusters:
            print(f"错误: 嵌入数据不足以形成 {n_clusters} 个簇。")
            return None, None
        
        if n_clusters in self.kmeans_models_:
            print(f"使用之前为 K={n_clusters} 训练的KMeans模型。")
            kmeans_model = self.kmeans_models_[n_clusters]
            # Recalculate labels if needed, or assume fit_predict was called if find_optimal_k was run
            cluster_labels = kmeans_model.labels_ # or kmeans_model.predict(embeddings) if only fit was done
        else:
            print(f"为 K={n_clusters} 训练新的KMeans模型。")
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
            cluster_labels = kmeans_model.fit_predict(embeddings)
            self.kmeans_models_[n_clusters] = kmeans_model # Store the new model
            self.inertia_[n_clusters] = kmeans_model.inertia_

        print(f"K-Means聚类完成，分为 {n_clusters} 个簇。")
        return kmeans_model, cluster_labels

    # --- HDBSCAN (Optional) ---
    # def hdbscan_cluster(self, embeddings, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES):
    #     """
    #     Performs HDBSCAN clustering.
    #     Args:
    #         embeddings (numpy.ndarray): The input sentence embeddings.
    #         min_cluster_size (int): The minimum size of clusters.
    #         min_samples (int, optional): The number of samples in a neighbourhood for a point
    #                                     to be considered as a core point.
    #     Returns:
    #         tuple: (HDBSCAN model, cluster_labels (numpy.ndarray)) or (None, None) if error.
    #     """
    #     if embeddings is None:
    #         print("错误: 嵌入数据为空。")
    #         return None, None
    #     try:
    #         clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
    #                             min_samples=min_samples,
    #                             metric='euclidean', # Common for SBERT embeddings
    #                             gen_min_span_tree=True) # Useful for some visualizations/analyses
    #         cluster_labels = clusterer.fit_predict(embeddings)
    #         n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    #         print(f"HDBSCAN聚类完成。找到 {n_clusters_found} 个簇。有 {np.sum(cluster_labels == -1)} 个噪声点。")
    #         return clusterer, cluster_labels
    #     except Exception as e:
    #         print(f"HDBSCAN聚类失败: {e}")
    #         print("请确保已安装hdbscan库 (pip install hdbscan)。")
    #         return None, None

def evaluate_clustering(embeddings, labels):
    """
    Evaluates clustering performance using various metrics.
    Assumes labels are not -1 (noise for some algorithms like DBSCAN/HDBSCAN).
    Filters out noise points before calculation if present.
    """
    if embeddings is None or labels is None:
        return {}
        
    # Filter out noise points (label == -1) for silhouette, calinski_harabasz, davies_bouldin
    # as these metrics are not designed for noise or require specific handling.
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) < 2 or len(set(labels[non_noise_mask])) < 2 : # Not enough (non-noise) samples or clusters
        print("警告: 噪声点过滤后，样本或簇数量不足以计算评估指标。")
        return {"num_clusters_effective": len(set(labels[non_noise_mask]))}

    filtered_embeddings = embeddings[non_noise_mask]
    filtered_labels = labels[non_noise_mask]
    
    # Check again after filtering
    if len(set(filtered_labels)) < 2 or filtered_embeddings.shape[0] <= len(set(filtered_labels)):
        print("警告: 过滤噪声点后，有效簇数量不足2个，或样本数不大于簇数，无法计算部分指标。")
        return {
            "num_clusters_effective": len(set(filtered_labels)),
            "num_noise_points": np.sum(labels == -1)
        }

    metrics = {
        "silhouette_score": silhouette_score(filtered_embeddings, filtered_labels),
        "calinski_harabasz_score": calinski_harabasz_score(filtered_embeddings, filtered_labels),
        "davies_bouldin_score": davies_bouldin_score(filtered_embeddings, filtered_labels),
        "num_clusters_effective": len(set(filtered_labels)), # Number of actual clusters (excluding noise)
        "num_noise_points": np.sum(labels == -1) # For algorithms like HDBSCAN
    }
    print("\n聚类评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    return metrics


if __name__ == '__main__':
    print("测试 clustering.py...")

    # 生成一些假的嵌入数据用于测试
    # 假设有100条评论，每条评论是768维的嵌入 (SBERT常用维度)
    # 或者更小维度如384 for all-MiniLM-L6-v2
    from config import SBERT_MODEL_NAME
    from src.feature_extractor import SentenceEmbedder
    
    sbert_dim = SentenceEmbedder(SBERT_MODEL_NAME).model.get_sentence_embedding_dimension()
    if sbert_dim is None:
        sbert_dim = 384 # Fallback if model loading failed in test
        print(f"警告: 无法从模型获取维度，使用默认值 {sbert_dim}")

    np.random.seed(KMEANS_RANDOM_STATE)
    test_embeddings = np.random.rand(100, sbert_dim)

    cluster_analyzer = Clusterer()

    # 1. 测试寻找最优K
    print("\n--- 测试 K-Means K值选择 ---")
    k_eval_df = cluster_analyzer.find_optimal_k_kmeans(test_embeddings, k_range=range(2, 6))
    print("K值评估结果:")
    print(k_eval_df)

    # 2. 测试K-Means聚类 (假设我们选择K=3)
    chosen_k = 3
    if chosen_k in cluster_analyzer.kmeans_models_: # Check if model for k=3 was trained
        print(f"\n--- 测试 K-Means 聚类 (K={chosen_k}) ---")
        kmeans_model, kmeans_labels = cluster_analyzer.kmeans_cluster(test_embeddings, n_clusters=chosen_k)
        if kmeans_labels is not None:
            print(f"K-Means聚类标签 (前10个): {kmeans_labels[:10]}")
            print(f"簇标签分布: {pd.Series(kmeans_labels).value_counts().sort_index().to_dict()}")
            
            # 评估K-Means结果
            kmeans_metrics = evaluate_clustering(test_embeddings, kmeans_labels)
    else:
        print(f"K={chosen_k} 的模型未在find_optimal_k中训练，跳过K-Means测试。")


    # --- (可选) 测试HDBSCAN ---
    # print("\n--- 测试 HDBSCAN 聚类 ---")
    # Install hdbscan first: pip install hdbscan
    # try:
    #     hdbscan_model, hdbscan_labels = cluster_analyzer.hdbscan_cluster(test_embeddings)
    #     if hdbscan_labels is not None:
    #         print(f"HDBSCAN聚类标签 (前10个): {hdbscan_labels[:10]}")
    #         print(f"簇标签分布: {pd.Series(hdbscan_labels).value_counts().sort_index().to_dict()}")
    #         # 评估HDBSCAN结果 (HDBSCAN可以产生噪声点-1)
    #         hdbscan_metrics = evaluate_clustering(test_embeddings, hdbscan_labels)
    # except ImportError:
    #     print("HDBSCAN库未安装，跳过HDBSCAN测试。请运行 'pip install hdbscan'")
    # except Exception as e:
    #     print(f"HDBSCAN测试中发生错误: {e}")

    print("\nclustering.py 测试完成。")