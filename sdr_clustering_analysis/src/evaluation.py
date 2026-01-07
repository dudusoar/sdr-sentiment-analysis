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
from src.utils import save_csv  # Assuming save_csv is in utils.py

def get_top_keywords_for_clusters(texts_series, cluster_labels, n_keywords=10):
    """
    Extract the top TF-IDF keywords for each cluster.
    Args:
        texts_series (pd.Series): Series containing the raw text (or preprocessed text for TF-IDF).
        cluster_labels (np.array or pd.Series): Cluster labels for each text.
        n_keywords (int): Number of keywords to extract per cluster.
    Returns:
        dict: Dictionary where keys are cluster IDs and values are lists of keywords for that cluster.
    """
    if texts_series is None or cluster_labels is None:
        print("Error: Text data or cluster labels are empty.")
        return {}

    df_for_tfidf = pd.DataFrame({
        'text': texts_series,
        'cluster': cluster_labels
    })

    top_keywords = {}
    unique_clusters = sorted(df_for_tfidf['cluster'].unique())

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points often labeled as -1 by some algorithms
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

            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
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
        except ValueError as e:  # Happens if vocabulary is empty (e.g. all stop words)
            print(f"Warning: Unable to generate keywords for cluster {cluster_id}: {e}")
            top_keywords[cluster_id] = ["ERROR_GENERATING_KEYWORDS"]

    print("\nRepresentative keywords by cluster:")
    for cid, words in top_keywords.items():
        print(f"  Cluster {cid}: {', '.join(words)}")
    return top_keywords

def get_sample_comments_for_clusters(df_with_clusters,
                                     original_text_column=TEXT_COLUMN,  # From config
                                     cluster_label_column='cluster_label',
                                     n_samples=5):
    """
    Sample a few comments for each cluster.
    Args:
        df_with_clusters (pd.DataFrame): DataFrame containing original text and cluster labels.
        original_text_column (str): Column name in the DataFrame that contains the original comment text.
        cluster_label_column (str): Column name in the DataFrame that contains the cluster label.
        n_samples (int): Number of samples to draw per cluster.
    Returns:
        dict: Dictionary where keys are cluster IDs and values are lists of sampled comments for that cluster.
    """
    if df_with_clusters is None or original_text_column not in df_with_clusters.columns or \
       cluster_label_column not in df_with_clusters.columns:
        print("Error: Invalid DataFrame or missing required columns.")
        return {}

    sample_comments = {}
    unique_clusters = sorted(df_with_clusters[cluster_label_column].unique())

    print(f"\nSample comments by cluster (up to {n_samples} per cluster):")
    for cluster_id in unique_clusters:
        cluster_df = df_with_clusters[df_with_clusters[cluster_label_column] == cluster_id]
        samples = cluster_df[original_text_column].sample(
            min(n_samples, len(cluster_df)), random_state=42
        ).tolist()
        sample_comments[cluster_id] = samples
        print(f"  --- Cluster {cluster_id} (total {len(cluster_df)} comments) ---")
        for i, comment in enumerate(samples):
            print(f"    Sample {i+1}: {comment[:150] + '...' if len(comment) > 150 else comment}")  # Truncate long comments
    return sample_comments

def analyze_sentiment_distribution_in_clusters(df_with_clusters,
                                               cluster_label_column='cluster_label',
                                               sentiment_column=MANUAL_SENTIMENT_COLUMN):
    """Analyze the distribution of manual sentiment labels within each cluster, focusing only on labels 0, 1, and 2."""
    if df_with_clusters is None or cluster_label_column not in df_with_clusters.columns or \
       sentiment_column not in df_with_clusters.columns:
        print("Error: Invalid DataFrame or missing cluster label column or sentiment label column.")
        if df_with_clusters is not None:
            print(f"DataFrame columns: {df_with_clusters.columns.tolist()}")
        print(f"Expected cluster column: {cluster_label_column}, Expected sentiment column: {sentiment_column}")
        return pd.DataFrame()

    print(f"\nAnalyzing sentiment label ('{sentiment_column}') distribution across clusters ('{cluster_label_column}'):")

    # Keep only rows with sentiment labels 0, 1, 2
    df_filtered = df_with_clusters[df_with_clusters[sentiment_column].isin([0, 1, 2])].copy()
    df_filtered[sentiment_column] = pd.Categorical(df_filtered[sentiment_column])

    crosstab_abs = pd.crosstab(df_filtered[cluster_label_column], df_filtered[sentiment_column], dropna=False)
    crosstab_norm = pd.crosstab(df_filtered[cluster_label_column], df_filtered[sentiment_column], normalize='index', dropna=False)

    print("\nSentiment label distribution (absolute counts):")
    print(crosstab_abs)
    print("\nSentiment label distribution (percentages within cluster):")
    print(crosstab_norm.round(4) * 100)

    return crosstab_abs, crosstab_norm

# Placeholder for LDA topic distribution - you'll need to adapt this
# if you have LDA topic assignments for each comment in your DataFrame.
# def analyze_lda_topic_distribution_in_clusters(df_with_clusters,
#                                                cluster_label_column='cluster_label',
#                                                lda_topic_column=LDA_TOPIC_COLUMN):  # From config
#     """
#     Analyze the distribution of LDA topics within each cluster.
#     Args:
#         df_with_clusters (pd.DataFrame): DataFrame containing cluster labels and LDA topic labels.
#         cluster_label_column (str): Column name in the DataFrame that contains cluster labels.
#         lda_topic_column (str): Column name in the DataFrame that contains the dominant LDA topic.
#     Returns:
#         pd.DataFrame: Crosstab showing distribution of LDA topics across clusters.
#     """
#     if df_with_clusters is None or cluster_label_column not in df_with_clusters.columns or \
#        lda_topic_column not in df_with_clusters.columns:
#         print("Error: Invalid DataFrame or missing cluster label column or LDA topic column.")
#         return pd.DataFrame()
#
#     print(f"\nAnalyzing LDA topic ('{lda_topic_column}') distribution across clusters ('{cluster_label_column}'):")
#     df_with_clusters[lda_topic_column] = pd.Categorical(df_with_clusters[lda_topic_column])
#     crosstab_lda_abs = pd.crosstab(df_with_clusters[cluster_label_column], df_with_clusters[lda_topic_column], dropna=False)
#     crosstab_lda_norm = pd.crosstab(df_with_clusters[cluster_label_column], df_with_clusters[lda_topic_column], normalize='index', dropna=False)
#
#     print("\nLDA topic distribution (absolute counts):")
#     print(crosstab_lda_abs)
#     print("\nLDA topic distribution (percentages within cluster):")
#     print(crosstab_lda_norm.round(4) * 100)
#
#     return crosstab_lda_abs, crosstab_lda_norm

def save_cluster_profiles(cluster_keywords, cluster_samples, base_dir=CLUSTER_PROFILES_DIR):
    """Save each cluster's keywords and sample comments to separate files, and export to Excel (one Excel per cluster, two sheets)."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")

    for cluster_id in cluster_keywords.keys():
        # 1. Write keywords and comments to txt (backward-compatible with older usage)
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
        print(f"Cluster {cluster_id} profile file saved to: {filepath}")

        # 2. Write keywords and all comments to Excel
        excel_path = os.path.join(base_dir, f"cluster_{cluster_id}_profile.xlsx")
        # sheet1: keywords
        df_keywords = pd.DataFrame({'keywords': cluster_keywords[cluster_id] if cluster_id in cluster_keywords else []})
        # sheet2: all comments
        df_comments = pd.DataFrame({'comments': cluster_samples[cluster_id] if cluster_id in cluster_samples else []})
        with pd.ExcelWriter(excel_path) as writer:
            df_keywords.to_excel(writer, sheet_name='keywords', index=False)
            df_comments.to_excel(writer, sheet_name='comments', index=False)
        print(f"Cluster {cluster_id} Excel profile saved to: {excel_path}")

def plot_cluster_size_distribution(cluster_sizes, save_path=None):
    """Plot the distribution of cluster sizes."""
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
    """Show only sentiment labels 0, 1, 2, and display x-axis labels as integers."""
    # Keep only 0, 1, 2
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
    """Plot a silhouette analysis chart."""
    from sklearn.metrics import silhouette_samples
    import numpy as np

    # Compute silhouette coefficient for each sample
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
    """Save all clustering visualizations to the specified directory."""
    if vis_dir is None:
        vis_dir = os.path.join(RESULTS_DIR, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    # 1. Cluster size distribution
    cluster_sizes = df_with_clusters[cluster_label_column].value_counts().to_dict()
    plot_cluster_size_distribution(
        cluster_sizes,
        save_path=os.path.join(vis_dir, 'cluster_size_distribution.png')
    )
    # 2. Sentiment distribution heatmap
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
    # 3. Silhouette analysis
    plot_silhouette_analysis(
        embeddings,
        df_with_clusters[cluster_label_column],
        save_path=os.path.join(vis_dir, 'silhouette_analysis.png')
    )

def extract_keywords_for_clusters(df_with_clusters, embeddings, cluster_label_column='cluster_label', n_keywords=10):
    """Extract keywords for each cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )

    # Collect texts for each cluster
    cluster_texts = {}
    for cluster_id in df_with_clusters[cluster_label_column].unique():
        cluster_mask = df_with_clusters[cluster_label_column] == cluster_id
        cluster_texts[cluster_id] = df_with_clusters.loc[cluster_mask, TEXT_COLUMN].tolist()

    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id, texts in cluster_texts.items():
        if not texts:  # Skip empty clusters
            continue

        # Compute TF-IDF
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Compute average TF-IDF score for each term
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

        # Get top N keywords
        top_indices = np.argsort(avg_tfidf)[-n_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]

        cluster_keywords[cluster_id] = top_keywords

    return cluster_keywords

if __name__ == '__main__':
    print("Testing evaluation.py...")
    # Create a mock DataFrame containing text, cluster labels, and sentiment labels
    data = {
        TEXT_COLUMN: [
            "great delivery robot, very fast and efficient sdr",  # cluster 0, positive
            "amazing service, the robot is cute and helpful",     # cluster 0, positive
            "sdr technology is the future, love it",              # cluster 0, positive
            "bad experience, the robot got stuck in snow",        # cluster 1, negative
            "terrible sdr, food was cold and delivery late",      # cluster 1, negative
            "this robot is slow and blocks the sidewalk sometimes",  # cluster 1, negative/neutral
            "i dont know what to think about these robots",       # cluster 2, neutral
            "neutral comment about sidewalk delivery robots",     # cluster 2, neutral
            "just a robot doing its job on the pavement",         # cluster 2, neutral
            "noise point one not in any cluster",                 # cluster -1 (noise)
        ],
        'cluster_label': [0, 0, 0, 1, 1, 1, 2, 2, 2, -1],  # Simulated clustering result
        MANUAL_SENTIMENT_COLUMN: [1, 1, 1, 0, 0, 0, 2, 2, 2, 0]  # Simulated manual sentiment labels
        # Add 'dominant_topic': [0, 1, 0, 2, 2, 1, 0, 1, 2, 0] for LDA testing
    }
    test_df = pd.DataFrame(data)

    # 1. Test keyword extraction
    print("\n--- Testing keyword extraction ---")
    keywords = get_top_keywords_for_clusters(test_df[TEXT_COLUMN], test_df['cluster_label'], n_keywords=3)
    # (Actual output depends on TF-IDF results; this just calls the function.)

    # 2. Test sample comment extraction
    print("\n--- Testing sample comment extraction ---")
    samples = get_sample_comments_for_clusters(
        test_df,
        original_text_column=TEXT_COLUMN,
        cluster_label_column='cluster_label',
        n_samples=2
    )
    # (Output prints to console.)

    # 3. Test sentiment distribution analysis
    if MANUAL_SENTIMENT_COLUMN in test_df.columns:
        print("\n--- Testing sentiment distribution analysis ---")
        ct_abs, ct_norm = analyze_sentiment_distribution_in_clusters(
            test_df,
            cluster_label_column='cluster_label',
            sentiment_column=MANUAL_SENTIMENT_COLUMN
        )
        # (Output prints to console.)
    else:
        print(f"\nSkipping sentiment distribution analysis because column '{MANUAL_SENTIMENT_COLUMN}' is not in the DataFrame.")

    # 4. (Optional) Test LDA topic distribution analysis - requires an LDA topic column
    # if LDA_TOPIC_COLUMN in test_df.columns:
    #     print("\n--- Testing LDA topic distribution analysis ---")
    #     ct_lda_abs, ct_lda_norm = analyze_lda_topic_distribution_in_clusters(test_df, cluster_label_column='cluster_label', lda_topic_column=LDA_TOPIC_COLUMN)
    # else:
    #     print(f"\nSkipping LDA topic distribution analysis because '{LDA_TOPIC_COLUMN}' is not in the DataFrame or not defined in config.")

    # 5. Test saving cluster profile files
    print("\n--- Testing saving cluster profile files ---")
    save_cluster_profiles(keywords, samples, base_dir=os.path.join(RESULTS_DIR, "test_cluster_profiles"))
    # (Will create a test_cluster_profiles folder and save files.)

    # 6. Test saving cluster visualizations
    print("\n--- Testing saving clustering visualizations ---")
    save_cluster_visualizations(test_df, test_df[TEXT_COLUMN].values, cluster_label_column='cluster_label')
    # (Will create a visualizations folder and save plots.)

    print("\nevaluation.py testing completed.")
