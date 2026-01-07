# sdr_clustering_analysis/main.py
import pandas as pd
import os
import argparse
import time

# Internal project module imports
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
    Execute the complete clustering analysis pipeline.
    """
    start_time = time.time()
    print("Starting SDR comment clustering analysis pipeline...")
    create_project_directories() # Ensure result directories exist

    # 1. Load data
    print("\n--- Step 1: Load raw data ---")
    raw_df = load_raw_comments(filepath=raw_file_path)
    if raw_df is None or raw_df.empty:
        print("Data loading failed or data is empty, terminating pipeline.")
        return

    # Ensure the ID column used later is 'unique_comment_id' (guaranteed by data_loader)
    current_id_col = 'unique_comment_id'


    # 2. (Optional) Text preprocessing
    # For SBERT, complex preprocessing is usually not needed, but can be done here if required
    texts_to_embed = raw_df[text_col] # Default to original text column

    if use_preprocessing:
        print(f"\n--- Step 2: Text preprocessing (use_preprocessing=True) ---")
        if use_embedding_cache and os.path.exists(preprocessed_cache_path):
            print(f"Attempting to load preprocessed text from cache: {preprocessed_cache_path}")
            processed_df = load_pickle(preprocessed_cache_path)
            if processed_df is not None and 'preprocessed_text' in processed_df.columns and \
               len(processed_df) == len(raw_df):
                texts_to_embed = processed_df['preprocessed_text']
                print("Successfully loaded preprocessed text from cache.")
            else:
                print("Preprocessed text cache invalid, will reprocess.")
                processed_df = preprocess_dataframe(raw_df.copy(), text_column=text_col, new_column_name='preprocessed_text')
                texts_to_embed = processed_df['preprocessed_text']
                save_pickle(processed_df[['preprocessed_text']], preprocessed_cache_path) # Save only relevant part
        else:
            processed_df = preprocess_dataframe(raw_df.copy(), text_column=text_col, new_column_name='preprocessed_text')
            texts_to_embed = processed_df['preprocessed_text']
            if use_embedding_cache: # implies saving preprocessed if embeddings will be cached
                 save_pickle(processed_df[['preprocessed_text']], preprocessed_cache_path)
    else:
        print(f"\n--- Step 2: Text preprocessing skipped (use_preprocessing=False) ---")
        # texts_to_embed is already raw_df[text_col]

    if texts_to_embed.empty:
        print("Error: Preprocessed/selected text data is empty, terminating pipeline.")
        return

    # 3. Feature extraction (Sentence Embeddings)
    print("\n--- Step 3: Feature extraction (Sentence Embeddings) ---")
    embeddings = get_sentence_embeddings(
        texts_series=texts_to_embed,
        use_cache=use_embedding_cache,
        cache_filepath=embedding_cache_path,
        sbert_model_name=sbert_model
    )
    if embeddings is None or embeddings.shape[0] == 0:
        print("Sentence embedding extraction failed or returned empty, terminating pipeline.")
        return
    if embeddings.shape[0] != len(raw_df):
        print(f"Error: Number of embedding vectors ({embeddings.shape[0]}) does not match number of original comments ({len(raw_df)}).")
        print("This may be caused by errors in text preprocessing or embedding process. Terminating pipeline.")
        return

    # 4. Clustering
    print(f"\n--- Step 4: Run {clustering_algorithm.upper()} clustering ---")
    cluster_analyzer = Clusterer()
    cluster_labels = None
    model_details = None # To store the trained clustering model if needed

    if clustering_algorithm.lower() == 'kmeans':
        if force_k:
            print(f"Running K-Means with a specified K: K={force_k}")
            selected_k = force_k
        else:
            print(f"Searching for the best K for K-Means based on evaluation metrics (range: {list(kmeans_k_range)})")
            k_eval_df = cluster_analyzer.find_optimal_k_kmeans(embeddings, k_range=kmeans_k_range)
            print("\nK evaluation (Inertia & Silhouette Score):")
            print(k_eval_df)
            # Simple K selection logic: choose the K with the highest silhouette score
            if not k_eval_df.empty:
                # Ensure Silhouette_Score is numeric and handle potential NaNs by filling with a very low value
                k_eval_df['Silhouette_Score_numeric'] = pd.to_numeric(k_eval_df['Silhouette_Score'], errors='coerce').fillna(-2)
                best_k_row = k_eval_df.loc[k_eval_df['Silhouette_Score_numeric'].idxmax()]
                selected_k = int(best_k_row['K'])
                print(f"Automatically selected K based on the highest silhouette score ({best_k_row['Silhouette_Score_numeric']:.4f}): {selected_k}")
            else:
                print(f"Unable to automatically select K; using the first value in the range: {min(kmeans_k_range)}")
                selected_k = min(kmeans_k_range)

        model_details, cluster_labels = cluster_analyzer.kmeans_cluster(embeddings, n_clusters=selected_k)

    # elif clustering_algorithm.lower() == 'hdbscan': # Uncomment and implement if using HDBSCAN
    #     model_details, cluster_labels = cluster_analyzer.hdbscan_cluster(
    #         embeddings,
    #         min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    #         min_samples=HDBSCAN_MIN_SAMPLES
    #     )
    else:
        print(f"Error: Unsupported clustering algorithm '{clustering_algorithm}'. Please choose 'kmeans'.") # Add hdbscan when ready
        return

    if cluster_labels is None:
        print("Clustering failed. Terminating pipeline.")
        return

    # Add cluster labels to the original DataFrame
    # Ensuring correct alignment if any rows were dropped during preprocessing/embedding
    results_df = raw_df.copy() # Start with a fresh copy of the original df
    results_df['cluster_label'] = cluster_labels

    # Save cluster assignment results
    if current_id_col in results_df.columns and text_col in results_df.columns:
        save_csv(results_df[[current_id_col, text_col, 'cluster_label']], CLUSTER_ASSIGNMENT_FILE)
    else: # Fallback if columns are missing, save what's available
        save_csv(results_df[['cluster_label']], CLUSTER_ASSIGNMENT_FILE)


    # 5. Clustering evaluation and analysis
    print("\n--- Step 5: Clustering evaluation and analysis ---")
    # 5a. Internal evaluation metrics
    print("\n--- 5a. Internal clustering evaluation metrics ---")
    clustering_metrics = evaluate_clustering(embeddings, cluster_labels)
    with open(CLUSTER_EVALUATION_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Clustering algorithm: {clustering_algorithm.upper()}\n")
        if clustering_algorithm.lower() == 'kmeans':
            f.write(f"Selected K: {selected_k}\n")
        f.write("\nEvaluation metrics:\n")
        for metric, value in clustering_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    print(f"Clustering evaluation metrics saved to: {CLUSTER_EVALUATION_FILE}")

    # 5b. Cluster profiling: keywords and sample comments
    print("\n--- 5b. Generate cluster profile files ---")
    # Use raw text or preprocessed text for keyword extraction depending on what is more meaningful.
    # SBERT is typically based on raw text, but TF-IDF keywords may benefit from lightly cleaned text.
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

    # 5c. Cross analysis between clusters and manual sentiment labels
    if sentiment_col and sentiment_col in results_df.columns:
        print(f"\n--- 5c. Analyze relationship between clusters and '{sentiment_col}' (manual sentiment labels) ---")
        senti_ct_abs, senti_ct_norm = analyze_sentiment_distribution_in_clusters(
            df_with_clusters=results_df,
            cluster_label_column='cluster_label',
            sentiment_column=sentiment_col
        )
        save_csv(senti_ct_abs, CLUSTER_SENTIMENT_CROSSTAB_FILE.replace(".csv", "_absolute.csv"), index=True)
        save_csv(senti_ct_norm, CLUSTER_SENTIMENT_CROSSTAB_FILE.replace(".csv", "_normalized.csv"), index=True)
    else:
        print(f"\nNote: No valid manual sentiment label column provided ('{sentiment_col}'). Skipping cluster–sentiment crosstab analysis.")

    # 5d. (Optional) Cross analysis between clusters and LDA topics
    # if lda_topic_col and lda_topic_col in results_df.columns:
    #     print(f"\n--- 5d. Analyze relationship between clusters and '{lda_topic_col}' (LDA topics) ---")
    #     lda_ct_abs, lda_ct_norm = analyze_lda_topic_distribution_in_clusters(
    #         df_with_clusters=results_df,
    #         cluster_label_column='cluster_label',
    #         lda_topic_column=lda_topic_col
    #     )
    #     save_csv(lda_ct_abs, CLUSTER_LDA_CROSSTAB_FILE.replace(".csv", "_absolute.csv"), index=True)
    #     save_csv(lda_ct_norm, CLUSTER_LDA_CROSSTAB_FILE.replace(".csv", "_normalized.csv"), index=True)
    # else:
    #     print(f"\nNote: No valid LDA topic column provided ('{lda_topic_col}'). Skipping cluster–LDA crosstab analysis.")


    end_time = time.time()
    print(f"\nSDR comment clustering analysis pipeline completed. Total runtime: {end_time - start_time:.2f} seconds.")
    print(f"All results have been saved to: {os.path.abspath(RESULTS_DIR)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SDR YouTube Comment Clustering Analysis Tool")
    parser.add_argument(
        '--raw_file', type=str, default=RAW_COMMENTS_FILE,
        help="Path to the raw comment data file (.xlsx or .csv)"
    )
    parser.add_argument(
        '--text_column', type=str, default=TEXT_COLUMN,
        help="Column name containing the comment text"
    )
    parser.add_argument(
        '--id_column', type=str, default=ID_COLUMN,
        help="Column name containing the comment ID (optional)"
    )
    parser.add_argument(
        '--sentiment_column', type=str, default=MANUAL_SENTIMENT_COLUMN,
        help="Column name containing manual sentiment labels (optional, for comparison analysis)"
    )
    # parser.add_argument( # Uncomment if you implement LDA topic integration
    #     '--lda_topic_column', type=str, default=None, # Example: config.LDA_TOPIC_COLUMN
    #     help="Column name containing LDA topic labels (optional, for comparison analysis)"
    # )
    parser.add_argument(
        '--use_preprocessing', action='store_true',
        help="Whether to preprocess the text (default: False; SBERT typically does not require heavy preprocessing)"
    )
    parser.add_argument(
        '--sbert_model', type=str, default=SBERT_MODEL_NAME,
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        '--no_embedding_cache', action='store_false', dest='use_embedding_cache',
        help="Do not use or save sentence embedding cache"
    )
    parser.add_argument(
        '--algorithm', type=str, default='kmeans', choices=['kmeans'], # Add 'hdbscan' when ready
        help="Clustering algorithm to use"
    )
    parser.add_argument(
        '--k', type=int, default=KMEANS_SELECTED_K, # Default from config, could be None
        help="Specify K for K-Means. If not provided, the program will attempt to automatically find the best K."
    )
    parser.add_argument(
        '--k_min', type=int, default=min(KMEANS_N_CLUSTERS_RANGE) if KMEANS_N_CLUSTERS_RANGE else 2,
        help="Minimum K when searching automatically (only used when --k is not specified)"
    )
    parser.add_argument(
        '--k_max', type=int, default=max(KMEANS_N_CLUSTERS_RANGE) if KMEANS_N_CLUSTERS_RANGE else 7,
        help="Maximum K when searching automatically (only used when --k is not specified)"
    )

    args = parser.parse_args()

    # Update KMeans K range if provided through CLI and no fixed K is specified
    k_range_to_use = range(args.k_min, args.k_max + 1)
    if args.k:  # User specified a K
        force_k_value = args.k
    else:  # User did not specify a K, so we search within the range
        force_k_value = None  # This will trigger find_optimal_k_kmeans
        print(f"Will search K in range: {list(k_range_to_use)}")
        # Update config-like variable for the pipeline based on CLI args
        # This is a bit of a workaround as config isn't dynamically updated
        # but the pipeline function takes these as direct args.

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
