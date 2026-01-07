# -*- coding: utf-8 -*-
import os
from gensim.models import LdaModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def check_model_files(model_base_path):
    """
    Check if all required model files exist
    :param model_base_path: Model base path (without extension)
    :return: bool, whether all files exist
    """
    required_files = [
        f"{model_base_path}",
        f"{model_base_path}.id2word",
        f"{model_base_path}.expElogbeta.npy",
        f"{model_base_path}.state"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required model files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    
    print("All required model files exist.")
    return True

def analyze_topic_keywords(model, topic_id, num_words=20):
    """
    Analyze keywords for a single topic
    """
    topic_words = model.show_topic(topic_id, topn=num_words)
    print(f"\nKeywords and weights for topic #{topic_id + 1}:")
    print("-" * 50)
    for word, weight in topic_words:
        print(f"{word:15} {weight:.4f}")
    return topic_words

def plot_topic_keywords(topic_words, topic_id, output_dir):
    """
    Plot topic keyword distribution
    """
    words, weights = zip(*topic_words)
    plt.figure(figsize=(12, 6))
    plt.bar(words, weights)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Keyword distribution for topic #{topic_id + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'topic_{topic_id + 1}_keywords.png'))
    plt.close()

def plot_topic_heatmap(model, output_dir):
    """
    Plot topic-word heatmap
    """
    # Get top 10 keywords for all topics
    num_topics = model.num_topics
    top_words = 10
    topic_words = []
    for i in range(num_topics):
        topic_words.extend([word for word, _ in model.show_topic(i, topn=top_words)])
    
    # Remove duplicates
    unique_words = list(set(topic_words))
    
    # Create topic-word matrix
    topic_word_matrix = np.zeros((num_topics, len(unique_words)))
    for i in range(num_topics):
        for word, weight in model.show_topic(i, topn=top_words):
            if word in unique_words:
                topic_word_matrix[i, unique_words.index(word)] = weight
    
    # Plot heatmap
    plt.figure(figsize=(15, 8))
    plt.imshow(topic_word_matrix, cmap='YlOrRd')
    plt.colorbar(label='Weight')
    plt.title('Topic-Word Distribution Heatmap')
    plt.xlabel('Keywords')
    plt.ylabel('Topics')
    plt.xticks(range(len(unique_words)), unique_words, rotation=45, ha='right')
    plt.yticks(range(num_topics), [f'Topic {i+1}' for i in range(num_topics)])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_word_heatmap.png'))
    plt.close()

def analyze_all_topics(model_path, num_words=20):
    """
    Analyze all topics in the model
    :param model_path: Model file path
    :param num_words: Number of keywords to display per topic
    """
    # Check model files
    model_base_path = model_path.replace('.gensim', '')
    if not check_model_files(model_base_path):
        print("Cannot continue analysis, please ensure all required model files exist.")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    model = LdaModel.load(model_path)
    
    # Create output directory
    output_dir = "topic_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze each topic
    for topic_id in range(model.num_topics):
        # Analyze topic keywords
        topic_words = analyze_topic_keywords(model, topic_id, num_words)
        
        # Plot keyword distribution
        plot_topic_keywords(topic_words, topic_id, output_dir)
        print(f"Visualization results for topic #{topic_id + 1} saved to: {output_dir}/topic_{topic_id + 1}_keywords.png")
        print("-" * 50)
    
    # Plot topic-word heatmap
    plot_topic_heatmap(model, output_dir)
    print(f"\nTopic-word heatmap saved to: {output_dir}/topic_word_heatmap.png")

if __name__ == "__main__":
    # Model path
    model_path = "topic_model_results/models/lda_model_10.gensim"
    
    # Analyze all topics
    analyze_all_topics(model_path) 