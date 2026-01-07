# coding: utf-8
'''
filename: visualization.py
function: Results visualization, word frequency statistics, and word cloud generation
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
    Calculate word frequency and save results
    
    Args:
        docs: List of word lists
        file_name: File name to save
        threshold: Frequency threshold, only keep words with frequency greater than threshold
    
    Returns:
        word_frequency_df: DataFrame containing word frequency statistics
    '''
    ensure_directory(WORD_FREQ_DIR)
    file_path = os.path.join(WORD_FREQ_DIR, file_name)
    
    # Count word frequency
    words_list = [word for row in docs for word in row]
    word_counts = Counter(words_list)
    word_counts_dict = dict(word_counts)
    sorted_word_counts = sorted(word_counts_dict.items(), key=lambda x: x[1], reverse=True)

    # Convert to dataframe
    word_frequency_df = pd.DataFrame(sorted_word_counts, columns=['word', 'frequency'])
    
    # Filter words with frequency greater than threshold
    index = word_frequency_df['frequency'] > threshold
    word_frequency_df = word_frequency_df[index]
    
    # Save results
    save_excel(word_frequency_df, file_path)
    print(f'Word frequency statistics saved to: {file_path}')

    return word_frequency_df

def generate_word_cloud(words, picture_name):
    '''
    Generate word cloud and save
    
    Args:
        words: Word frequency dictionary, keys are words, values are frequencies
        picture_name: Picture name to save
    
    Returns:
        None
    '''
    ensure_directory(WORD_FREQ_DIR)
    picture_path = os.path.join(WORD_FREQ_DIR, picture_name)
    
    # Create WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    # Generate word cloud based on word frequency
    wordcloud.generate_from_frequencies(frequencies=words)
    # Save word cloud
    wordcloud.to_file(picture_path)
    print(f'Word cloud image saved to: {picture_path}')

    # Display word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(picture_name.replace('.png', ''))
    plt.show()

def analyze_word_frequency(comments, output_dir=WORD_FREQ_DIR):
    '''
    Analyze word frequency and generate word clouds
    
    Args:
        comments: DataFrame containing comment data
        output_dir: Output directory
    
    Returns:
        None
    '''
    ensure_directory(output_dir)
    
    # Overall word frequency statistics
    word_frequency_df = word_count(comments['f_word_list'], file_name='word_frequency.xlsx', threshold=0)
    
    # Save words with frequency > 1
    word_frequency_df_small = word_count(comments['f_word_list'], file_name='word_frequency_small.xlsx', threshold=1)
    
    # Count word frequency by label
    for label in [0, 1, 2]:
        index = comments['label1'] == label
        word_frequency_df_label = word_count(
            comments['f_word_list'][index], 
            file_name=f'word_frequency_{label}.xlsx', 
            threshold=0
        )
    
    # Generate word clouds
    # Overall word frequency word cloud
    words = word_frequency_df.set_index('word').to_dict()['frequency']
    generate_word_cloud(words, picture_name='wordcloud.png')
    
    # Word cloud for words with frequency > 1
    words_small = word_frequency_df_small.set_index('word').to_dict()['frequency']
    generate_word_cloud(words_small, picture_name='wordcloud_small.png')
    
    # Generate word clouds by label
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
    Plot model performance comparison
    
    Args:
        results: List containing evaluation results for each model
        model_names: List of model names
        metrics: List of metrics to plot, default is ['accuracy', 'precision', 'recall', 'f1-score']
    
    Returns:
        None
    '''
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Extract results
    metric_values = {metric: [] for metric in metrics}
    
    for result in results:
        for metric in metrics:
            if metric == 'accuracy':
                metric_values[metric].append(result['accuracy'])
            else:
                # Calculate average for each class
                values = [result[str(label)][metric] for label in [0, 1, 2] if str(label) in result]
                metric_values[metric].append(np.mean(values))
    
    # Plot
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].bar(model_names, metric_values[metric])
        axes[i].set_title(f'Average {metric.capitalize()}')
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel(metric.capitalize())
        
        # Add value labels
        for j, v in enumerate(metric_values[metric]):
            axes[i].text(j, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.show()