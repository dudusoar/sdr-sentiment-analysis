# -*- coding: utf-8 -*-
# Import required packages

# --- Data Processing ---
import pandas as pd
import numpy as np

# --- Language Processing ---
from nltk.corpus import stopwords # Load stopwords
from nltk.stem import WordNetLemmatizer # Lemmatization
from pattern.en import lemma # Lemmatization
from nltk.tokenize import word_tokenize # Tokenization
import string
import re
import os

# --- Topic Modeling ---
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel # Calculate coherence score
import gensim

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Global Variables and Constants ---

# Contraction replacements
CONTRACTIONS = {
    "didn't": "did not", "don't": "do not", "doesn't": "does not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "mightn't": "might not", "mustn't": "must not", "we're": "we are",
    "you're": "you are", "they're": "they are", "I'm": "I am", "he's": "he is",
    "she's": "she is", "it's": "it is", "we've": "we have", "you've": "you have",
    "they've": "they have", "I've": "I have", "kinda": "kind of",
    "sorta": "sort of", "wanna": "want to", "gonna": "going to",
    "gotta": "got to", "hafta": "have to", "needa": "need to",
    "outta": "out of", "lemme": "let me", "gimme": "give me",
    "pavement": "sidewalk", "used": "use",
}

# Stop words
STOP_WORDS = set(stopwords.words('english')).copy() | set(["'s", "n't", 'lol', "'m", "'re", "'d", "'ve", 'would', 'could', 'gonna'])

# Punctuation
PUNCTUATION = set(string.punctuation) | set(['...', '``', "''", '’', '..', '....', '.....', '“', '”'])
PUNCTUATION_REMOVED = PUNCTUATION - set()

# Additional words to exclude
ADDITIONAL_WORDS = set([
    'robots', 'robot', 'bot', 'sidewalk', 'deliver', 'delivery', 'human',
    'people', 'person', 'someone', 'everyone', 'one', 'thing', 'u', 'think',
    'get', 'like', 'going', 'go', 'make', 'see', 'door', 'dash', 'gp', 'ye',
    'wow', 'ha', 'haha', 'well', 'yeah', 'oh', 'also', 'really', 'still',
    'actually', 'probably', 'eventually', 'already', 'much', 'more', 'many',
    'even',
])

# --- Function Definitions ---

def word_replace(text):
    """Replace contractions"""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

def text_preprocessing(text, filtering=True, lemmatization=True):
    """
    Text preprocessing
    :param filtering: Whether to remove stopwords
    :param lemmatization: Whether to perform lemmatization
    """
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = word_replace(text)
    words = word_tokenize(text)
    words = [word for word in words if not re.match(r'\d+', word)]
    words = [w for w in words if w not in PUNCTUATION_REMOVED]
    if filtering:
        words = [w for w in words if w not in STOP_WORDS]
    if lemmatization:
        words = [lemma(w) for w in words]
    return words

def prepare_corpus(filename, additional_words):
    """
    Load and preprocess data, create dictionary and corpus.
    :param filename: Data file name (Excel)
    :param additional_words: Additional words to exclude from topic modeling
    :return: (corpus, dictionary, data_topic)
    """
    comments = pd.read_excel(filename)
    comments['processed_text'] = comments['pure_text'].apply(text_preprocessing)
    mask = comments['processed_text'].apply(lambda x: x != [])
    comments = comments[mask].reset_index(drop=True)

    data_topic = comments['processed_text'].apply(lambda x: [word for word in x if word not in additional_words]).tolist()
    data_topic = [text for text in data_topic if text]

    dictionary = Dictionary(data_topic)
    corpus = [dictionary.doc2bow(text) for text in data_topic]

    return corpus, dictionary, data_topic

def plot_keywords(lda_model, num_topics, output_dir, num_words=10, cmap='tab10'):
    """Plot and save keyword distribution"""
    colors = plt.cm.get_cmap(cmap, num_topics)
    nrows = (num_topics + 1) // 2 if num_topics > 1 else 1
    ncols = 2 if num_topics > 1 else 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, nrows * 5), sharey=True)
    axs = np.ravel(axs) # 展平数组以便于索引

    for i in range(num_topics):
        topic_keywords = lda_model.show_topic(i, topn=num_words)
        words, importance = zip(*topic_keywords)
        axs[i].bar(words, importance, color=colors(i))
        axs[i].set_title(f'Topic {i+1}', fontweight='bold', fontsize=14)
        axs[i].set_ylabel('Importance', fontsize=14)
        axs[i].tick_params(axis='x', labelsize=14, rotation=45)
        axs[i].tick_params(axis='y', labelsize=12)

    # 隐藏多余的子图
    for i in range(num_topics, len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout(pad=5.0)
    plt.savefig(os.path.join(output_dir, f'topic_keywords_distribution_{num_topics}.png'))
    plt.close(fig) # Close figure to free memory

def analyze_specific_topics(corpus, dictionary, topic_numbers, output_dir):
    """
    Train models for specified number of topics and perform in-depth analysis.
    :param topic_numbers: List containing topic numbers to analyze
    :param output_dir: Directory to save result plots
    """
    print("--- In-depth analysis for specific topic numbers ---")
    
    # Create models subdirectory
    models_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created model save directory: {models_dir}")
    
    for num in topic_numbers:
        print(f"Analyzing {num} topics...")
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num, passes=20, chunksize=2000, iterations=600, alpha='asymmetric', eta='auto')
        
        # Save model to models subdirectory
        model_path = os.path.join(models_dir, f'lda_model_{num}.gensim')
        lda_model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        for i in range(num):
            print(f"Keywords for topic #{i+1} (model: {num} topics):")
            print(lda_model.show_topic(i))
        print('-'*50)
        plot_keywords(lda_model, num, output_dir)

def find_optimal_topics(corpus, dictionary, data_topic, start=1, end=21):
    """
    Train LDA models for a range of topic numbers and compute evaluation metrics.
    :return: (coherence_values, perplexity_values)
    """
    print("\n--- Finding optimal number of topics ---")
    num_topics_list = range(start, end)
    coherence_values = []
    perplexity_values = []

    for num_topics in num_topics_list:
        print(f"Computing for {num_topics} topics...")
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=20, chunksize=2000, iterations=600, alpha='asymmetric', eta='auto')
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_topic, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model_lda.get_coherence())
        log_perplexity = lda_model.log_perplexity(corpus)
        perplexity_values.append(2**(-log_perplexity))

    return list(num_topics_list), coherence_values, perplexity_values

def plot_evaluation_metrics(num_topics_list, coherence_values, perplexity_values, output_dir):
    """Plot coherence and perplexity scores"""
    # Coherence score plot
    plt.figure(figsize=(10, 5))
    plt.plot(num_topics_list, coherence_values)
    plt.xticks(np.arange(min(num_topics_list), max(num_topics_list)+1, 1.0))
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title('Coherence score vs. number of topics')
    plt.savefig(os.path.join(output_dir, 'coherence_score.png'))
    plt.close()

    # Perplexity score plot
    plt.figure(figsize=(10, 5))
    plt.plot(num_topics_list, perplexity_values)
    plt.xticks(np.arange(min(num_topics_list), max(num_topics_list)+1, 1.0))
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title('Perplexity vs. number of topics')
    plt.savefig(os.path.join(output_dir, 'perplexity_score.png'))
    plt.close()

def main(start_topics=1, end_topics=21):
    """
    Main execution module
    :param start_topics: Starting number of topics for analysis
    :param end_topics: Ending number of topics for analysis (exclusive)
    """
    filename = "data/comments.xlsx"
    output_dir = "topic_model_results"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Prepare data and corpus
    corpus, dictionary, data_topic = prepare_corpus(filename, ADDITIONAL_WORDS)

    # Find optimal number of topics
    num_topics_range, coherence_scores, perplexity_scores = find_optimal_topics(
        corpus, dictionary, data_topic, 
        start=start_topics, 
        end=end_topics
    )

    # Plot evaluation metrics
    plot_evaluation_metrics(num_topics_range, coherence_scores, perplexity_scores, output_dir)
    
    # Train and save models for each topic number
    analyze_specific_topics(corpus, dictionary, num_topics_range, output_dir)
    
    print(f"\nAnalysis complete, result plots saved to '{output_dir}' directory.")

if __name__ == '__main__':
    # Ensure necessary NLTK data is downloaded
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

    # You can adjust the topic number range by modifying the parameters here.
    main(start_topics=1, end_topics=21)
