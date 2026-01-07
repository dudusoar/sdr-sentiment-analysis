# sdr_clustering_analysis/src/text_preprocessor.py
import re
import pandas as pd
# import nltk # Uncomment if using NLTK for more advanced preprocessing
# from nltk.corpus import stopwords # Uncomment if using NLTK stopwords
# from nltk.stem import WordNetLemmatizer # Uncomment if using NLTK lemmatizer
# nltk.download('stopwords', quiet=True) # Uncomment if needed
# nltk.download('wordnet', quiet=True) # Uncomment if needed
# nltk.download('omw-1.4', quiet=True) # Uncomment if needed for wordnet

# --- Basic Cleaning Functions ---
def to_lowercase(text):
    """Converts text to lowercase."""
    return text.lower()

def remove_urls(text):
    """Removes URLs from text."""
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_html_tags(text):
    """Removes HTML tags from text."""
    return re.sub(r'<.*?>', '', text)

def remove_special_characters(text, remove_digits=False):
    """Removes special characters, optionally digits."""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    return re.sub(pattern, '', text)

def normalize_whitespace(text):
    """Normalizes multiple whitespace characters to a single space."""
    return re.sub(r'\s+', ' ', text).strip()

# --- Advanced Cleaning Functions (Optional, often not needed for SBERT) ---
# def remove_stopwords(text, custom_stopwords=None):
#     """Removes stopwords from text."""
#     stop_words = set(stopwords.words('english'))
#     if custom_stopwords:
#         stop_words.update(custom_stopwords)
#     words = text.split()
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     return " ".join(filtered_words)

# def lemmatize_text(text):
#     """Lemmatizes text using WordNetLemmatizer."""
#     lemmatizer = WordNetLemmatizer()
#     words = text.split()
#     lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(lemmatized_words)

# --- Main Preprocessing Function ---
def preprocess_text_for_sbert(text,
                              lowercase=True,
                              remove_url=True,
                              remove_html=True,
                              remove_special_chars=False, # Often SBERT handles special chars well
                              remove_digits_if_removing_specials=False,
                              normalize_space=True
                              # advanced_stopwords=False, # Example for optional advanced steps
                              # advanced_lemmatize=False  # Example for optional advanced steps
                              ):
    """
    Applies a series of preprocessing steps to a single text string,
    optimized for models like Sentence-BERT that prefer less aggressive cleaning.
    """
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = to_lowercase(text)
    if remove_url:
        text = remove_urls(text)
    if remove_html:
        text = remove_html_tags(text)
    if remove_special_chars:
        text = remove_special_characters(text, remove_digits=remove_digits_if_removing_specials)
    # if advanced_stopwords: # Example
    #     text = remove_stopwords(text)
    # if advanced_lemmatize: # Example
    #     text = lemmatize_text(text)
    if normalize_space:
        text = normalize_whitespace(text)
    return text

def preprocess_dataframe(df, text_column, new_column_name='preprocessed_text'):
    """
    Applies preprocessing to a DataFrame text column.
    """
    if text_column not in df.columns:
        print(f"错误: 列 '{text_column}' 在DataFrame中未找到。")
        return df # Or raise an error

    print(f"开始预处理 '{text_column}' 列...")
    df[new_column_name] = df[text_column].apply(preprocess_text_for_sbert)
    print(f"文本预处理完成。新列 '{new_column_name}' 已添加。")
    return df

if __name__ == '__main__':
    print("测试 text_preprocessor.py...")

    sample_texts = [
        "This is a Test comment with URL https://example.com and some <TAGS>!",
        "  Another   one with EXCESSIVE   whitespace and numbers 123.  ",
        "GREAT SDR! AMAZING!!! #SDR #DeliveryRobot",
        None, # Test for non-string input
        "This is a clean sentence."
    ]
    expected_outputs_basic = [ # Assuming default (minimal) preprocessing for SBERT
        "this is a test comment with url and some !", # URLs, HTML removed, lowercase, normalized whitespace
        "another one with excessive whitespace and numbers 123.",
        "great sdr! amazing!!! #sdr #deliveryrobot",
        "",
        "this is a clean sentence."
    ]
    
    # Test with default SBERT-friendly settings
    print("\n--- 测试 SBERT 友好型预处理 (默认设置) ---")
    for i, text in enumerate(sample_texts):
        processed = preprocess_text_for_sbert(text)
        print(f"原始: '{text}'")
        print(f"处理后: '{processed}'")
        # Basic assertion, can be made more rigorous
        # assert processed == expected_outputs_basic[i], f"Test case {i} failed with default settings."

    # Test with more aggressive special character removal
    print("\n--- 测试移除特殊字符的预处理 ---")
    expected_outputs_remove_special = [
        "this is a test comment with url and some tags",
        "another one with excessive whitespace and numbers 123", # Digits kept by default
        "great sdr amazing sdr deliveryrobot",
        "",
        "this is a clean sentence"
    ]
    for i, text in enumerate(sample_texts):
        processed_spec_chars = preprocess_text_for_sbert(text, remove_special_chars=True)
        print(f"原始: '{text}'")
        print(f"处理后 (去特殊字符): '{processed_spec_chars}'")
        # Basic assertion
        # assert processed_spec_chars == expected_outputs_remove_special[i], f"Test case {i} failed with remove_special_chars=True."

    # Test DataFrame preprocessing
    print("\n--- 测试 DataFrame 预处理 ---")
    sample_df = pd.DataFrame({'original_comment': sample_texts})
    processed_df = preprocess_dataframe(sample_df, text_column='original_comment')
    print(processed_df[['original_comment', 'preprocessed_text']].head())

    print("\ntext_preprocessor.py 测试完成。")