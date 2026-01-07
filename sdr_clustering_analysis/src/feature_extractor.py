# sdr_clustering_analysis/src/feature_extractor.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from config import SBERT_MODEL_NAME, EMBEDDINGS_FILE, PREPROCESSED_COMMENTS_FILE
from src.utils import save_pickle, load_pickle, get_device_for_sbert  # Assuming get_device_for_sbert is in utils

class SentenceEmbedder:
    def __init__(self, model_name=SBERT_MODEL_NAME, device=None):
        """
        Initialize the sentence embedder.
        Args:
            model_name (str): Name of the SentenceTransformer model on Hugging Face.
            device (str, optional): 'cuda' or 'cpu'. If None, auto-detect.
        """
        if device is None:
            self.device = get_device_for_sbert()  # Using the utility from utils.py
        else:
            self.device = device

        print(f"Loading SentenceTransformer model: {model_name} on device: {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {model_name}. Error: {e}")
            print("Please ensure the sentence-transformers library is installed "
                  "(pip install sentence-transformers) and the model name is correct.")
            self.model = None

    def encode_sentences(self, sentences, batch_size=32, show_progress_bar=True):
        """
        Encode a list of sentences into embedding vectors.
        Args:
            sentences (list of str): List of sentences to be encoded.
            batch_size (int): Batch size during encoding.
            show_progress_bar (bool): Whether to show the encoding progress bar.
        Returns:
            numpy.ndarray: Array of sentence embeddings; returns None if the model is not loaded.
        """
        if self.model is None:
            print("Error: SentenceTransformer model was not loaded successfully. Encoding aborted.")
            return None

        if (not sentences or
            not isinstance(sentences, (list, pd.Series)) or
            not all(isinstance(s, str) for s in sentences)):
            print("Error: Input sentences must be a non-empty list or Pandas Series of strings.")
            return np.array([])  # Return empty array for consistency

        print(f"Start encoding {len(sentences)} sentences (batch size: {batch_size})...")
        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            print("Sentence encoding completed.")
            return embeddings
        except Exception as e:
            print(f"An error occurred during sentence encoding: {e}")
            return None


def get_sentence_embeddings(texts_series,
                            use_cache=True,
                            cache_filepath=EMBEDDINGS_FILE,
                            sbert_model_name=SBERT_MODEL_NAME):
    """
    Obtain sentence embeddings for a list of texts, with optional caching.
    Args:
        texts_series (pd.Series): Pandas Series containing texts to be encoded.
        use_cache (bool): Whether to attempt loading/saving embeddings from/to cache.
        cache_filepath (str): Path to the embedding cache file.
        sbert_model_name (str): Name of the SentenceTransformer model.
    Returns:
        numpy.ndarray: Array of sentence embeddings.
    """
    if use_cache and os.path.exists(cache_filepath):
        print(f"Attempting to load sentence embeddings from cache: {cache_filepath}")
        embeddings = load_pickle(cache_filepath)
        if embeddings is not None and len(embeddings) == len(texts_series):
            print("Sentence embeddings successfully loaded from cache.")
            return embeddings
        else:
            print("Cache file is invalid or does not match the current data size. Recomputing embeddings.")

    embedder = SentenceEmbedder(model_name=sbert_model_name)
    if embedder.model is None:
        return None  # Model loading failed

    # Convert Series to list of strings for the encoder
    texts_list = texts_series.tolist()
    embeddings = embedder.encode_sentences(texts_list)

    if embeddings is not None and use_cache:
        save_pickle(embeddings, cache_filepath)

    return embeddings


if __name__ == '__main__':
    print("Testing feature_extractor.py...")

    # Prepare some sample texts (simulating outputs from data_loader and text_preprocessor)
    sample_preprocessed_texts = pd.Series([
        "this is a test comment with url and some tags",
        "another one with excessive whitespace and numbers 123",
        "great sdr amazing sdr deliveryrobot",
        "this is a clean sentence for embedding",
        "yet another comment to be embedded"
    ])

    print(f"\nSBERT model in use: {SBERT_MODEL_NAME}")

    # Test embedding generation (first run without cache)
    print("\n--- Testing first-time embedding generation (no cache or cache missing) ---")
    # Remove any existing cache file for a clean test
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)
        print(f"Removed old cache file: {EMBEDDINGS_FILE}")

    embeddings1 = get_sentence_embeddings(sample_preprocessed_texts, use_cache=True)  # Will save to cache
    if embeddings1 is not None:
        print(f"Embeddings obtained. Shape: {embeddings1.shape}")
        assert embeddings1.shape == (
            len(sample_preprocessed_texts),
            SentenceEmbedder(SBERT_MODEL_NAME).model.get_sentence_embedding_dimension()
        )
        print("Embedding dimensions match the model output dimension.")

    # Test loading embeddings from cache
    if embeddings1 is not None:  # Only test caching if first attempt was successful
        print("\n--- Testing embedding loading from cache ---")
        embeddings2 = get_sentence_embeddings(sample_preprocessed_texts, use_cache=True)
        if embeddings2 is not None:
            assert np.array_equal(embeddings1, embeddings2), \
                "Cached embeddings do not match the originally computed embeddings."
            print("Cached embeddings are identical to the originally computed embeddings.")

    print("\nfeature_extractor.py testing completed.")
