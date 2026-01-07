# sdr_clustering_analysis/src/feature_extractor.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from config import SBERT_MODEL_NAME, EMBEDDINGS_FILE, PREPROCESSED_COMMENTS_FILE
from src.utils import save_pickle, load_pickle, get_device_for_sbert # Assuming get_device_for_sbert is in utils

class SentenceEmbedder:
    def __init__(self, model_name=SBERT_MODEL_NAME, device=None):
        """
        初始化句子嵌入器。
        Args:
            model_name (str): Hugging Face上SentenceTransformer的模型名称。
            device (str, optional): 'cuda' 或 'cpu'。如果为None，则自动检测。
        """
        if device is None:
            self.device = get_device_for_sbert() # Using the utility from utils.py
        else:
            self.device = device
        
        print(f"正在加载SentenceTransformer模型: {model_name} 到设备: {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"模型 {model_name} 加载成功。")
        except Exception as e:
            print(f"加载SentenceTransformer模型失败: {model_name}. 错误: {e}")
            print("请确保已安装sentence-transformers库 (pip install sentence-transformers) 并且模型名称正确。")
            self.model = None

    def encode_sentences(self, sentences, batch_size=32, show_progress_bar=True):
        """
        将句子列表编码为嵌入向量。
        Args:
            sentences (list of str): 需要编码的句子列表。
            batch_size (int):编码时的批处理大小。
            show_progress_bar (bool): 是否显示编码进度条。
        Returns:
            numpy.ndarray: 句子的嵌入向量数组，如果模型未加载则返回None。
        """
        if self.model is None:
            print("错误: SentenceTransformer模型未成功加载。无法进行编码。")
            return None
        
        if not sentences or not isinstance(sentences, (list, pd.Series)) or not all(isinstance(s, str) for s in sentences):
            print("错误: 输入的句子应为非空字符串列表或Pandas Series。")
            return np.array([]) # Return empty array for consistency

        print(f"开始对 {len(sentences)} 条句子进行编码 (批大小: {batch_size})...")
        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            print("句子编码完成。")
            return embeddings
        except Exception as e:
            print(f"句子编码过程中发生错误: {e}")
            return None

def get_sentence_embeddings(texts_series,
                            use_cache=True,
                            cache_filepath=EMBEDDINGS_FILE,
                            sbert_model_name=SBERT_MODEL_NAME):
    """
    获取文本列表的句子嵌入，可选择使用缓存。
    Args:
        texts_series (pd.Series): 包含待编码文本的Pandas Series。
        use_cache (bool): 是否尝试从缓存加载/保存嵌入。
        cache_filepath (str): 嵌入缓存文件的路径。
        sbert_model_name (str): SentenceTransformer模型名称。
    Returns:
        numpy.ndarray: 句子的嵌入向量数组。
    """
    if use_cache and os.path.exists(cache_filepath):
        print(f"尝试从缓存文件加载句子嵌入: {cache_filepath}")
        embeddings = load_pickle(cache_filepath)
        if embeddings is not None and len(embeddings) == len(texts_series):
            print("成功从缓存加载句子嵌入。")
            return embeddings
        else:
            print("缓存文件无效或与当前数据量不匹配，将重新计算嵌入。")

    embedder = SentenceEmbedder(model_name=sbert_model_name)
    if embedder.model is None:
        return None # Model loading failed

    # Convert Series to list of strings for encoder
    texts_list = texts_series.tolist()
    embeddings = embedder.encode_sentences(texts_list)

    if embeddings is not None and use_cache:
        save_pickle(embeddings, cache_filepath)
    
    return embeddings

if __name__ == '__main__':
    print("测试 feature_extractor.py...")

    # 准备一些示例文本 (模拟从data_loader和text_preprocessor来的输出)
    sample_preprocessed_texts = pd.Series([
        "this is a test comment with url and some tags",
        "another one with excessive whitespace and numbers 123",
        "great sdr amazing sdr deliveryrobot",
        "this is a clean sentence for embedding",
        "yet another comment to be embedded"
    ])

    print(f"\n使用的SBERT模型: {SBERT_MODEL_NAME}")
    
    # 测试获取嵌入 (不使用缓存首次运行)
    print("\n--- 测试首次获取嵌入 (不使用缓存或缓存不存在) ---")
    # 清理可能的旧缓存文件以进行干净测试
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)
        print(f"已删除旧的缓存文件: {EMBEDDINGS_FILE}")

    embeddings1 = get_sentence_embeddings(sample_preprocessed_texts, use_cache=True) # 会保存到缓存
    if embeddings1 is not None:
        print(f"获取到嵌入向量，形状: {embeddings1.shape}")
        assert embeddings1.shape == (len(sample_preprocessed_texts), SentenceEmbedder(SBERT_MODEL_NAME).model.get_sentence_embedding_dimension())
        print("嵌入维度与模型输出维度一致。")

    # 测试从缓存加载嵌入
    if embeddings1 is not None: # Only test caching if first attempt was successful
        print("\n--- 测试从缓存获取嵌入 ---")
        embeddings2 = get_sentence_embeddings(sample_preprocessed_texts, use_cache=True)
        if embeddings2 is not None:
            assert np.array_equal(embeddings1, embeddings2), "从缓存加载的嵌入与首次计算的不一致。"
            print("从缓存加载的嵌入与首次计算的一致。")

    print("\nfeature_extractor.py 测试完成。")