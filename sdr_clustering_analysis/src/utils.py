# sdr_clustering_analysis/src/utils.py
import os
import pickle
import pandas as pd
import torch # For checking GPU availability

def save_pickle(data, filepath):
    """将数据保存为pickle文件"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到: {filepath}")
    except Exception as e:
        print(f"保存pickle文件失败: {filepath}. 错误: {e}")

def load_pickle(filepath):
    """从pickle文件加载数据"""
    if not os.path.exists(filepath):
        print(f"Pickle文件未找到: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"数据已从 {filepath} 加载")
        return data
    except Exception as e:
        print(f"加载pickle文件失败: {filepath}. 错误: {e}")
        return None

def save_csv(df, filepath, index=False):
    """将DataFrame保存为CSV文件"""
    try:
        df.to_csv(filepath, index=index, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
        print(f"CSV文件已保存到: {filepath}")
    except Exception as e:
        print(f"保存CSV文件失败: {filepath}. 错误: {e}")

def get_device_for_sbert():
    """为SentenceTransformer获取合适的设备 (GPU or CPU)"""
    if torch.cuda.is_available():
        print("检测到可用GPU，将使用cuda设备。")
        return 'cuda'
    else:
        print("未检测到可用GPU，将使用cpu设备。")
        return 'cpu'

if __name__ == '__main__':
    # 简单测试
    print("测试utils.py...")
    
    # 测试设备检测
    device = get_device_for_sbert()
    print(f"SBERT将使用的设备: {device}")

    # 测试pickle保存和加载
    test_data_pkl = {"key": "value", "numbers": [1, 2, 3]}
    test_pkl_path = "test_data.pkl"
    save_pickle(test_data_pkl, test_pkl_path)
    loaded_data_pkl = load_pickle(test_pkl_path)
    if loaded_data_pkl == test_data_pkl:
        print("Pickle保存和加载测试成功。")
    else:
        print("Pickle保存和加载测试失败。")
    if os.path.exists(test_pkl_path):
        os.remove(test_pkl_path)

    # 测试CSV保存 (简单示例)
    test_df_csv = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    test_csv_path = "test_data.csv"
    save_csv(test_df_csv, test_csv_path)
    if os.path.exists(test_csv_path):
        print(f"CSV文件 {test_csv_path} 保存测试成功 (请手动检查内容)。")
        os.remove(test_csv_path)
    else:
        print(f"CSV文件 {test_csv_path} 保存测试失败。")

    print("utils.py 测试完成。")