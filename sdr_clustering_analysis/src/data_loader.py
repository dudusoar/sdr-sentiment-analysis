# sdr_clustering_analysis/src/data_loader.py
import pandas as pd
import os
from config import RAW_COMMENTS_FILE, TEXT_COLUMN, ID_COLUMN, MANUAL_SENTIMENT_COLUMN

def load_raw_comments(filepath=RAW_COMMENTS_FILE):
    """
    从Excel或CSV文件加载原始评论数据。
    确保指定的文本列、ID列和情感标签列存在。
    """
    if not os.path.exists(filepath):
        print(f"错误: 原始数据文件未找到: {filepath}")
        return None

    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            print(f"错误: 不支持的文件格式: {filepath}. 请使用 .xlsx 或 .csv。")
            return None
        print(f"原始数据已从 {filepath} 加载。共 {len(df)} 条评论。")

        # 检查必要的列是否存在
        required_columns = [TEXT_COLUMN]
        if ID_COLUMN: # ID列是可选的，但推荐有
            required_columns.append(ID_COLUMN)
        if MANUAL_SENTIMENT_COLUMN: # 手动情感标签列也是可选的，但用于后续对比分析
            required_columns.append(MANUAL_SENTIMENT_COLUMN)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: 原始数据中缺少以下列: {', '.join(missing_columns)}")
            print(f"请检查config.py中的TEXT_COLUMN, ID_COLUMN, MANUAL_SENTIMENT_COLUMN设置是否与Excel/CSV文件列名匹配。")
            print(f"文件中存在的列: {df.columns.tolist()}")
            return None

        # 筛选掉文本列为空的行
        original_len = len(df)
        df.dropna(subset=[TEXT_COLUMN], inplace=True)
        df = df[df[TEXT_COLUMN].astype(str).str.strip() != '']
        if len(df) < original_len:
            print(f"已移除 {original_len - len(df)} 条文本内容为空的评论。")

        # 确保文本列是字符串类型
        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

        # 如果ID列配置了，确保其唯一性，若不唯一则创建唯一ID
        if ID_COLUMN and ID_COLUMN in df.columns:
            if df[ID_COLUMN].duplicated().any():
                print(f"警告: 配置的ID列 '{ID_COLUMN}' 中存在重复值。将创建新的唯一ID列 'unique_comment_id'。")
                df['unique_comment_id'] = range(len(df))
                df.rename(columns={ID_COLUMN: f"original_{ID_COLUMN}"}, inplace=True) # 保留原始ID列
                # 更新config中的ID_COLUMN常量到新的唯一ID列，虽然config本身不会改，但程序内部可以用新的
                # 或者，更好的做法是在返回的df中使用固定的新ID列名
                # 此处我们选择返回带有 'unique_comment_id' 的df，让调用者知道
                print(f"请在后续分析中使用 'unique_comment_id' 作为唯一标识。")
            else: # 如果原始ID列已经是唯一的，可以将其设为索引或重命名为 'unique_comment_id'
                df.rename(columns={ID_COLUMN: 'unique_comment_id'}, inplace=True)

        elif not ID_COLUMN or ID_COLUMN not in df.columns:
            print(f"提示: 未配置有效ID列 ('{ID_COLUMN}')。将创建新的唯一ID列 'unique_comment_id'。")
            df['unique_comment_id'] = range(len(df))


        print(f"数据加载和初步处理完成。剩余 {len(df)} 条有效评论。")
        return df

    except Exception as e:
        print(f"加载原始数据文件失败: {filepath}. 错误: {e}")
        return None

if __name__ == '__main__':
    print("测试 data_loader.py...")
    # 假设你的 combined_comments.xlsx 文件在 config.py 指定的 DATA_DIR 中
    # 并且 TEXT_COLUMN 等在 config.py 中已正确设置
    comments_df = load_raw_comments()

    if comments_df is not None:
        print(f"\n成功加载 {len(comments_df)} 条评论。")
        print("\n前5条评论的预览 (文本、ID、情感标签):")
        
        preview_cols = []
        # 使用 TEXT_COLUMN 和 MANUL_SENTIMENT_COLUMN from config
        if TEXT_COLUMN in comments_df.columns:
            preview_cols.append(TEXT_COLUMN)
        
        # 优先使用 'unique_comment_id' 如果它存在
        if 'unique_comment_id' in comments_df.columns:
            preview_cols.append('unique_comment_id')
        elif ID_COLUMN and ID_COLUMN in comments_df.columns: # 否则回退到配置的ID_COLUMN
            preview_cols.append(ID_COLUMN)

        if MANUAL_SENTIMENT_COLUMN and MANUAL_SENTIMENT_COLUMN in comments_df.columns:
            preview_cols.append(MANUAL_SENTIMENT_COLUMN)
        
        if preview_cols:
            print(comments_df[preview_cols].head())
        else:
            print("无法展示预览，请确保config.py中的列名配置正确。")
            print(comments_df.head())

        print(f"\nDataFrame列名: {comments_df.columns.tolist()}")
    else:
        print("数据加载失败。请检查错误信息和文件路径/内容。")