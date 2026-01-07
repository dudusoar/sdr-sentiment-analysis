import torch
from torch.utils.data import Dataset

# 制作数据集
class Tags_dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len, intent2id=None,prediction=False):
        self.data = data
        self.prediction = prediction
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.intent2id = intent2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data.iloc[i]['pure_text']
        tokenized_input = self.tokenizer(text, truncation=True, max_length=self.max_seq_len)
        pad_length = self.max_seq_len - len(tokenized_input['input_ids']) + 1
        tokenized_input['input_ids'] = tokenized_input['input_ids'] + [self.tokenizer.pad_token_id] * pad_length
        tokenized_input['attention_mask'] = tokenized_input['attention_mask'] + [0] * pad_length
        tokenized_input['intent_labels'] = self.data.iloc[i]['label']
        if self.prediction:
            return {
                'input_ids': torch.tensor(tokenized_input['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokenized_input['attention_mask'], dtype=torch.long),
                'intent_labels': torch.tensor(tokenized_input['intent_labels'], dtype=torch.long),
            }
        return {
            'input_ids': torch.tensor(tokenized_input['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_input['attention_mask'], dtype=torch.long),
            'intent_labels': torch.tensor(tokenized_input['intent_labels'], dtype=torch.long),
        }

