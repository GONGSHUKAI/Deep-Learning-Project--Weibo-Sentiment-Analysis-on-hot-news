from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class WeiboDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, item):
        text = self.dataframe.iloc[item]['tweet']
        label = self.dataframe.iloc[item]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class WeiboSentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, item):
        text = self.dataframe.iloc[item]['tweet']
        label = 1 if self.dataframe.iloc[item]['label'] == '__label__positive' else 0
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
