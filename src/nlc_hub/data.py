import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nlc_hub.prompt import FraudPrompt

prompt_obj = FraudPrompt()
prompt_obj.examples = []

# 自定义Dataset
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: BertTokenizer, max_length=512, stride=256):
        """
        texts: List[str] 原始文本
        labels: List[int] 每条文本的标签（涉诈：0，非涉诈：1）
        tokenizer: 使用的 tokenizer
        max_length: 每段最大 token 长度（建议512）
        stride: 滑动窗口步长（建议256）
        """
        self.samples = []
        for text, label in zip(texts, labels):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(token_ids), stride):
                chunk = token_ids[i:i + max_length]
                if not chunk:
                    break
                chunk_text = tokenizer.decode(chunk)
                self.samples.append((chunk_text, label))
                if i + max_length >= len(token_ids):
                    break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors="pt")
        # squeeze 是为了去掉 batch 维度，方便 DataLoader
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path, tokenizer, test_ratio=0.2, max_len=128):
    # 加载数据（假设你有一个CSV，包含'text'和'label'列）
    df = pd.read_csv(data_path)  # 替换为你的数据路径
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=test_ratio, random_state=42
    )

    return FraudDataset(train_texts, train_labels, tokenizer, max_len), FraudDataset(val_texts, val_labels, tokenizer, max_len)


# ====== Dummy Dataset (replace with your real dataset) ======
class FraudLLMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.data = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 如果模型没有 pad_token，就把 eos 当作 pad。
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        messages = [{"role": "user", "content": prompt_obj.format_prompt(text)}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"   #固定补齐到 max_length
        )

        # squeeze 去掉 batch 这一维
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

def load_llm_data(data_path, tokenizer, test_ratio=0.2, max_length=1024):
    # 加载数据（假设你有一个CSV，包含'text'和'label'列）
    df = pd.read_csv(data_path)  # 替换为你的数据路径
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=test_ratio, random_state=42
    )

    return FraudLLMDataset(train_texts, train_labels, tokenizer, max_length), FraudLLMDataset(val_texts, val_labels, tokenizer, max_length)

