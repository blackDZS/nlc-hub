"""
通用大模型+多类型分类头（MLP/RNN/LSTM/BiLSTM）训练器

本模块实现了适配大多数大模型（如 Qwen、BERT、LLAMA、Baichuan 等）+多种分类头（MLP、RNN、LSTM、BiLSTM）的训练器，支持自定义特征提取、分类头结构配置。
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
from .base_trainer import TrainerBase
from typing import Callable

class GeneralLLMClassifierTrainer(TrainerBase):
    """
    通用大模型+多类型分类头训练器。

    支持自定义特征提取、MLP/RNN/LSTM/BiLSTM 分类头结构，兼容不同大模型。

    参数：
        model: 主体大模型
        classifier: 分类器（MLP/RNN/LSTM/BiLSTM）
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        tokenizer: 分词器
        criterion: 损失函数
        optimizer: 优化器
        data_collator: 数据整理函数
        feature_extractor: 特征提取函数，输入为模型输出和 batch，返回特征
        batch_size: 批大小，默认 8
        epochs: 训练轮数，默认 3
        save_dir: 模型保存目录，默认 './logs/llm_model'
        dtype: 特征类型，默认 "float32"
    """
    def __init__(self, model, classifier, train_dataset, val_dataset, test_dataset, tokenizer, criterion, optimizer, data_collator, feature_extractor: Callable, batch_size=8, epochs=3, save_dir="./logs/llm_model", dtype="float32"):
        super().__init__(model, train_dataset, val_dataset, tokenizer=tokenizer, batch_size=batch_size, epochs=epochs, save_dir=save_dir, data_collator=data_collator)
        self.classifier = classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.test_dataset = test_dataset
        self.feature_extractor = feature_extractor
        self.dtype = dtype
        # 分类器权重类型同步
        if self.dtype == "bfloat16":
            self.classifier = self.classifier.to(torch.bfloat16)
        else:
            self.classifier = self.classifier.to(torch.float32)

    def log(self, metrics: dict, step: int = None):
        super().log(metrics, step=step)

    def evaluate(self, dataloader=None, step: int = None):
        self.model.eval()
        self.classifier.eval()
        all_preds = []
        all_labels = []
        if dataloader is None:
            dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.model.device)
                if labels.ndim == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                outputs = self.model(**inputs, output_hidden_states=True)
                features = self.feature_extractor(outputs, batch)
                # 新增：特征类型转换
                if self.dtype == "bfloat16":
                    features = features.to(torch.bfloat16)
                else:
                    features = features.float()
                logits = self.classifier(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        self.log({
            "val/accuracy": acc,
            "val/f1": f1,
            "val/precision": precision,
            "val/recall": recall
        }, step=step)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        best_f1 = 0
        global_step = 0
        for epoch in range(self.epochs):
            total_loss = 0
            self.model.eval()
            self.classifier.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.model.device)
                if labels.ndim == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    features = self.feature_extractor(outputs, batch)
                    # 新增：特征类型转换
                    if self.dtype == "bfloat16":
                        features = features.to(torch.bfloat16)
                    else:
                        features = features.float()
                logits = self.classifier(features)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                global_step += 1
            avg_loss = total_loss / len(train_loader)
            self.log({"train/loss": avg_loss}, step=global_step)
            print(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")
            metrics = self.evaluate(val_loader, step=global_step)
            print(f'epoch:{epoch}->metrics:{metrics}')
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                self.save("best_classifier.pt")
        self.save("final_classifier.pt")
        print("\n在测试集上进行最终评估...")
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        test_metrics = self.evaluate(test_loader, step=global_step)
        print(f'测试集评估结果: {test_metrics}')
        self.log({
            "test/accuracy": test_metrics["accuracy"],
            "test/f1": test_metrics["f1"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"]
        }, step=global_step)

    def save(self, name="final_classifier.pt"):
        """
        保存当前分类器权重到指定目录。
        参数：
            name (str): 文件名，默认 'final_classifier.pt'
        """
        save_path = os.path.join(self.save_dir, name)
        torch.save(self.classifier.state_dict(), save_path)


