"""
静态向量+分类器训练器

本模块实现了以BERT等大模型作为特征提取器，MLP/RNN/LSTM/BiLSTM等作为分类器的训练器。
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from tqdm import tqdm
from .base_trainer import TrainerBase

class StaticVectorClassifierTrainer(TrainerBase):
    """
    静态向量+分类器训练器。

    参数：
        feature_extractor: 特征提取器（如BERT/RoBERTa等transformer模型）
        classifier: 分类器（MLP/RNN/LSTM/BiLSTM）
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        tokenizer: 分词器
        criterion: 损失函数
        optimizer: 优化器
        data_collator: 数据整理函数
        batch_size: 批大小
        epochs: 训练轮数
        save_dir: 模型保存目录
        dtype: 特征类型
    """
    def __init__(self, feature_extractor, classifier, train_dataset, val_dataset, test_dataset, tokenizer, criterion, optimizer, data_collator, batch_size=8, epochs=3, save_dir="./logs/static_vector_model", dtype="float32"):
        super().__init__(feature_extractor, train_dataset, val_dataset, tokenizer=tokenizer, batch_size=batch_size, epochs=epochs, save_dir=save_dir, data_collator=data_collator)
        self.classifier = classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.test_dataset = test_dataset
        self.dtype = dtype
        # 冻结特征提取器参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        # 分类器权重类型同步
        if self.dtype == "bfloat16":
            self.classifier = self.classifier.to(torch.bfloat16)
        else:
            self.classifier = self.classifier.to(torch.float32)

    def extract_features(self, batch):
        """
        用特征提取器提取静态向量（如[CLS]向量）。
        """
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
            outputs = self.model(**inputs, output_hidden_states=True)
            # 取最后一层[CLS]向量（BERT/RoBERTa等）
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
            else:
                features = outputs[0][:, 0, :]
            if self.dtype == "bfloat16":
                features = features.to(torch.bfloat16)
            else:
                features = features.float()
        return features

    def evaluate(self, dataloader=None, step: int = None):
        self.model.eval()
        self.classifier.eval()
        all_preds = []
        all_labels = []
        if dataloader is None:
            dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)
        with torch.no_grad():
            for batch in dataloader:
                labels = batch['labels'].to(self.model.device)
                if labels.ndim == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                features = self.extract_features(batch)
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
            self.classifier.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                labels = batch['labels'].to(self.model.device)
                if labels.ndim == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                features = self.extract_features(batch)
                print(features.shape)
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
