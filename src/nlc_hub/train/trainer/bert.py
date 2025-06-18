"""
BERT 模型训练器
本模块实现了基于 BERT 的分类训练器，支持模型训练、验证与保存，适用于文本分类等任务。
"""
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .base_trainer import TrainerBase
import torch

class BertTrainer(TrainerBase):
    """
    基于 BERT 的分类训练器。

    该训练器封装了 Huggingface Transformers 的 Trainer，支持训练、验证、模型保存等功能。

    参数：
        model: BERT 主体模型
        train_dataset: 训练集
        val_dataset: 验证集
        tokenizer: 分词器
        training_args: 训练参数（transformers.TrainingArguments）
        save_dir: 保存模型的目录
        dtype: 模型精度（float32、float16、bfloat16）
    """
    def __init__(self, model, train_dataset, val_dataset, tokenizer, training_args, save_dir="./logs/bert_model", dtype="float32"):
        """
        初始化 BertTrainer。

        参数：
            model: BERT 主体模型
            train_dataset: 训练集
            val_dataset: 验证集
            tokenizer: 分词器
            training_args: 训练参数
            save_dir: 保存模型的目录
            dtype: 模型精度
        """
        super().__init__(model, train_dataset, val_dataset, tokenizer=tokenizer)
        from transformers import Trainer
        self.training_args = training_args
        self.save_dir = save_dir
        self.dtype = dtype
        # 根据dtype转换模型精度
        if dtype == "bfloat16":
            self.model = model.to(dtype=torch.bfloat16)
        elif dtype == "float16":
            self.model = model.to(dtype=torch.float16)
        else:
            self.model = model.to(dtype=torch.float32)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )

    def compute_metrics(self, pred):
        """
        计算评估指标（准确率、精确率、召回率、F1）。

        参数：
            pred: transformers.EvalPrediction 对象
        返回：
            dict: 包含 accuracy, f1, precision, recall 的评估指标。
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        # 记录到TensorBoard
        self.log({
            "val/accuracy": acc,
            "val/f1": f1,
            "val/precision": precision,
            "val/recall": recall
        }, step=self.trainer.state.global_step)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self):
        """
        执行模型训练流程。
        """
        self.trainer.train()
        self.save()

    def save(self, name="bert_classifier_model"):
        """
        保存当前模型和分词器到指定目录。

        参数：
            name (str): 保存的文件夹名，默认 'bert_classifier_model'
        """
        save_path = os.path.join(self.save_dir, name)
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
