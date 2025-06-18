import argparse
import yaml
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from torch import nn
from torch.optim import AdamW
import pandas as pd
from nlc_hub.data import load_data, load_llm_data, FraudLLMDataset, FraudDataset
from nlc_hub.train.trainer.bert import BertTrainer
import logging
from nlc_hub.train.trainer.llm_classify import GeneralLLMClassifierTrainer
from nlc_hub.train.trainer.classify import classifier_map
from nlc_hub.train.trainer.feature_extra import (
    extract_cls, extract_mean, extract_sequence
)
from nlc_hub.train.trainer.bert_classify import StaticVectorClassifierTrainer


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using a YAML configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    logging.info(f"Configuration: {config}")
    
    # 设置 CUDA 设备
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Using device: {device}")

    # 根据 trainer_type 选择训练器
    if config['trainer_type'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(config['tokenizer_name_or_path'])
        model = BertForSequenceClassification.from_pretrained(
            config['model_name_or_path'],
            num_labels=config['num_labels']
        )
        model = model.to(device)  # 将模型移动到指定设备
        train_dataset, val_dataset = load_data(
            config['data_path'], tokenizer,
            test_ratio=config['test_ratio'],
            max_len=config['max_length']
        )
        
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            no_cuda=not str(device).startswith('cuda'),  # 根据设备设置是否使用 CUDA
            fp16=True if str(config.get('dtype', '')).lower() == 'float16' else False,
            bf16=True if str(config.get('dtype', '')).lower() == 'bfloat16' else False,
            max_steps=-1,
        )
        
        trainer = BertTrainer(
            model, train_dataset, val_dataset, tokenizer, training_args,
            save_dir=config['output_dir'],
            dtype=config.get('dtype', 'float32')
        )
        trainer.train()

    elif config['trainer_type'] == 'llm':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name_or_path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name_or_path'],
            torch_dtype="auto",
            device_map=device
        )
        hidden_size = model.config.hidden_size
        # 分类头和特征提取函数映射

        feature_map = {
            "cls": extract_cls,
            "mean": extract_mean,
            "sequence": extract_sequence,
        }
        classifier_type = config.get('classifier_type', 'mlp').lower()
        feature_type = config.get('feature_extractor', 'cls').lower()
        ClassifierClass = classifier_map[classifier_type]
        feature_extractor = feature_map[feature_type]
        # 分类头参数自动适配
        if classifier_type == 'mlp':
            classifier = ClassifierClass(hidden_size, config['num_labels']).to(device)
        elif classifier_type == 'rnn':
            classifier = ClassifierClass(hidden_size, config.get('rnn_hidden', 128), config['num_labels']).to(device)
        elif classifier_type == 'lstm':
            classifier = ClassifierClass(hidden_size, config.get('lstm_hidden', 128), config['num_labels']).to(device)
        elif classifier_type == 'bilstm':
            classifier = ClassifierClass(hidden_size, config.get('bilstm_hidden', 128), config['num_labels']).to(device)
        else:
            raise ValueError(f"不支持的分类头类型: {classifier_type}")
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(classifier.parameters(), lr=float(config['learning_rate']))
        train_dataset, val_dataset = load_llm_data(config['data_path'], tokenizer, test_ratio=config['test_ratio'], max_length=config['max_length'])
        test_data = pd.read_csv(config['test_data_path'])
        test_dataset = FraudLLMDataset(test_data['text'].tolist(), test_data['label'].tolist(), tokenizer, max_length=config['max_length'])
        data_collator = None
        trainer = GeneralLLMClassifierTrainer(
            model, classifier, train_dataset, val_dataset, test_dataset, tokenizer,
            criterion, optimizer, data_collator, feature_extractor,
            batch_size=config['batch_size'], epochs=config['epochs'], save_dir=config['output_dir'],
            dtype=config.get('dtype', 'float32')
        )
        trainer.train()
        test_metrics = trainer.evaluate()
        logging.info(f"Final test metrics: {test_metrics}")
        print(f"test_acc {test_metrics['accuracy']}")

    elif config['trainer_type'] == 'static_vector':
        # 1. 加载分词器和特征提取器（如BERT/RoBERTa等）
        tokenizer = BertTokenizer.from_pretrained(config['tokenizer_name_or_path'])
        from transformers import BertModel
        feature_extractor = BertModel.from_pretrained(config['model_name_or_path'])
        feature_extractor = feature_extractor.to(device)
        # 2. 构建分类器
        hidden_size = feature_extractor.config.hidden_size
        classifier_type = config.get('classifier_type', 'mlp').lower()
        ClassifierClass = classifier_map[classifier_type]
        if classifier_type == 'mlp':
            classifier = ClassifierClass(hidden_size, config['num_labels']).to(device)
        elif classifier_type == 'rnn':
            classifier = ClassifierClass(hidden_size, config.get('rnn_hidden', 128), config['num_labels']).to(device)
        elif classifier_type == 'lstm':
            classifier = ClassifierClass(hidden_size, config.get('lstm_hidden', 128), config['num_labels']).to(device)
        elif classifier_type == 'bilstm':
            classifier = ClassifierClass(hidden_size, config.get('bilstm_hidden', 128), config['num_labels']).to(device)
        else:
            raise ValueError(f"不支持的分类头类型: {classifier_type}")
        # 3. 损失函数、优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(classifier.parameters(), lr=float(config['learning_rate']))
        # 4. 数据集
        train_dataset, val_dataset = load_data(
            config['data_path'], tokenizer,
            test_ratio=config['test_ratio'],
            max_len=config['max_length']
        )
        test_data = pd.read_csv(config['test_data_path'])
        test_dataset = FraudDataset(test_data['text'].tolist(), test_data['label'].tolist(), tokenizer, max_length=config['max_length'])
        data_collator = None
        # 5. 实例化Trainer
        trainer = StaticVectorClassifierTrainer(
            feature_extractor, classifier, train_dataset, val_dataset, test_dataset, tokenizer,
            criterion, optimizer, data_collator,
            batch_size=config['batch_size'], epochs=config['epochs'], save_dir=config['output_dir'],
            dtype=config.get('dtype', 'float32')
        )
        trainer.train()
        test_metrics = trainer.evaluate()
        logging.info(f"Final test metrics: {test_metrics}")
        print(f"test_acc {test_metrics['accuracy']}")
    
    else:
        raise ValueError(f"Unsupported trainer type: {config['trainer_type']}")

def cli_main():
    args = parse_args()
    main(args.config)

if __name__ == '__main__':
    cli_main() 