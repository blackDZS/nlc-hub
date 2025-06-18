# ====== 常用特征提取函数 ======
# 1. 取 [CLS] embedding（适用于 BERT、Qwen3 等）
def extract_cls(outputs, batch):
    if hasattr(outputs, 'hidden_states'):
        hidden = outputs.hidden_states[-1]
    elif isinstance(outputs, dict) and 'hidden_states' in outputs:
        hidden = outputs['hidden_states'][-1]
    elif isinstance(outputs, (tuple, list)) and hasattr(outputs[0], 'shape'):
        hidden = outputs[0]
    else:
        raise ValueError('无法从模型输出中提取 hidden_states')
    return hidden[:, 0, :].float()

# 2. 平均池化（适用于无 [CLS] 的模型）
def extract_mean(outputs, batch):
    if hasattr(outputs, 'hidden_states'):
        hidden = outputs.hidden_states[-1]
    elif isinstance(outputs, dict) and 'hidden_states' in outputs:
        hidden = outputs['hidden_states'][-1]
    elif isinstance(outputs, (tuple, list)) and hasattr(outputs[0], 'shape'):
        hidden = outputs[0]
    else:
        raise ValueError('无法从模型输出中提取 hidden_states')
    mask = batch.get('attention_mask', None)
    if mask is not None:
        mask = mask.to(hidden.device).unsqueeze(-1)
        hidden = hidden * mask
        return hidden.sum(1) / mask.sum(1)
    else:
        return hidden.mean(1)

# 3. 输出序列全部 hidden states（适用于 RNN/LSTM/BiLSTM 分类头）
def extract_sequence(outputs, batch):
    if hasattr(outputs, 'hidden_states'):
        return outputs.hidden_states[-1]
    elif isinstance(outputs, dict) and 'hidden_states' in outputs:
        return outputs['hidden_states'][-1]
    elif isinstance(outputs, (tuple, list)) and hasattr(outputs[0], 'shape'):
        return outputs[0]
    else:
        raise ValueError('无法从模型输出中提取 hidden_states') 
