import os
from torch.utils.tensorboard import SummaryWriter

class TrainerBase:
    def __init__(self, model, train_dataset, val_dataset, tokenizer=None, batch_size=32, epochs=3, lr=1e-4, save_dir="./logs/model", data_collator=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_dir = save_dir
        self.data_collator = data_collator
        os.makedirs(save_dir, exist_ok=True)
        self.tb_log_dir = os.path.join(save_dir, "tensorboard")
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)

    def log(self, metrics: dict, step: int = None):
        """
        记录日志到TensorBoard。
        metrics: dict，键为tag，值为scalar。
        step: 步数，可选。
        """
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()

    def save(self, name="model"):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
