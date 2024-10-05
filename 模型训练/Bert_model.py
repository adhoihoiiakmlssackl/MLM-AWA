import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForMaskedLM, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from BERT.tokenization import HexTokenizer, read_hex_data
from BERT.dataset import HexDataset, calculate_accuracy
from BERT.attention_mask import collate_fn

# 初始化配置
config = BertConfig(
    vocab_size=261,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    output_attentions=True,
)

# 使用配置初始化BertForMaskedLM模型
model = BertForMaskedLM(config=config)
print(model)

# # 加载数据和分词器
tokenizer = HexTokenizer()
file_name = 'odd_lines.txt'
strings = read_hex_data(file_name)
dataset = HexDataset(hex_data=strings, tokenizer=tokenizer)

# 划分数据集为训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 设置优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 70
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 定义早停类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 初始化早停机制
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

# 开始训练循环
epoch_loss_values = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        accuracy = calculate_accuracy(outputs.logits, labels, labels != -100)
        total_accuracy += accuracy

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            accuracy = calculate_accuracy(outputs.logits, labels, labels != -100)
            val_loss += loss.item()
            val_accuracy += accuracy

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break
#
# # 保存模型
# torch.save(model.state_dict(), 'model_path_test5-29-加入早停-请求应答训练（正常BERT模型）.pth')
