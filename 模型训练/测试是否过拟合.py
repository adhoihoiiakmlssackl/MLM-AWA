import torch
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from BERT.tokenization import HexTokenizer, read_hex_data
from BERT.dataset import HexDataset
from BERT.attention_mask import collate_fn

# 配置 BERT 模型参数
config = BertConfig(
    vocab_size=261,  # 指定词汇表的大小，必须与您的分词器中的大小匹配
    hidden_size=256,  # 隐藏层的大小
    num_hidden_layers=2,  # 隐藏层数量
    num_attention_heads=2,  # 注意力头的数量
    intermediate_size=256,  # 中间层的大小
    hidden_act="gelu",  # 隐藏层的激活函数
    hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率
    attention_probs_dropout_prob=0.1,  # 注意力层的 dropout 概率
    initializer_range=0.02,  # 权重初始化的范围
    output_attentions=True,  # 输出注意力权重
)

# 加载预训练的 BERT 模型
model = BertForMaskedLM(config=config)
model.load_state_dict(torch.load('model_path_256-5-25-2head.pth'))
model.eval()

# 加载分词器和数据集
tokenizer = HexTokenizer()
hex_data = read_hex_data('1.txt')

# 创建测试集的数据加载器
test_dataset = HexDataset(hex_data=hex_data, tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn)

# 定义损失函数
criterion = cross_entropy

# 评估函数
def evaluate(model, data_loader):
    model.eval()  # 确保模型处于评估模式
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 计算损失
            total_loss += loss.item()

            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100  # -100 在损失计算中被忽略
            correct_predictions = (predictions == labels) & mask
            total_correct += correct_predictions.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy

# 评估测试集的表现
test_loss, test_accuracy = evaluate(model, test_loader)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 判断过拟合情况的标准
# 如果仅有测试集结果：
# if test_loss > 0.5:  # 这个值可以根据实际情况调整
#     print("The model might be overfitting or not generalizing well.")
# else:
#     print("The model is performing well on the test set.")
