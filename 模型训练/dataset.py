import torch
from BERT.attention_mask import mask_and_pad_tokens,mask_tokens
from BERT.tokenization import HexTokenizer,read_hex_data
from torch.utils.data import Dataset
import torch.nn.functional as F
tokenizer=HexTokenizer()
# #联合训练但是仍旧是bert基本掩码机制训练的时候用
# class HexDataset(Dataset):
#     def __init__(self, hex_data, tokenizer):
#         self.tokenizer = tokenizer
#         self.hex_data = hex_data
#
#     def __len__(self):
#         return len(self.hex_data)
#
#     def __getitem__(self, idx):
#         # 获取原始十六进制字符串
#         hex_string = self.hex_data[idx]
#
#         # 在字符串开头增加[REQ]标记
#         hex_string = '[REQ]' + hex_string
#
#         # 在字符串中找到第二个"32"出现的位置，并在它之前插入[SEP]和[RESP]标记
#         response_start_idx = hex_string.find('32', hex_string.find('32') + 1)
#         if response_start_idx != -1:
#             hex_string = hex_string[:response_start_idx] + '[SEP][RESP]' + hex_string[response_start_idx:]
#
#         # 使用自定义分词器对字符串进行分词
#         tokens = self.tokenizer.tokenize(hex_string)
#         # 将tokens转换为对应的ID
#         token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#         # 应用掩码策略
#         masked_token_ids, labels = mask_tokens(token_ids, self.tokenizer.vocab)
#         # 创建attention_mask
#         attention_mask = [1] * len(masked_token_ids)
#
#         return {
#             'input_ids': torch.tensor(masked_token_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }

# # ///这个是单独训练的
# class HexDataset(Dataset):
#     def __init__(self, hex_data, tokenizer):
#         self.tokenizer = tokenizer
#         self.hex_data = hex_data
#
#     def __len__(self):
#         return len(self.hex_data)
#
#     def __getitem__(self, idx):
#         hex_string = self.hex_data[idx]
#         # 使用自定义分词器对字符串进行分词
#         tokens = self.tokenizer.tokenize(hex_string)
#         # 添加特殊token [CLS] 和 [SEP]
#         tokens = ['[CLS]'] + tokens + ['[SEP]']
#         # 将tokens转换为对应的ID
#         token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#         # 应用掩码策略
#         masked_token_ids, labels = mask_tokens(token_ids, self.tokenizer.vocab)
#         # 创建attention_mask
#         attention_mask = [1] * len(masked_token_ids)
#
#         return {
#             'input_ids': torch.tensor(masked_token_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }
###这个是用来计算单独训练加载数据计算single和together用的
class HexDataset(Dataset):
    def __init__(self, hex_data, tokenizer):
        self.tokenizer = tokenizer
        self.hex_data = hex_data
        self.preprocessed_data = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data)) for data in hex_data]

    def __len__(self):
        return sum(len(data) for data in self.preprocessed_data)

    def __getitem__(self, idx):
        total_tokens = 0
        for sequence_idx, token_ids in enumerate(self.preprocessed_data):
            if idx < total_tokens + len(token_ids):
                token_idx = idx - total_tokens
                break
            total_tokens += len(token_ids)

        masked_token_ids = token_ids[:]  # 创建副本以进行掩码
        labels = [-100] * len(token_ids)  # 初始化labels

        if token_idx < len(token_ids):
            labels[token_idx] = token_ids[token_idx]
            masked_token_ids[token_idx] = self.tokenizer.vocab['[MASK]']

        return {
            'input_ids': torch.tensor(masked_token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(masked_token_ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'token_idx': torch.tensor(token_idx, dtype=torch.long)  # 返回被掩码的token索引
        }


# # 这个是拼接完成之后 只掩盖一个mask后面的都用pad进行填充的
# class HexDataset(Dataset):
#     def __init__(self, hex_data, tokenizer):
#         self.tokenizer = tokenizer
#         self.hex_data = hex_data
#
#     def __len__(self):
#         return len(self.hex_data)
#
#     def __getitem__(self, idx):
#         # 获取原始十六进制字符串
#         hex_string = self.hex_data[idx]
#
#         # 在字符串开头增加[REQ]标记
#         hex_string = '[REQ]' + hex_string
#
#         # 在字符串中找到第二个"32"出现的位置，并在它之前插入[SEP]和[RESP]标记
#
#         response_start_idx = hex_string.find('32', hex_string.find('32') + 1)
#         if response_start_idx != -1:
#             hex_string = hex_string[:response_start_idx] + '[SEP][RESP]' + hex_string[response_start_idx:]
#
#         # 使用自定义分词器对字符串进行分词
#         tokens = self.tokenizer.tokenize(hex_string)
#
#         # 将tokens转换为对应的ID
#         token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#
#         # 找到[REQ]和[RESP]标记的索引位置
#         req_index = token_ids.index(self.tokenizer.vocab['[REQ]'])
#         try:
#             resp_index = token_ids.index(self.tokenizer.vocab['[RESP]'])
#         except ValueError:
#             resp_index = len(token_ids)  # 如果没有找到[RESP]，则整个序列视为请求
#
#         # 裁剪请求部分以确保长度不超过256
#         end_of_request = min(req_index + 1 + 256, resp_index) if resp_index != len(token_ids) else req_index + 1 + 256
#
#         # 组合裁剪后的请求部分和完整的响应部分
#         adjusted_token_ids = token_ids[:end_of_request] + token_ids[resp_index:]
#
#         adjusted_token_ids, labels, attention_mask = mask_and_pad_tokens(token_ids, self.tokenizer.vocab,
#                                                                          start_index=end_of_request)
#
#         return {
#             'input_ids': torch.tensor(adjusted_token_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),  # 确保使用新的attention_mask
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }

def calculate_accuracy(logits, labels, mask):
    # 计算准确率
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    return correct.sum().item() / mask.sum().item()




def calculate_entropy(logits, attention_mask):
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

    # 应用掩码，仅保留非填充部分的熵值
    masked_entropy = entropy * attention_mask
    return masked_entropy


