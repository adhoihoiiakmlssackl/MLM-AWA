import random
from torch.nn.utils.rnn import pad_sequence
from BERT.tokenization import HexTokenizer
tokenizer=HexTokenizer()
#概率掩码
def mask_tokens(token_ids, vocab, mask_probability=0.20):
    # token_ids: 输入token的ID列表
    # vocab: 词汇表，包含特殊token
    # mask_probability: 掩码的概率，默认15%
    masked_token_ids = list(token_ids)  # 创建一个可修改的副本
    labels = [-100] * len(token_ids)  # 初始化标签列表，-100表示将忽略此位置的损失计算

    # 随机选择掩码的位置，确保不掩码特殊token
    candidate_indices = [i for i, token_id in enumerate(token_ids)
                         if token_id not in (vocab['[REQ]'], vocab['[SEP]'], vocab['[RESP]'],vocab['[PAD]'])]
    num_to_mask = int(len(candidate_indices) * mask_probability)
    random.shuffle(candidate_indices)
    mask_indices = candidate_indices[:num_to_mask]

    for index in mask_indices:
        labels[index] = token_ids[index]  # 保存真实的token ID作为标签
        # 替换为 [MASK]
        masked_token_ids[index] = vocab['[MASK]']

    return masked_token_ids, labels
# #滚动掩码
# def mask_tokens_sequentially(token_ids, vocab):
#     masked_token_ids = list(token_ids)  # 创建一个可修改的副本
#     labels = [-100] * len(token_ids)  # 初始化标签列表，-100表示将忽略此位置的损失计算
#
#     # 在每个调用中只掩码一个token，滚动掩码
#     for mask_position in range(len(token_ids)):
#         if token_ids[mask_position] in [vocab['[CLS]'], vocab['[SEP]'], vocab['[PAD]']]:
#             continue  # 跳过特殊token的掩码
#         masked_token_ids[mask_position] = vocab['[MASK]']  # 替换为[MASK]
#         labels[mask_position] = token_ids[mask_position]  # 保存原始token的ID为标签
#
#     return masked_token_ids, labels  # 返回掩码后的token IDs和对应的标签

#
# def mask_response_tokens_only(token_ids, vocab, start_index=0):
#     # token_ids: 输入token的ID列表
#     # vocab: 词汇表，包含特殊token
#     # start_index: 响应部分开始的索引
#
#     # 创建一个可修改的token_ids副本
#     masked_token_ids = list(token_ids)
#     labels = [-100] * len(token_ids)  # 初始化labels列表，-100表示将忽略此位置的损失计算
#
#     # 只在响应部分应用掩码
#     response_token_ids = masked_token_ids[start_index:]
#
#     # 应用掩码策略，例如，随机选取15%的tokens应用掩码
#     mask_probability = 0.15
#     num_to_mask = int(len(response_token_ids) * mask_probability)
#     mask_indices = random.sample(range(start_index, len(token_ids)), num_to_mask)
#
#     for index in mask_indices:
#         # 确保不掩码特殊token
#         if token_ids[index] not in [ vocab['[SEP]'], vocab['[REQ]'], vocab['[RESP]'], vocab['[PAD]']]:
#             masked_token_ids[index] = vocab['[MASK]']
#             labels[index] = token_ids[index]
#
#     return masked_token_ids, labels
import random

def mask_and_pad_tokens(token_ids, vocab, start_index):
    # 初始化掩码后的token IDs和标签
    masked_token_ids = list(token_ids)
    labels = [-100] * len(token_ids)  # -100表示在计算损失时忽略此位置
    mask_probability = 0.15
    # 仅考虑从start_index开始的token用于掩码
    candidate_indices = [
        i for i in range(start_index, len(token_ids))
        if token_ids[i] not in (vocab['[SEP]'], vocab['[REQ]'], vocab['[RESP]'], vocab['[PAD]'])
    ]
    # print("start_index:", start_index)  # 调试信息
    # print("candidate_indices length:", len(candidate_indices))  # 调试信息
    # print("num_to_mask before min/max:", int(len(candidate_indices) * mask_probability))  # 调试信息

    # 计算要掩码的token数量

    num_to_mask = max(1, int(len(candidate_indices) * mask_probability))  # 至少掩码1个token
    mask_indices = random.sample(candidate_indices, num_to_mask)
    attention_mask = [1] * len(token_ids)
    if mask_indices:
        first_mask_index = min(mask_indices)
        masked_token_ids[first_mask_index] = vocab['[MASK]']
        labels[first_mask_index] = token_ids[first_mask_index]

        # 从第一个掩码位置之后开始填充[PAD]并更新attention_mask为0
        for i in range(first_mask_index + 1, len(token_ids)):
            masked_token_ids[i] = vocab['[PAD]']
            labels[i] = -100  # 忽略此位置的损失计算
            attention_mask[i] = 0  # 标记为不关注的位置

    return masked_token_ids, labels, attention_mask


def prepare_sequence(request_tokens, response_tokens, tokenizer, max_length=512):
    # 以[REQ]作为起始标记
    sep_token_id = tokenizer.vocab['[SEP]']
    pad_token_id = tokenizer.vocab['[PAD]']
    req_token_id = tokenizer.vocab['[REQ]']
    resp_token_id = tokenizer.vocab['[RESP]']

    # 确保请求包不超过200个token
    truncated_request_tokens = [req_token_id] + request_tokens[:256]

    # 特殊标记的数量：[CLS], [REQ], [SEP], [RESP], [SEP] （总共5个）
    special_tokens_count = 3

    # 确保响应包加上请求包和特殊标记不超过模型最大长度限制
    max_resp_length = max_length - len(truncated_request_tokens) - special_tokens_count
    truncated_response_tokens = [resp_token_id] + response_tokens[:max_resp_length]

    # 组合序列
    token_ids = [req_token_id] + truncated_request_tokens + [sep_token_id] + [resp_token_id]+ truncated_response_tokens

    # 创建attention mask
    attention_mask = [1] * len(token_ids)

    # 如果需要，进行填充
    padding_length = max_length - len(token_ids)
    token_ids += [pad_token_id] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask


def collate_fn(batch, max_length=512):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 动态计算这个批次的最大长度，但不超过512
    max_length_batch = min(max(len(ids) for ids in input_ids), max_length)

    # 对每个序列进行裁剪或填充至相同长度
    input_ids_padded = pad_sequence([ids.clone().detach()[:max_length_batch] for ids in input_ids],
                                    batch_first=True, padding_value=tokenizer.vocab['[PAD]'])
    attention_mask_padded = pad_sequence([mask.clone().detach()[:max_length_batch] for mask in attention_mask],
                                         batch_first=True, padding_value=0)
    labels_padded = pad_sequence([lbl.clone().detach()[:max_length_batch] for lbl in labels],
                                 batch_first=True, padding_value=-100)

    return {'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_padded}
