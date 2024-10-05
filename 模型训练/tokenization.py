import torch
class HexTokenizer:
    def __init__(self):
        # 创建一个词汇表，映射十六进制字符到十进制ID
        self.vocab = {f"{i:02x}": i for i in range(256)}
        # 添加特殊token
        self.vocab['[PAD]'] = 256
        self.vocab['[REQ]'] = 257
        self.vocab['[RESP]'] = 258
        self.vocab['[MASK]'] = 259
        self.vocab['[SEP]'] = 260
        # 反向映射，从ID到十六进制字符
        self.ids_to_tokens = {i: f"{i:02x}" for i in range(256)}
        self.ids_to_tokens[256] = '[PAD]'
        self.ids_to_tokens[257] = '[REQ]'
        self.ids_to_tokens[258] = '[RESP]'
        self.ids_to_tokens[259] = '[MASK]'
        self.ids_to_tokens[260] = '[SEP]'

    def tokenize(self, hex_string):
        # 清除字符串中的空格
        hex_string = hex_string.replace(' ', '')

        tokens = []
        i = 0
        while i < len(hex_string):
            # 检测特殊标记
            if hex_string[i:].startswith('[REQ]'):
                tokens.append('[REQ]')
                i += len('[REQ]')
            elif hex_string[i:].startswith('[RESP]'):
                tokens.append('[RESP]')
                i += len('[RESP]')
            elif hex_string[i:].startswith('[MASK]'):
                tokens.append('[MASK]')
                i += len('[MASK]')
            elif hex_string[i:].startswith('[SEP]'):
                tokens.append('[SEP]')
                i += len('[SEP]')
            else:
                # 正确分割十六进制字符
                tokens.append(hex_string[i:i + 2])
                i += 2
        return tokens

#这个是单独训练
# class HexTokenizer:
#     def __init__(self):
#         # 创建一个词汇表，映射十六进制字符到十进制ID
#         self.vocab = {f"{i:02x}": i for i in range(256)}
#         # 添加特殊token
#         self.vocab['[PAD]'] = 256
#         self.vocab['[CLS]'] = 257
#         self.vocab['[SEP]'] = 258
#         self.vocab['[MASK]'] = 259
#         # 反向映射，从ID到十六进制字符
#         self.ids_to_tokens = {i: f"{i:02x}" for i in range(256)}
#         self.ids_to_tokens[256] = '[PAD]'
#         self.ids_to_tokens[257] = '[CLS]'
#         self.ids_to_tokens[258] = '[SEP]'
#         self.ids_to_tokens[259] = '[MASK]'
#
#     def tokenize(self, hex_string):
#         # 清除字符串中的空格
#         hex_string = hex_string.replace(' ', '')
#
#         tokens = []
#         i = 0
#         while i < len(hex_string):
#             # 检测特殊标记
#             if hex_string[i:].startswith('[CLS]'):
#                 tokens.append('[CLS]')
#                 i += 5  # 跳过 '[CLS]'
#             elif hex_string[i:].startswith('[SEP]'):
#                 tokens.append('[SEP]')
#                 i += 5  # 跳过 '[SEP]'
#             elif hex_string[i:].startswith('[MASK]'):
#                 tokens.append('[MASK]')
#                 i += 6  # 跳过 '[MASK]'
#             else:
#                 # 正确分割十六进制字符
#                 tokens.append(hex_string[i:i + 2])
#                 i += 2
#         return tokens
    def convert_tokens_to_ids(self, tokens):
        # 转换token为ID
        return [self.vocab[token] for token in tokens if token in self.vocab]
    def convert_ids_to_tokens(self, ids):
        # 转换ID为token
        return [self.ids_to_tokens[id] for id in ids if id in self.ids_to_tokens]

    def encode(self, text, add_special_tokens=True, max_length=None, pad_to_max_length=True,
               padding=False, truncation=False, return_tensors='pt'):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = tokens
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        token_ids = self.convert_tokens_to_ids(tokens)

        # 检查是否需要填充
        if pad_to_max_length and max_length is not None:
            # 计算需要填充的数量
            padding_length = max_length - len(token_ids)
            # 填充 [PAD] token
            token_ids += [self.vocab['[PAD]']] * padding_length

        # 转换为 PyTorch 张量
        input_ids = torch.tensor([token_ids])

        # 如果需要返回张量
        if return_tensors == 'pt':
            # 创建 attention_mask
            attention_mask = [1 if token_id != self.vocab['[PAD]'] else 0 for token_id in token_ids]
            attention_mask = torch.tensor([attention_mask])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        return token_ids

    def decode(self, token_ids):
        # 使用convert_ids_to_tokens将ID转换为token
        tokens = self.convert_ids_to_tokens(token_ids)
        # 将token列表合并为字符串，特殊tokens和十六进制tokens按原样保留
        decoded_string = ''.join(tokens)
        return decoded_string

def read_hex_data(file_path):
    """
    从文本文件中读取十六进制数据。
    假设每行是由逗号分隔的十六进制字符串。
    """
    with open(file_path, 'r') as file:
        # 读取所有行
        lines = file.readlines()
    # 去除空白字符并分割每行的数据
    hex_data = []
    for line in lines:
        # 去除逗号并去除空白字符
        hex_string = line.replace(',', '').strip()
        hex_data.append(hex_string)
    return hex_data


# class HexTokenizer:
#     def __init__(self):
#         # 创建一个词汇表，映射十六进制字符到十进制ID
#         self.vocab = {f"{i:02x}": i for i in range(256)}
#         # 添加特殊token
#         self.vocab['[PAD]'] = 256
#         self.vocab['[REQ]'] = 257
#         self.vocab['[RESP]'] = 258
#         self.vocab['[MASK]'] = 259
#         # 反向映射，从ID到十六进制字符
#         self.ids_to_tokens = {i: f"{i:02x}" for i in range(256)}
#         self.ids_to_tokens[256] = '[PAD]'
#         self.ids_to_tokens[257] = '[REQ]'
#         self.ids_to_tokens[258] = '[RESP]'
#         self.ids_to_tokens[259] = '[MASK]'
#
#     def tokenize(self, hex_string):
#         # 清除字符串中的空格
#         hex_string = hex_string.replace(' ', '')
#
#         tokens = []
#         i = 0
#         while i < len(hex_string):
#             # 检测特殊标记
#             if hex_string[i:].startswith('[REQ]'):
#                 tokens.append('[REQ]')
#                 i += 5  # 跳过 '[CLS]'
#             elif hex_string[i:].startswith('[RESP]'):
#                 tokens.append('[RESP]')
#                 i += 5  # 跳过 '[SEP]'
#             elif hex_string[i:].startswith('[MASK]'):
#                 tokens.append('[MASK]')
#                 i += 6  # 跳过 '[MASK]'
#             else:
#                 # 正确分割十六进制字符
#                 tokens.append(hex_string[i:i + 2])
#                 i += 2
#         return tokens