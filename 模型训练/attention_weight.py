import torch

def print_top_attention_positions(attention_weights, top_k_percentage=5):
    num_layers = len(attention_weights)

    for layer in range(num_layers):
        layer_attention = attention_weights[layer].squeeze(0)  # 假设 batch_size=1
        num_heads, seq_length, _ = layer_attention.shape

        print(f"Layer {layer}:")
        for head in range(num_heads):
            print(f"  Head {head}:")
            top_k = max(int(seq_length * top_k_percentage / 100), 1)  # 计算30%的位置数目

            for token_index in range(seq_length):
                # 获取每个token的前30%最大注意力权重位置
                top_weights_indices = torch.topk(layer_attention[head, token_index], top_k).indices
                print(f"    Token {token_index} -> Top Attention Positions: {top_weights_indices.tolist()}")


def collect_top_attention_weights(attention_weights, token_idx, top_k_percentage=10):
    num_layers = len(attention_weights)
    top_weights_per_layer = []

    for layer in range(num_layers):
        layer_attention = attention_weights[layer].squeeze(0)  # 假设 batch_size=1
        num_heads, seq_length, _ = layer_attention.shape
        top_weights = []

        weights_per_head = []
        for head in range(num_heads):
            top_k = max(int(seq_length * top_k_percentage / 100), 1)
            top_k_weights, top_k_indices = torch.topk(layer_attention[head, token_idx], top_k)
            weights_per_head.append((top_k_indices.tolist(), top_k_weights.tolist()))
        top_weights.append(weights_per_head)

        top_weights_per_layer.append(top_weights)

    return top_weights_per_layer
def find_overlapping_positions(top_positions_per_layer):
    overlapping_positions_per_layer = []

    for layer, top_positions in enumerate(top_positions_per_layer):
        overlapping_positions = []
        for i in range(len(top_positions) - 1):
            overlap_per_head = []
            for head in range(len(top_positions[i])):
                overlap = set(top_positions[i][head]).intersection(top_positions[i + 1][head])
                overlap_per_head.append(list(overlap))
            overlapping_positions.append(overlap_per_head)
        overlapping_positions_per_layer.append((layer, overlapping_positions))

    return overlapping_positions_per_layer


def group_tokens_into_fields_per_head(top_weights_per_layer, overlap_threshold=0.5):
    fields_per_layer_per_head = []

    for layer, top_weights in enumerate(top_weights_per_layer):
        fields_per_head = []
        num_heads = len(top_weights[0][0])
        seq_length = len(top_weights)

        for head in range(num_heads):
            fields = []
            i = 0
            while i < seq_length - 1:
                positions_i = set(top_weights[i][0][head][0])
                positions_i_plus_1 = set(top_weights[i + 1][0][head][0])
                overlap_positions = positions_i.intersection(positions_i_plus_1)

                overlap_weight_i = sum(
                    [top_weights[i][0][head][1][top_weights[i][0][head][0].index(pos)] for pos in overlap_positions])
                overlap_weight_i_plus_1 = sum(
                    [top_weights[i + 1][0][head][1][top_weights[i + 1][0][head][0].index(pos)] for pos in overlap_positions])

                if overlap_weight_i + overlap_weight_i_plus_1 < overlap_threshold:
                    fields.append((i,))
                    i += 1
                else:
                    fields.append((i, i + 1))
                    i += 2

            if i == seq_length - 1:
                fields.append((i,))
            fields_per_head.append(fields)
        fields_per_layer_per_head.append((layer, fields_per_head))

    return fields_per_layer_per_head

# def group_tokens_into_fields_per_head(top_weights_per_layer, overlap_threshold=50, required_agreements=3):
#     fields_per_layer_per_head = []
#
#     for layer, top_weights in enumerate(top_weights_per_layer):
#         fields_per_head = []
#         seq_length = len(top_weights)
#         num_heads = len(top_weights[0])
#
#         for head in range(num_heads):
#             fields = []
#             i = 0
#             while i < seq_length:
#                 field = [i]  # 初始化字段为当前token
#                 # 尝试将后续的token加入当前字段
#                 while i < seq_length - 1:
#                     next_token = i + 1
#                     total_agreement_count = 0  # 初始化全字段满足条件的计数器
#
#                     # 检查当前字段中的每一个token与next_token的重叠权重是否满足条件
#                     for field_token in field:
#                         agreement_count = 0  # 对每一对token初始化符合条件的计数
#                         for layer_idx in range(len(top_weights_per_layer)):
#                             for head_idx in range(len(top_weights_per_layer[layer_idx][0])):
#                                 positions_field_token = set(top_weights_per_layer[layer_idx][field_token][head_idx][0])
#                                 positions_next_token = set(top_weights_per_layer[layer_idx][next_token][head_idx][0])
#                                 overlap_positions = positions_field_token.intersection(positions_next_token)
#
#                                 overlap_weight_field_token = sum([top_weights_per_layer[layer_idx][field_token][
#                                                                       head_idx][1][
#                                                                       top_weights_per_layer[layer_idx][field_token][
#                                                                           head_idx][0].index(pos)] for pos in
#                                                                   overlap_positions])
#                                 overlap_weight_next_token = sum([top_weights_per_layer[layer_idx][next_token][head_idx][
#                                                                      1][top_weights_per_layer[layer_idx][next_token][
#                                     head_idx][0].index(pos)] for pos in overlap_positions])
#
#                                 if overlap_weight_field_token > overlap_threshold / 100.0 and overlap_weight_next_token > overlap_threshold / 100.0:
#                                     agreement_count += 1
#
#                         # 如果满足条件的层和头部数达到了指定的 required_agreements，则计为一次全字段满足
#                         if agreement_count >= required_agreements:
#                             total_agreement_count += 1
#
#                     # 判断是否所有token都与next_token满足条件
#                     if total_agreement_count == len(field):
#                         field.append(next_token)
#                         i = next_token
#                     else:
#                         break
#                 fields.append(tuple(field))
#                 i += 1
#
#             fields_per_head.append(fields)
#         fields_per_layer_per_head.append((layer, fields_per_head))
#
#     return fields_per_layer_per_head

# def print_top_mask_attention_positions(attention_weights):
#     num_layers = len(attention_weights)
#
#     for layer in range(num_layers):
#         layer_attention = attention_weights[layer].squeeze(0)  # 假设 batch_size=1
#         num_heads, seq_length, _ = layer_attention.shape
#
#         print(f"Layer {layer}:")
#         for head in range(num_heads):
#             print(f"  Head {head}:")
#             mask_indices = torch.nonzero(layer_attention[head].sum(dim=1))[:, 0]  # 找到非零元素的位置
#             mask_indices = mask_indices[:5]  # 只保留前五个掩码位置
#
#             for token_index in mask_indices:
#                 # 获取每个mask位置的前五个最大注意力权重位置
#                 top_weights_indices = torch.topk(layer_attention[head, token_index], 5).indices
#                 print(f"    Token {token_index.item()} -> Top Attention Positions: {top_weights_indices.tolist()}")
#
# def collect_top_mask_attention_weights(attention_weights, masked_positions):
#     num_layers = len(attention_weights)
#     top_weights_per_layer = []
#
#     for layer in range(num_layers):
#         layer_attention = attention_weights[layer].squeeze(0)  # 假设 batch_size=1
#         num_heads, seq_length, _ = layer_attention.shape
#         top_weights = []
#
#         for head in range(num_heads):
#             weights_per_head = []
#
#             for mask_index in masked_positions:
#                 mask_index = mask_index.item()  # assuming a single mask position
#                 top_k_weights, top_k_indices = torch.topk(layer_attention[head, mask_index], 5)
#                 weights_per_head.append((mask_index, top_k_indices.tolist(), top_k_weights.tolist()))
#
#             top_weights.append(weights_per_head)
#
#         top_weights_per_layer.append(top_weights)
#
#     return top_weights_per_layer
