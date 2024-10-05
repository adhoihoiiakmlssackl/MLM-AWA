def calculate_attention_weight_sum_per_sequence(sequence1, sequence2):
    """
    计算两个序列中每个序列的重合位置的权重值总和。

    参数:
    - sequence1: 第一个序列的数据，格式为 {layer: {head: (indices, weights)}}
    - sequence2: 第二个序列的数据，格式相同

    返回:
    - result: 一个字典，包含每个序列的每个层每个头部的权重值总和
    """
    result = {
        'Sequence1': {},
        'Sequence2': {}
    }

    # 遍历所有层
    for layer in sequence1:
        result['Sequence1'][layer] = {}
        result['Sequence2'][layer] = {}

        # 遍历每个头部
        for head in sequence1[layer]:
            # 获取两个序列中当前层当前头部的索引和权重
            indices1, weights1 = sequence1[layer][head]
            indices2, weights2 = sequence2[layer][head]

            # 找出重合的索引
            common_indices = set(indices1).intersection(set(indices2))

            # 计算每个序列在重合索引处的权重值总和
            weight_sum1 = sum(weights1[indices1.index(index)] for index in common_indices if index in indices1)
            weight_sum2 = sum(weights2[indices2.index(index)] for index in common_indices if index in indices2)

            result['Sequence1'][layer][head] = weight_sum1
            result['Sequence2'][layer][head] = weight_sum2

    return result
def parse_sequence_file(file_path):
    sequences = {}
    current_sequence = None
    current_layer = None
    current_head = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Sequence'):
                sequence_name = line.split(':')[0]
                current_sequence = sequence_name
                sequences[current_sequence] = {}
            elif line.startswith('Layer'):
                layer_name = line.split(':')[0]
                current_layer = layer_name
                sequences[current_sequence][current_layer] = {}
            elif line.startswith('Head'):
                head_name = line.split(':')[0]
                current_head = head_name
            elif line.startswith('Attention weight'):
                parts = line.split(':')
                # 提取括号内的内容并解析为列表
                indices_str = line[line.find('['):line.find(']')+1]
                weights_str = line[line.rfind('['):line.rfind(']')+1]
                indices = eval(indices_str)
                weights = eval(weights_str)
                sequences[current_sequence][current_layer][current_head] = (indices, weights)

    return sequences


def main():
    # 解析文件
    sequence_file_path = '4.txt'  # 替换为实际文件路径
    sequences = parse_sequence_file(sequence_file_path)

    # 假设我们要比较文件中的前两个序列
    sequence_names = list(sequences.keys())
    sequence1 = sequences[sequence_names[0]]
    sequence2 = sequences[sequence_names[1]]

    result = calculate_attention_weight_sum_per_sequence(sequence1, sequence2)

    for sequence in result:
        print(f"{sequence}:")
        for layer in result[sequence]:
            for head in result[sequence][layer]:
                print(f"  {layer} {head}: {result[sequence][layer][head]}")


if __name__ == '__main__':
    main()
