import json


# 定义函数来自动标注序列
def auto_label(sequence):
    # 将序列每两个字符分割为一个token
    tokens = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    labeled_sequence = {
        "text": sequence,
        "entities": []
    }

    # 根据规则设置标签
    field_indices = [
        [0], [1], [2, 3], [4, 5], [6, 7], [8, 9], [10], [11], [12], [13]
        , [14], [15], [16, 17], [18, 19], [20], [21, 22, 23]
    ]
    for field in field_indices:
        # 将索引转换为实际的字符偏移
        start_offset = field[0] * 2
        end_offset = (field[-1] + 1) * 2
        # 生成字段的标签，例如 'Field0', 'Field1', ...
        label = str(field[0])
        # 添加实体标签
        labeled_sequence["entities"].append({
            "id": len(labeled_sequence["entities"]),
            "start_offset": start_offset,
            "end_offset": end_offset,
            "label":  str(len(labeled_sequence["entities"]))
        })

    return labeled_sequence


# 读取原始数据
with open('s7comm-2.txt', 'r') as file:
    data = file.readlines()

# 应用自动标注
labeled_data = [auto_label(line.strip()) for line in data]

# 保存为Doccano格式
with open('s7comm-2.jsonl', 'w') as outfile:
    for entry in labeled_data:
        json.dump(entry, outfile)
        outfile.write('\n')

