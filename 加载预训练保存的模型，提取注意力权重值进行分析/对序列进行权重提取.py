def parse_line(line):
    # 解析文件中的一行数据，返回一个元组 (seq, masked_position)
    if line.startswith("Sequence"):
        seq = int(line.split()[1][:-1])
        return seq, None, None
    elif line.startswith("Masked position (adjusted):"):
        masked_position = int(line.split()[3][1:-1])
        return None, masked_position, None
    elif "Attention weight:" in line:
        head_start_index = line.find("Head") + len("Head") + 1
        head_end_index = line.find(":", head_start_index)
        head = line[head_start_index:head_end_index]
        attention_weight_start_index = line.find(":", head_end_index) + len(":") + 1
        attention_weight = line[attention_weight_start_index:].strip()
        return None, None, (head, attention_weight)
    else:
        return None, None, None

def sort_file(input_file, output_file):
    sequences = []
    current_seq = None

    with open(input_file, 'r') as f:
        for line in f:
            seq, masked_position, head_data = parse_line(line)
            if seq is not None:
                current_seq = {"seq": seq, "masked_position": masked_position, "head_data": []}
                sequences.append(current_seq)
            elif masked_position is not None:
                current_seq["masked_position"] = masked_position
            elif head_data is not None:
                current_seq["head_data"].append(head_data)

    sequences.sort(key=lambda x: x["masked_position"])

    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(f"Sequence {seq['seq']}:\n")
            # f.write(f"Masked position (original):[{seq['masked_position']}] \n")
            f.write(f"Masked position (adjusted): [{seq['masked_position']}]\n")
            for head, attention_weight in seq['head_data']:
                f.write(f"Head {head}:\n")
                f.write(f"Attention weight: {attention_weight}\n")
if __name__ == "__main__":
    input_file = "experiment_results.txt"  # 输入文件路径
    output_file = "s7comm-2.txt"  # 输出文件路径
    sort_file(input_file, output_file)
def format_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    current_sequence = None
    current_layer = -1  # 初始化当前层级为 -1，因为第一个 "Layer" 是从 0 开始计数
    current_head = None

    for line in lines:
        line = line.strip()
        if line.startswith("Sequence"):
            current_sequence = line
            output_lines.append(current_sequence + "\n")
            current_layer = -1  # 每个序列开始时，将当前层级重新设置为 -1
        elif line.startswith("Masked position") or line.startswith("Attention weight"):
            output_lines.append(line + "\n")
        elif line.startswith("Layer"):
            current_layer += 1
            output_lines.append(line + ":\n")
        elif line.startswith("Head"):
            if "Head 0" in line:
                current_layer += 1
                output_lines.append("Layer " + str(current_layer) + ":\n")
            current_head = line
            output_lines.append(current_head + "\n")
        elif line.strip() == "":
            output_lines.append("\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(output_lines))
# 调用函数并提供输入文件和输出文件路径
format_data("s7comm-2.txt", "3")


