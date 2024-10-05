import re

# 定义序列类
class Sequence:
    def __init__(self, sequence_number):
        self.sequence_number = sequence_number
        self.masked_positions = []
        self.attention_weights = []

# 从文本中提取序列数据
def extract_sequences(text):
    sequences = []
    lines = text.split("\n")
    current_sequence = None
    for line in lines:
        if line.startswith("Sequence"):
            sequence_number = int(line.split("::")[0].split()[-1])
            current_sequence = Sequence(sequence_number)
            sequences.append(current_sequence)
        elif line.startswith("Masked position"):
            masked_position = int(re.search(r"\[(\d+)\]", line).group(1))
            current_sequence.masked_positions.append(masked_position)
        elif line.startswith("Attention weight"):
            attention_weight = [float(x) for x in re.findall(r"\d+\.\d+", line)]
            current_sequence.attention_weights.append(attention_weight)
    return sequences

# 清理重复的数据
def clean_duplicates(sequences):
    cleaned_sequences = []
    masked_positions_set = set()
    for sequence in sequences:
        for masked_position in sequence.masked_positions:
            if masked_position not in masked_positions_set:
                cleaned_sequences.append(sequence)
                masked_positions_set.add(masked_position)
                break
    return cleaned_sequences

# 将序列对象转换为字符串
def sequences_to_string(sequences):
    output = ""
    for sequence in sequences:
        output += f"Sequence {sequence.sequence_number}::\n"
        for i, masked_position in enumerate(sequence.masked_positions):
            output += f"Masked position (adjusted): [{masked_position}]\n"
            for j, attention_weight in enumerate(sequence.attention_weights[i]):
                output += f"Attention position: {attention_weight}  Attention weight: {sequence.attention_weights[i+1][j]}\n"
    return output

# 读取原始文件内容
with open("3", "r") as file:
    original_text = file.readlines()

# 提取序列数据
sequences = []
current_sequence = []
for line in original_text:
    if line.startswith("Sequence"):
        if current_sequence:
            sequences.append(current_sequence)
            current_sequence = []
        current_sequence.append(line)
    else:
        current_sequence.append(line)
sequences.append(current_sequence)  # 处理最后一个序列

# 清理重复数据
cleaned_sequences = []
masked_positions_set = set()
for sequence in sequences:
    masked_position = int(sequence[1].split(":")[1].strip().split("[")[1].split("]")[0])
    if masked_position not in masked_positions_set:
        cleaned_sequences.append(sequence)
        masked_positions_set.add(masked_position)

# 将清理后的结果保存到文件中
with open("4.txt", "w") as file:
    for sequence in cleaned_sequences:
        for line in sequence:
            file.write(line)

print("清理完成，并已保存到 4.txt 中。")

