# original_file_path = 'xiugai.txt'
# new_file_path = 'xunlian.txt'
#
# # 读取原始文件，处理每行，并将结果写入新文件
# with open(original_file_path, 'r') as file, open(new_file_path, 'w') as new_file:
#     for line in file:
#         # 删除每行的前28个字符
#         trimmed_line = line[14:].strip()
#         # 检查删除后的行是否以"32"开头
#         if trimmed_line.startswith("32"):
#             # 如果是，则将裁剪后的行写入到新文件
#             new_file.write(trimmed_line + '\n')
#         # 如果不是"32"开头，则忽略该行，不执行任何操作
#
# print(f"Trimmed sequences saved to {new_file_path}")
#
# # filtered_file_path = 's7comm-1.txt'
# # non_empty_file_path = 's7comm-2.txt'
# #
# # # 读取过滤后的文件，移除空行，并将结果写入新文件
# # with open(filtered_file_path, 'r') as file, open(non_empty_file_path, 'w') as new_file:
# #     for line in file:
# #         if line.strip():  # 检查行是否非空
# #             new_file.write(line)
#
# # print(f"Non-empty sequences saved to {non_empty_file_path}")
def split_and_save_file_by_line_parity(file_path, odd_lines_path, even_lines_path):
    # 分别存储奇数行和偶数行的列表
    odd_lines = []
    even_lines = []

    # 读取文件，并将奇偶行分别存储
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number % 2 == 0:
                even_lines.append(line)  # 保留换行符，以便写入文件时保持原有格式
            else:
                odd_lines.append(line)

    # 将奇数行写入指定的文件
    with open(odd_lines_path, 'w') as file:
        file.writelines(odd_lines)

    # 将偶数行写入指定的文件
    with open(even_lines_path, 'w') as file:
        file.writelines(even_lines)


# 调用函数，假设原始文件名为'sample.txt'
# 奇数行数据将保存到'odd_lines.txt'，偶数行数据将保存到'even_lines.txt'
split_and_save_file_by_line_parity('5.txt', 'odd_lines.txt', 'even_lines.txt')
#
# print("奇数行数据已保存到 odd_lines.txt")
# print("偶数行数据已保存到 even_lines.txt")
