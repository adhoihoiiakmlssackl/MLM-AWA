pcap 包内含所有训练数据
对PCAP包数据处理其是将wireshark导出的原始数据进行处理，使其数据只有从应用层开始的原始数据
模型训练包含 分词，修改掩码机制、模型训练、保存训练模型、滚动掩码评估阶段
加载预训练保存模型包含提取每一个序列每个token注意力权重位置，计算重合位置处的注意力权重值、生成字段标签
原始数据打标签包含对原始数据进行标签标注工作，以此来与生成的字段标签作对比


