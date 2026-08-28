ChemCoTBench-V2/
├── raw_benchmark_data/          # 核心：模型输入和最终答案
├── process_evaluation_data/     # 核心：标准推理过程和过程标签
├── formal_templates/            # 每个任务有哪些中间状态字段
├── verifier_rule_descriptions/  # 每个 verifier 检查什么
├── task_schema/                 # 每个 JSON 的字段类型说明
├── evaluation_split_metadata/   # 数据量和文件划分元数据
├── prompt_templates/            # 如何向模型提问的格式说明
├── sample_examples/             # 少量配对示例
├── viewer/                      # Hugging Face 网页展示版本
├── manifest.json                # 核心文件索引
└── README.md