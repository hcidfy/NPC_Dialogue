# GameDialogue

目标：基于 `LooksJuicy/Chinese-Roleplay-Novel` 微调 `Qwen/Qwen3-4B-Instruct-2507`，得到一个简单的 NPC 对话模型（SFT + LoRA）。

## 目录
- `01_download/`：下载模型与数据集到本地
- `02_analysis/`：EDA、tokenizer 与长度分布统计
- `03_clean/`：清洗、统一 ChatML、划分数据集
- `04_training/`：LoRA 训练、合并、评测
- `05_inference/`：聊天、基准、前后对比

## 快速开始
1) 安装依赖

```bash
pip install -r GameDialogue/requirements.txt
```

如果你在 PyCharm 里直接运行 `GameDialogue/**/xxx.py`，工作目录经常会变成脚本所在文件夹，导致出现 `ModuleNotFoundError: No module named 'GameDialogue'`。解决方法是把本项目做一次“可编辑安装”：

```bash
pip install -e .
```

2) 下载数据与模型

```bash
python GameDialogue/01_download/download_dataset.py
python GameDialogue/01_download/download_model.py
python GameDialogue/01_download/verify.py
```

3) EDA

```bash
python GameDialogue/02_analysis/dataset_statistics.py
python GameDialogue/02_analysis/tokenizer_statistics.py
python GameDialogue/02_analysis/length_distribution.py
```

4) 清洗与转换为统一 ChatML

```bash
python GameDialogue/03_clean/remove_null.py
python GameDialogue/03_clean/normalize_text.py
python GameDialogue/03_clean/remove_duplicate.py
python GameDialogue/03_clean/convert_chatml.py
python GameDialogue/03_clean/split_dataset.py
```

5) LoRA 训练

```bash
python GameDialogue/04_training/train_lora.py
```

6) 合并 LoRA

```bash
python GameDialogue/04_training/merge_lora.py
```

7) 推理与对比

```bash
python GameDialogue/05_inference/chat.py --model_path GameDialogue/models/Qwen3-4B-Instruct-merged
python GameDialogue/05_inference/compare_before_after.py
```
