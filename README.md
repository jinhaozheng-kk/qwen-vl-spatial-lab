# Qwen3-VL 空间关系指令微调实验

在一张 8 GB RTX 4060 Laptop GPU 上，用 Qwen3-VL-2B-Instruct 判断红圆相对蓝方块的方向，并比较两张图中方向是否改变。仓库含全部合成图、训练代码、可直接加载的 LoRA 权重和逐题预测。

## 测到了什么

64 条训练指令，32 条独立种子生成的测试指令。两种任务各占一半。两次评测的题目、顺序和贪心解码参数相同。

| 任务（各 16 题） | 基础模型 → LoRA，标签判断正确 | 基础模型 → LoRA，符合输出格式 | 基础模型 → LoRA，严格匹配 |
| --- | --- | --- | --- |
| 单图，判断左 / 右 / 上 / 下 | 16 → 16 | 8 → 16 | 8 → 16 |
| 双图，判断相对方向有没有变化 | 12 → 15 | 16 → 16 | 12 → 15 |

严格匹配要求去除首尾空白后与标准答案完全相同。格式指标仅检查是否输出一个允许的大写标签。标签判断忽略大小写并提取完整候选词，适用于本次短答案。单图 test-014 的标准答案是 BELOW，基础模型输出 Below，微调后输出 BELOW。这里改善的是指令遵循。

### 双图失败案例 · test-003

| 图 1 | 图 2 |
| --- | --- |
| ![第一张测试图](data/test/003-0.png) | ![第二张测试图](data/test/003-1.png) |

题目问红圆相对蓝方块的方向有无变化。两张图中的物体位置、大小有扰动，但相对方向一致，标准答案为 NO。两个模型都输出 YES。这个例子说明格式正确的回答也可能包含空间判断错误。仅凭输出无法确定模型误判的内部原因。

[基础模型逐题预测](results/base.json) · [LoRA 逐题预测](results/lora.json) · [环境、显存与逐样本损失](results/run.json)

## 图像怎样进入语言模型

处理器按 user 消息中 image 项出现的顺序读取图像并插入视觉占位符，随后拼入文本问题。视觉编码器将像素划成 16×16 空间 patch。当前处理器把本实验的 224×224 方图调整至 256×256，得到 16×16 个空间 patch。merger 将相邻 2×2 特征组合并映射到语言模型的 2048 维隐藏空间，因此每图占 64 个视觉 token，双图占 128 个。逐题日志记录了实际输入中的图像占位 token 数。

语言模型在对应占位位置接收合并后的视觉向量；DeepStack 还将多个视觉层的特征送入语言模型的早期层。增加分辨率可保留更多局部细节，也会加长输入序列。本实验沿用预训练的视觉编码器和投影层，仅更新语言侧 LoRA。[官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## 数据与后训练

数据生成器先选方向，再绘制带随机位置、尺寸扰动的红圆和蓝方块。双图负例重新绘图但保留方向，正例改变方向。训练集和测试集分别使用种子 1701 和 3177。JSONL 每行包含 id、task 和 messages；user.content 依次放一或两张图的路径及问题，assistant.content 保存标准答案。[实际训练数据格式](data/train/rows.jsonl)

基础模型完成零样本评测后，训练器对同一对话分别编码提示前缀和完整问答，验证前缀一致，再将前缀标签置为 -100。只有回答及模板末尾参与监督。训练结束后卸载训练适配器，从磁盘重新加载权重，在留出集重新生成答案。

| 配置 | 本次运行 |
| --- | --- |
| 精度与注意力 | BF16 基座，PyTorch SDPA |
| LoRA | q_proj、v_proj；rank 8，alpha 16，dropout 0.05 |
| 优化器 | AdamW，固定学习率 0.0001，梯度裁剪 1.0 |
| 训练规模 | 1 epoch，micro-batch 1，累积 2，32 次参数更新 |
| 推理峰值 | 基础模型 4.010 GiB，加载适配器后 4.027 GiB |
| 训练峰值 | 4.847 GiB |

显存为各阶段 PyTorch 的峰值已分配张量显存，驱动和缓存另占空间。任务使用固定颜色、形状、问法与视角，测试集只检验相同生成规则下的新场景；自然照片、遮挡和开放式空间描述仍需独立评估。

## 复现与推理

实测 Windows、Python 3.12.11、CUDA 版 PyTorch。以下命令在仓库根目录、具备兼容 NVIDIA 驱动的 Python 环境运行。首次运行需要下载官方基础权重，已有权重可通过 --model 指向本地模型目录。

~~~sh
python -m pip install -r requirements.txt
python infer.py data/test/003-0.png data/test/003-1.png --adapter adapter --question "Between image 1 and image 2, did the red circle change sides relative to the blue square? Answer exactly YES or NO."
~~~

省略 --adapter adapter 即可运行基础模型。单图推理只传一个图片路径。以下命令重新生成数据、训练适配器并覆盖本地实验输出。

~~~sh
python data_gen.py
python experiment.py
~~~
