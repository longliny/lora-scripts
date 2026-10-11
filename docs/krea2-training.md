# Krea 2 Raw LoRA 训练

在现有训练工作台选择 **Krea 2 Raw LoRA**，即可使用项目原生训练器。训练循环、数据处理、缓存、权重转换、保存和续训位于 `mikazuki/training/`；模型层使用 Diffusers，适配器层使用 PEFT，不调用或安装 OneTrainer。

当前支持单张 CUDA GPU、BF16 计算和 DiT LoRA。已使用本机 `models/Krea-2-Raw` 完整权重与 `train/bi` 六张图片完成 FP8/NF4 短训、保存、FP8 断点恢复和预览测试。短训用于验证功能与显存，不代表已验证长期收敛或成品画质。

## 从工作台开始

1. 在训练器选择框选择 **Krea 2 Raw LoRA**。
2. 底模填写 `krea/Krea-2-Raw`，或选择完整 Diffusers 目录。在线下载前，需在[模型页面](https://huggingface.co/krea/Krea-2-Raw)获得访问权限，并在运行环境登录 Hugging Face；工作台不保存访问令牌。
3. 本地目录需包含 `model_index.json`，以及 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/`、`scheduler/`。已持有原始 `raw.safetensors` 时，在专家视图的「Transformer 权重覆盖」中选择该文件；其余组件仍来自完整底模。
4. 选择图片目录。每张图片配同名 UTF-8 `.txt` Caption，例如 `image01.png` 与 `image01.txt`；支持递归子目录。缺失或空 Caption 会明确报错。关闭 Caption 打乱，将保留 Token 数设为 0。
5. 设置输出名称，先使用默认配置：NF4 底模、Flash Attention 2、BF16 计算、Rank/Alpha 16、学习率 `1e-4`、AdamW8bit、梯度检查点、VAE 分块和 Batch Size 1。需要 FP8 时，在「显存与性能 → 底模权重精度」选择「FP8 E4M3（BF16 计算）」。
6. 预览编译结果并启动。原有任务中心可查看日志、Loss、学习率、停止任务与输出文件。

默认分辨率 1024×1024，Bucket 步长 16；训练和采样宽高都必须是 16 的倍数。NF4 降低冻结 DiT 权重占用，计算仍为 BF16，可训练 LoRA 参数为 FP32。BF16 底模仅参数本身约 24GB，32GB 显卡还需留出激活、优化器和显示程序显存，因此不作为这台机器的默认方案。

## FP8 与 Attention

`base_precision=fp8` 使用项目实现的逐输出通道缩放 FP8 E4M3 权重存储。主 DiT 和文本融合块的冻结线性权重压缩到 FP8；输入、时间、输出投影保留 BF16，归一化保留模型规定的 FP32，LoRA 使用 FP32。先在 CPU 加载、初始化适配器并量化，再搬到 GPU，避免先占用整个 BF16 底模的显存。前向和反向逐层还原 BF16 权重，自定义反向仅保存 FP8 权重与缩放值，不为每层常驻保留 BF16 副本。这是 FP8 权重存储，不是 FP8 Tensor Core 矩阵计算。

项目新增 `attn_mode`，控制 DiT 注意力：

| 值 | 行为 |
| --- | --- |
| `sdpa` | PyTorch 自动选择注意力算子；CLI 默认值。 |
| `flash` | 显式调用独立 `flash-attn` 的 Flash Attention 2 前后向；工作台默认值。启动时检查实际 GPU 是否可用，不静默退回数学实现。 |
| `xformers` | 可选 xFormers 自动调度。当前环境已卸载，界面禁用该选项；安装兼容版本后刷新即可恢复。 |

Krea2 使用分组查询注意力（GQA）。独立 Flash 原生处理共享 K/V 头的梯度，无需扩展 K/V 头。项目对不同长度 Caption 和 CFG 的有效键值去除 Padding，保持全部 Query 和原始 Rotary 位置。GPU 测试比较了带掩码、不带掩码及完全屏蔽行的输出和 Q/K/V 梯度。文本编码器缓存阶段仍使用 SDPA。

本次安装的 Windows 官方 PyTorch 2.14.1 未编译原生 Flash Attention，不能仅凭设置 `SDPA` 就认定启用了 Flash。项目的 `flash` 使用单独安装的 `flash-attn`，不依赖 xFormers。缺少包时，工作台禁用该选项，新草稿默认使用 SDPA。

不提供文本编码器 LoRA、全量微调、多 GPU、随机裁剪或 Caption 动态增强；训练器会拒绝不支持的配置。支持 AdamW/AdamW8bit、恒定/带预热恒定/线性/余弦学习率；预热步数为整数。

## 缓存与原始权重

编码分三阶段：先加载 VAE 缓存图片并释放，再加载 Qwen3-VL 缓存文本并释放，最后加载 DiT 训练。训练期间无需将三个组件同时保留在显存。

部分官方快照由 Transformers 5 保存，使用列表形式的 `extra_special_tokens` 和 `rope_parameters`。项目加载器在内存中转换为当前 Transformers 4 的等价字段，保留 Token ID、RoPE theta 和 mRoPE 分段，并检查文本编码器没有缺失或不匹配的权重；不修改底模文件。

缓存默认在输出目录的 `.krea2-cache/`；可在工作台指定独立缓存目录。文本缓存只存有效 Token，保留位于 Padding 后的 Assistant 后缀。图片、Caption、模型文件、Token 上限或种子变化后会生成新缓存，不复用不匹配的内容。缓存使用 Safetensors，并按需从磁盘读取。

Diffusers 0.39 没有 Krea 2 的 `from_single_file`。项目因此将原始权重逐张量重命名、重塑调制表并写为约 2GB 一片的 Diffusers 权重。转换结果存入缓存并在后续训练复用；需要额外的磁盘空间，大致等于原始权重大小。未知、缺失或形状不匹配的张量会使转换失败。

## 保存、续训与预览

- LoRA 文件：`output/<输出名称>.safetensors`，定期保存添加 `-step000100` 或 `-epoch000001` 后缀。使用 Diffusers Krea2Pipeline 支持的命名和 PEFT 元数据，保留 Alpha/Rank 缩放。
- 启用「保存训练状态」后，同目录生成 `-state/`，内含 LoRA、优化器、调度器、随机状态及精确的数据游标。将该目录填入「继续训练状态」即可恢复。
- 续训需要原训练计划的底模、数据、Rank、Alpha、底模精度、Attention、优化器、种子、梯度累积和总步数保持一致。它用于继续中断的计划，不用于改变计划后追加训练。校验签名和适配器哈希可发现不匹配或未完成的断点。
- 预览使用当前 Raw 底模，推荐 52 步、CFG 3.5。Krea 的此值按 `cond + CFG × (cond − uncond)` 解释，CFG 0 关闭引导。预览会临时加载 VAE，可能增加峰值显存；图片保存在输出目录的 `sample/`。

Krea 推荐在 Raw 上训练、将 LoRA 加载到 Turbo 推理。Raw 预览不能代表 Turbo 的最终效果；Turbo 通常使用 8 步、CFG 0。参考[官方模型说明](https://huggingface.co/krea/Krea-2-Raw)和 [Diffusers 的 Krea 2 说明](https://github.com/huggingface/diffusers/blob/v0.39.0/examples/dreambooth/README_krea2.md)。

恢复会还原优化器、随机状态与数据游标。Flash Attention 2 默认反向是非确定性算子，后续更新不保证逐位一致；升级环境或切换 Attention 后也不能把结果视为同一条数值轨迹。此前 Torch 2.7 / xFormers Flash 的完整 FP8 从第 3 步恢复时，第 4 步前向 Loss 与原训练完全相同，后续最大 Loss 差为 `1.95e-5`。当前环境的小模型 SDPA 续训测试单独验证了逐位相同。

## 环境选择与升级

当前环境为 **PyTorch 2.14.1+cu130 / torchvision 0.29.1+cu130 / flash-attn 2.8.3+cu130torch2.14**，xFormers 已卸载。保留 Transformers 4.57.6、NumPy 1.26.4、Protobuf 3.20.3，配合 Diffusers 0.39.0、PEFT 0.18.1、Accelerate 1.15.0 和 bitsandbytes 0.50.2。

安装步骤、社区 Flash wheel 来源与校验、SDXL/Anima 兼容范围、历史 xFormers 0.0.35 / dev1147 候选结论见 [训练环境说明](training-environment.md)。当前配置基于这台 Windows / RTX 5090 的实测，不能声称在所有模型上都是最快方案。

## 验证和维护

```powershell
.\venv\Scripts\python.exe -m unittest discover -s mikazuki/tests -p 'test_*.py'
$env:KREA2_GPU_TESTS = '1'
.\venv\Scripts\python.exe -m unittest mikazuki.tests.training.test_krea2.Krea2GpuTests -v
Remove-Item Env:KREA2_GPU_TESTS
.\venv\Scripts\python.exe -m mikazuki.catalog.inspector generate --scripts-root scripts/dev
```

GPU 测试使用小规模的真实 Krea2Transformer、Qwen3VLModel 与 AutoencoderKLQwenImage，覆盖缓存、NF4/FP8、AdamW8bit、GQA/掩码/Flash 前后向、冻结底模、SDPA 精确续训、LoRA 元数据加载以及出图。普通测试覆盖 UI 配置编译、输入约束、FP8 输入梯度及保存张量精度、分桶、缓存失效、VAE 归一化、Patch 顺序、动态时间步和原始权重分片转换。

环境检查脚本逐一导入九个工作台训练器，并在指定 `--gpu` 时检查 SDPA、独立 Flash、AdamW8bit 和 SDXL/Anima/FLUX 的小型网络反向传播；xFormers 未安装时报告为不可用而非环境故障。这些检查不等于所有模型都完成了完整底模训练；其余模型的全量训练与收敛仍需对应权重和数据。

2026-10-10 最终后端回归共 219 项：217 项通过，2 项既有条件测试跳过，包含全部 Krea2 GPU 测试。浏览器确认 FP8 和 Attention 下拉框沿用现有控件、布局与配色。已有前端 8 项测试、TypeScript 检查和 Vite 构建通过。

### 完整权重实测（2026-10-10）

设备：Windows / RTX 5090 32GB；用户提供的 `models/Krea-2-Raw` 和 `train/bi` 六张图片。1024 面积预算下实际 Bucket 为 880×1168，Batch 1，Rank/Alpha 16，学习率 `1e-4`，AdamW8bit，梯度检查点，每组 6 步。

| 环境 / 底模精度 | 第 2–6 步平均耗时 | 峰值 allocated | 峰值 reserved | 平均 Loss |
| --- | ---: | ---: | ---: | ---: |
| **当前 Torch 2.14.1 / 独立 Flash / FP8** | **3.05 秒/步** | **16.04 GiB** | **17.10 GiB** | **0.05742** |
| 历史 Torch 2.7 / xFormers Flash / FP8 | 2.66 秒/步 | 16.07 GiB | 17.40 GiB | 0.05742 |
| 历史 Torch 2.7 / xFormers Flash / NF4 | 2.42 秒/步 | 9.49 GiB | 10.63 GiB | 0.06338 |

当前 FP8 训练完成六步更新、定期/最终 LoRA 和训练状态保存，以及一次 512×512、4 步 CFG 预览。显存由 PyTorch 进程统计，包含这次预览，不包含其他程序、驱动或缓存编码阶段。第 1 步为 3.48 秒，表中耗时排除首步和保存/采样；不能把这些数字当作整个任务的总耗时。旧 NF4 完整底模结果只代表旧环境，当前 NF4 已通过真实小模型训练测试。

测试在按授权检查 ComfyUI 队列为空、卸载其模型后进行。短训 Loss 不能用来比较最终画质；完整 BF16、SDXL/Anima 完整底模、长训收敛和 Turbo 完整模型加载未在此验收中测试。

同卡独立注意力基准（4096 Token、48 个 Q 头、12 个 K/V 头、Head Dim 128、BF16，前向与反向）中，当前 Torch SDPA efficient attention 为 **37.20ms / 0.59GiB**，独立 Flash Attention 2 为 **10.99ms / 0.59GiB**。Profiler 确认 `flash_attn::_flash_attn_forward` 和 `_flash_attn_backward`。约 3.38 倍的注意力算子差异不能套用到整个训练过程。

```powershell
.\venv\Scripts\python.exe -m mikazuki.scripts.benchmark_krea2_attention --output logs/krea2-attention-current.json
```

当前环境汇总为 `logs/attention-upgrade-validation.json`，环境检查为 `logs/training-environment-torch214-flash.json`，算子基准为 `logs/krea2-attention-torch214-flash.json`。当前完整模型产物位于 `output/krea2-validation/torch214-fp8-flash/`，逐步耗时与显存记录在 `krea2-fp8-flash-result.json`。历史 FP8/NF4、续训及候选环境报告仍保留在 `logs/krea2-validation-report.json` 等文件中。这些 LoRA 是短训验收产物，不能当作完成训练的模型使用。

## 参考与实现边界

- [Krea 官方模型](https://huggingface.co/krea/Krea-2-Raw)：Raw/Turbo 的用途及模型资产。
- [Krea 官方代码](https://github.com/krea-ai/krea-2)：架构与原始权重约定。
- [OneTrainer Krea2Model](https://github.com/Nerogar/OneTrainer/blob/master/modules/model/Krea2Model.py)、[BaseKrea2Setup](https://github.com/Nerogar/OneTrainer/blob/master/modules/modelSetup/BaseKrea2Setup.py)：交叉核对文本层选择、有效 Token、潜变量归一化、Patch 顺序和时间偏移。
- [Diffusers 0.39 Krea2Pipeline](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/krea2/pipeline_krea2.py)：模型库 API、文本模板与采样规范。

训练实现入口为 `scripts/krea2_train_network.py`，参数目录通过 `mikazuki/catalog/inspector.py` 自动生成，中文说明和默认值位于 `mikazuki/catalog/overlays/krea2_lora.yaml`。未修改 vendored sd-scripts 的训练算法，未复制或运行 OneTrainer 的训练器。
