# MOSS-Image LoRA 训练

在现有训练工作台选择 **MOSS-Image LoRA**。训练、缓存、LoRA 保存及采样使用项目 `scripts/dev` 中的 **kohya-ss/sd-scripts v0.12.2**；与 Anima、SDXL 等训练器共用同一套本地源码，项目负责界面参数转换和任务管理。MOSS、Krea2 和 Anima 共用升级后的项目 `venv`，无需额外环境。

## 安装或升级主环境

先结束训练/标注任务并停止项目后端，在项目根目录的 PowerShell 中运行。首次安装可直接执行 `scripts/install.ps1`；已有主环境可以运行以下兼容入口：

```powershell
.\scripts\setup_moss.ps1
# 如需代理：
.\scripts\setup_moss.ps1 -Proxy http://127.0.0.1:10808
```

脚本默认升级 `venv/Scripts/python.exe` 所属的主环境，也可用 `-Python` 指定项目使用的环境。它仅安装统一依赖，源码已随项目提供，不再下载另一份训练代码。安装后应输出 `Local MOSS sd-scripts ready`、`scripts/dev` 中的文件路径和 `CUDA: True`。升级完成后重新启动后端，使所有训练任务使用新依赖。

上游源码已整体合并到 `scripts/dev`，固定提交 `11d8cedd0c67b93a10f8b2d75f43554f6f12c794`。版本来源见 `scripts/dev/UPSTREAM.json`。MOSS 的项目入口和其他训练器一样直接导入本地模块，不再使用独立 runtime 目录、动态加载器、专用源码安装器或启动时源码哈希锁。

当前验证组合：Windows、Python 3.12.15、Torch 2.14.1+cu130、Transformers 5.17.0、Diffusers 0.40.0、PEFT 0.19.1、Accelerate 1.15.0、Hugging Face Hub 1.32.0。统一依赖位于 `requirements-cu130.txt` 与 `requirements.txt`。

工作台与所有训练器使用同一个 Python 解释器（正常启动时为 `venv/Scripts/python.exe`）。旧的 `MIKAZUKI_MOSS_PYTHON` 不再生效。之前的 `.venv-moss` 和 `.venv-moss-v0122` 目录不再用于启动任务；本次保留这些目录作为历史环境。Linux 未在本次验证。

## 从工作台训练

1. 选择 **MOSS-Image LoRA**。
2. 底模选择完整的 Base 或 SFT Diffusers 目录；适配器自动定位 transformer、text_encoder、vae 的首个 safetensors 分片及 tokenizer 子目录。也可填 Hugging Face 仓库 ID，下载发生在启动后。使用 ComfyUI 单文件底模时，另填 Qwen3.5 编码器、FLUX.2 VAE，离线运行还需本地 Tokenizer。
3. 选择图片与同名 `.txt` Caption 目录。Caption 保留换行；固定文本缓存要求关闭打乱。分辨率及 Bucket 步长使用 16 的倍数。
4. 首次保留 Batch 1、Rank/Alpha 16、BF16、SDPA、AdamW8bit、梯度检查点。短训时清空“最大 Epoch”，最大步数设为 6。
5. 可开启采样，默认 50 步、CFG 3；短训使用 512×512。专家模式启用“训练前采样”可保存更新权重前的基线图。
6. 点击“检查配置”，确认启动命令使用项目 `venv`，再开始训练。任务中心显示状态、日志和上游 TensorBoard 指标。

文本默认上限 512 Token，专家模式最多 4096。缓存默认放内存；启用磁盘缓存后，上游在数据图片旁写入 `_moss.npz` / `_moss_te.npz`。更换模型、图片、Caption 或 Token 上限时应重建相关缓存。

重跑已生成的工作台配置：

```powershell
.\venv\Scripts\python.exe scripts/moss_image_train_network.py --config_file config/runs/<运行编号>/train.toml
```

启动时在相邻位置额外保存 `train.upstream.toml`，其中是已解析组件路径及 LoRA 目标层的上游原生配置。

## 从旧版迁移

| 项目 | v0.12.2 行为 |
| --- | --- |
| 训练环境 | 升级项目 venv，三种训练器共用 requirements.txt |
| 旧草稿 / TOML | moss.lora 映射到 networks.lora_generic，时间步 moss 映射到 sigma |
| vae_tiling / training_cache_dir | 移除；改用上游 VAE 与缓存实现 |
| 旧 .moss-cache | 不复用 |
| 旧 state.pt 状态目录 | 不兼容，预检拒绝；不能恢复旧优化器进度 |
| 旧 LoRA 权重 | 可作为“初始 LoRA”加载，训练进度重新开始 |
| 上游状态目录 | 使用“继续训练状态”恢复，适配器启用数据步数跳过 |

默认训练采样为上游 sigma + logit_normal，均值 -1.09、标准差 0.95、Shift 1。它不包含旧实现的 0.95 截断和 1% 均匀混合。上游 FLUX.2 VAE 使用均值编码，旧实现采用采样编码并支持分块；两次短训 loss 不能用来直接判断哪版更好。

## 保存与支持范围

LoRA 主文件为 `<输出目录>/<输出名称>.safetensors`，采用上游 networks.lora_generic 格式，FP32 保存。定期检查点、Accelerate 状态目录及 sample/ 预览也在输出目录中。`<输出名称>-result.json` 记录上游版本、配置路径、训练调用总耗时、整个调用的 CUDA allocated / reserved 峰值；逐步 loss 从任务中心事件读取。峰值包含缓存与采样，不等同于整张显卡所有应用的占用。

当前界面支持单 GPU 的 DiT LoRA，只训练主干 42 层的注意力投影，或同时训练 MLP。文本细化层、时间调制、输出头、Qwen3.5 和 VAE 冻结。专家模式提供 ConvRot INT8 和 Block Swap 参数；本次完整模型验收使用 BF16，未验证 INT8、Block Swap 或最低显存需求。INT8 融合算子需要兼容的 Triton；缺失时上游使用解量化回退。W4A8 GPTQ 不在本次支持范围。

采样使用上游 Euler、50 步、CFG 3、Shift 5，负向分支跳过第 22 层。上游与旧采样器的随机数生成方式不同；同种子的新旧预览不保证相同构图。权重对照应把新旧 LoRA 放进同一个上游推理流程。完整 BF16 训练与续训不承诺逐 bit 一致。

## 验证

```powershell
# 主环境：配置及项目回归
.\venv\Scripts\python.exe -m unittest discover -s mikazuki/tests -t . -v
# MOSS 已包含在同一次回归中，无需专用开关或另起测试进程。
.\venv\Scripts\python.exe -m pip check
```

2026-10-23 本地源码集成：231 项回归中 229 项通过、2 项权限跳过；三种完整底模短训均通过，MOSS 任务为 `job_20261023-012500-3d0771`。详见[最新验收记录](training-environment.md#本地源码集成验收2026-10-23)。

以下为初次迁移的历史记录。2026-10-22：后端 230 项中 218 项通过、12 项条件跳过；隔离环境上游测试 3 项通过；前端 8 项测试、类型检查及生产构建通过；pip check 无依赖冲突。日志为 `logs/moss-v0122-backend-tests.log`、`logs/moss-v0122-upstream-tests.log`、`logs/moss-v0122-frontend-tests.log` 和 `logs/moss-v0122-frontend-build.log`。

完整模型短训沿用 `models/MOSS-Image-Base` 和 `train/bi` 的六张图片：RTX 5090 32GB、1024 面积预算、实际 Bucket 880×1168、Batch 1、Rank/Alpha 16、BF16、SDPA、AdamW8bit、学习率 1e-4、梯度检查点、种子 1337。旧版 2026-10-18 验收保留在 `logs/moss-validation.json` 和 `output/moss-validation/bf16/`。

共享主环境的最新验收见[训练环境说明](training-environment.md)。下面保留迁移到上游时的独立环境历史结果，供追溯。

### v0.12.2 初次迁移历史结果

| 项目 | 实测 |
| --- | --- |
| 工作台主任务 | job_20261022-165334-d052d6，成功，退出码 0 |
| 6 步任务总耗时 | 88 秒，包含首次缓存、训练前及训练后两张预览 |
| 上游训练调用耗时 | 75.20 秒 |
| 全流程峰值 allocated / reserved | 14.87 / 14.90 GiB |
| 六步平均 loss | 0.362081，全部有限 |
| LoRA | 37,158,912 个可训练参数，756 个 FP32 张量，252 个 up 矩阵均已更新 |
| 任务中心指标 | 4 项、19 个点 |
| 第 3 步状态恢复 | job_20261022-165634-c3442b，成功，仅执行第 4–6 步，耗时 51 秒 |
| 独立出图 | 底模、旧 LoRA、新 LoRA 均由上游推理器成功生成 512×512 图片 |

主权重：`output/moss-validation/v0122/moss-v0122-bf16-smoke.safetensors`。续训结果在 `output/moss-validation/v0122-resume/`。恢复后首步 loss 与原运行一致，后两步不同，最大绝对差 0.07770；确认状态及进度可恢复，不宣称精确复现。

对照图位于 `output/moss-validation/v0122-comparison/`，三个子目录分别为 base、legacy、upstream。统一使用未修改的上游 `moss_image_minimal_inference.py`，种子 1337、50 步、CFG 3、Shift 5、Token 上限 512、LoRA 强度 1。三张图片构图相近，脸部和明暗有轻微变化，没有出现黑图或明显破损。六步不足以判断风格学习效果或质量优劣。

完整验收记录为 `logs/moss-v0122-validation.json`，包含任务、逐步 loss、环境、数据文件哈希及权重/图片 SHA256。工作台截图为 `logs/moss-v0122-ui.png`；推理日志为 `logs/moss-v0122-inference-{base,legacy,upstream}.log`。旧版训练耗时包含一张预览，而新版包含两张，且训练算法不同，不能直接比较端到端速度和 loss。

复现新 LoRA 的同参数出图（项目根目录 PowerShell）：

```powershell
.\venv\Scripts\python.exe scripts/dev/moss_image_minimal_inference.py `
  --dit models/MOSS-Image-Base/transformer/diffusion_pytorch_model-00001-of-00003.safetensors `
  --text_encoder models/MOSS-Image-Base/text_encoder/model-00001-of-00004.safetensors `
  --vae models/MOSS-Image-Base/vae/diffusion_pytorch_model.safetensors `
  --tokenizer_cache_dir models/MOSS-Image-Base/tokenizer --max_token_length 512 `
  --lora_weight output/moss-validation/v0122/moss-v0122-bf16-smoke.safetensors --lora_multiplier 1 `
  --prompt 'A cinematic portrait of a young adult woman, natural light, detailed eyes, realistic photography.' `
  --image_size 512 512 --seed 1337 --infer_steps 50 --guidance_scale 3 --flow_shift 5 `
  --device cuda --attn_mode torch --save_path output/moss-validation/v0122-comparison/replay
```

将 LoRA 路径替换为 `output/moss-validation/bf16/moss-bf16-smoke.safetensors` 可复现旧权重；去掉 `--lora_weight` 和 `--lora_multiplier` 可生成底模图。短训仅验证接入、保存、加载和出图，不能代表长期收敛或成品质量。

## 来源

- [sd-scripts v0.12.2 发布说明](https://github.com/kohya-ss/sd-scripts/releases/tag/v0.12.2)
- [固定版本的上游训练器](https://github.com/kohya-ss/sd-scripts/blob/11d8cedd0c67b93a10f8b2d75f43554f6f12c794/moss_image_train_network.py)
- [官方代码与技术报告](https://github.com/OpenMOSS/MOSS-Image)
- [官方 Base 模型](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
