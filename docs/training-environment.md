# 项目共享训练环境

2026-10-11 正式环境已迁移到 Windows x64 / Python 3.12.15 / RTX 5090（SM 12.0、32GB、驱动 610.62）。当前采用 PyTorch 2.14.1 + CUDA 13.0、torchvision 0.29.1 和独立 Flash Attention 2。xFormers 已从项目虚拟环境卸载，训练不依赖 Triton 或 torch.compile。

工作台及其训练子进程统一使用项目 `venv` 中的 Python 3.12.15，基础解释器位于持久的 `runtime/python/cpython-3.12.15-windows-x86_64-none`。2026-10-22 主环境升级为 Transformers 5.17、Diffusers 0.40、Hugging Face Hub 1.32、PEFT 0.19.1；Krea2、Anima、MOSS 共用这一环境，MOSS 不再选择额外解释器。2026-10-23 又将整个 `scripts/dev` 从 v0.12.0 升级为 v0.12.2，MOSS 与其他 sd-scripts 训练器共用本地源码及 library；来源记录在 `scripts/dev/UPSTREAM.json`。无需额外源码安装或动态加载。

2026-10-11 的 Python 迁移曾保持全部 105 个软件包版本一致；该历史对照与回退位置见 [Python 升级评估与迁移记录](python-upgrade-assessment.md)。本次升级前依赖清单保存在 `logs/shared-env-before-freeze.txt`，安装清单为 `logs/shared-env-install.json`。旧 MOSS 虚拟环境目录保留用于追溯，已不参与任务启动。

## 使用 Attention 选项

在工作台的「显存与性能 → Attention 实现/计算方式」中选择：

| 模型 | SDPA 选项 | Flash 选项 | 项目入口 |
| --- | --- | --- | --- |
| SDXL | `sdpa`（默认） | `flash` | `scripts/sdxl_train_network.py` |
| Anima | `torch`（默认），兼容 `sdpa` | `flash` | `scripts/anima_train_network.py` |
| Krea2 | `sdpa`（CLI 默认） | `flash`（工作台默认） | `scripts/krea2_train_network.py` |
| MOSS-Image | `sdpa`（默认） | 当前适配器不提供 | `scripts/moss_image_train_network.py` |

Flash 需要 FP16/BF16；Krea2 固定使用 BF16 计算，其「底模权重精度」可选 BF16、NF4 或 FP8 E4M3。FP8 用于冻结底模权重存储，LoRA 保持 FP32，矩阵计算为 BF16。详见 [Krea2 使用说明](krea2-training.md)。

未安装 xFormers 时，所有训练器相关下拉项和关闭状态的旧开关均不可开启；旧草稿中已开启的开关允许关闭。服务端也会拒绝仍指定缺失后端的配置。安装兼容 xFormers 后刷新页面即可恢复选项，包已安装不代表其 CUDA 内核适用于所有显卡。Flash 未安装时也会禁用；新草稿自动采用 SDPA/Torch，已有草稿不会被静默改写。

本机官方 Windows PyTorch wheel 没有内置 FlashAttention，选择 SDPA 不等于使用 Flash。项目的 `flash` 显式使用 `flash-attn` 前后向，并在加载大模型前做 GPU 训练内核检查，失败时明确报错。

## 安装或复现

先结束项目的训练/标注任务并停止开发服务，在项目根目录的 Python 3.12 虚拟环境中执行。跨 Python 版本需要重建 `venv`，不能仅替换解释器文件。CUDA 13.0 需要支持该运行时的 NVIDIA 驱动；Flash 需求文件会按 Python 版本选择 Windows x64 / Torch 2.14 的 cp312 或保留的 cp310 构建。

```powershell
.\venv\Scripts\python.exe -m pip freeze --all > logs\requirements-before-upgrade.txt
.\venv\Scripts\python.exe -m pip uninstall -y xformers
.\venv\Scripts\python.exe -m pip install -r requirements-cu130.txt
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install --no-deps -r requirements-flash-windows.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m mikazuki.scripts.check_training_environment --gpu --output logs/training-environment.json
```

Windows 安装脚本已采用相同配置。新建虚拟环境时优先使用项目已安装的持久 Python 3.12.15；`.python-version` 同样固定为 3.12.15。`torch`/`torchvision` 来自 [PyTorch CUDA 13.0 官方索引](https://download.pytorch.org/whl/cu130)。Flash wheel 来自 [mjun0812 的社区构建 v0.10.2](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.10.2)，基于 Dao-AILab Flash Attention 2.8.3 源码，包含适配新 Torch/MSVC 的构建补丁，并非 Dao-AILab 官方 Windows 二进制。需求文件固定 URL、版本和发布资产 SHA256，当前 cp312 构建为：

```text
090e78a61613e6c27a948b14808ba358fe77e250ff701b3c7fe1be42569e7134
```

| 组件 | 当前版本 |
| --- | --- |
| Python | 3.12.15 |
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| flash-attn | 2.8.3+cu130torch2.14 |
| xformers | 未安装，可选 |
| diffusers / transformers | 0.40.0 / 5.17.0 |
| accelerate / peft | 1.15.0 / 0.19.1 |
| bitsandbytes / safetensors | 0.50.2 / 0.8.0 |
| huggingface-hub | 1.32.0 |
| numpy / protobuf | 1.26.4 / 3.20.3 |

统一使用根目录 `requirements.txt`；NumPy 1.x、Protobuf 3 及原有 WebUI 依赖继续保留。不要直接用 vendored `scripts/dev/requirements.txt` 覆盖本项目环境。其他 Python/平台需自行选择匹配构建并做 GPU 验证；本次兼容结论限于上述机器。Transformers 5 的迁移约定见[官方说明](https://github.com/huggingface/transformers/blob/v5.17.0/MIGRATION_GUIDE_V5.md)。

## 本地源码集成验收（2026-10-23）

现有 `scripts/dev` 整体升级至 sd-scripts v0.12.2：新增 13 个文件、更新 54 个文件，251 个文件与所选上游源码一致。保留项目的 Flash Attention、Anima AdaLN 精度和工作台参数适配。
MOSS 入口直接导入本地 `scripts/dev/moss_image_train_network.py`，与 Anima、SDXL 共用同一个 `train_network` 模块及 `library`。已移除专用源码安装器、动态加载器和源码锁；安装项目依赖后即可运行。原 runtime 源码移至 `temp/upstream-integration/previous-moss-runtime` 作为历史副本，训练前已确认原路径不存在。

沿用 10 月 22 日的数据及配置，各做六步训练并保存预览：

| 模型 | 任务 | 结果 | 含缓存及预览耗时 | 平均 loss |
| --- | --- | --- | --- | --- |
| krea2 | job_20261023-012500-33f70d | 成功 | 91 秒 | 0.058678 |
| anima | job_20261023-012500-eed0c5 | 成功 | 43 秒 | 0.067801 |
| moss | job_20261023-012500-3d0771 | 成功 | 74 秒 | 0.362058 |

全部任务退出码为 0，LoRA 的 up 矩阵实际更新数分别为 224、280、252；六步 loss 和权重均为有限值。产物位于 `output/local-upstream-validation/{krea2,anima,moss}/`。各保存一张 512×512 预览；Krea2 仍是 4 步快速出图，不作为画质评估。六步短训验证接入、反向、保存及出图，不代表长期收敛。

同一个主环境进程运行 231 项回归：229 项通过，2 项因 Windows 符号链接权限跳过。MOSS 原生模型测试已纳入常规回归，无需 `MOSS_UPSTREAM_TESTS` 或独立测试进程。生成器成功导入全部十个训练器参数入口；上游新增的层交换参数已补齐中文说明，多行 Caption 参数由数据集配置管理。`pip check` 通过。

证据：`logs/local-upstream-validation.json`（完整任务、配置、指标和产物哈希）、`logs/local-upstream-tests.log`、`logs/local-upstream-imports.json`（实际模块路径及共享对象检查）、`logs/local-upstream-source-check.json`（251 个源码文件校验）。

## 共享环境验收（2026-10-22）

三种模型都通过工作台的编译和任务队列启动，命令中的解释器均为项目 `venv/Scripts/python.exe`。使用 `train/bi` 六张图片、1024 分辨率预算、Batch 1、学习率 1e-4、AdamW8bit、梯度检查点、随机种子 1337，各训练六步并保存一张 512×512 预览。

| 模型 | 完整权重与训练配置 | 结果 | 含缓存和预览的任务耗时 | 平均 loss |
| --- | --- | --- | --- | --- |
| Krea2 | models/Krea-2-Raw，FP8 底模 / BF16 计算，Flash，Rank/Alpha 16 | 成功，224 个 LoRA up 矩阵已更新 | 97 秒 | 0.058680 |
| Anima | 本机 anima-base-v1.0、Qwen3-0.6B、Qwen-Image VAE，BF16，Flash，Rank/Alpha 8 | 成功，280 个 LoRA up 矩阵已更新 | 42 秒 | 0.067809 |
| MOSS | models/MOSS-Image-Base，BF16，SDPA，Rank/Alpha 16 | 成功，252 个 LoRA up 矩阵已更新 | 69 秒 | 0.362072 |

Anima 完整组件从 `D:/AICreate/AiPicture/SD-Trainer-v2.9.1/SD-Trainer/sd-models/anima/` 读取，文件名分别为 `anima-base-v1.0.safetensors`、`qwen_3_06b_base.safetensors`、`qwen_image_vae.safetensors`；没有修改或复制这些底模。该次测试补齐了此前只有 Anima 小模型验证的范围。

三个任务 ID 依次为 `job_20261022-171800-fc19c9`、`job_20261022-171800-dada54`、`job_20261022-171800-c3e615`，退出码均为 0。产物分别位于 `output/shared-env-validation/krea2/`、`anima/` 和 `moss/`，包含 LoRA、训练状态及 sample/ 预览。Krea2 为了快速检查使用 4 个推理步，预览明显偏模糊，不能作为质量样张；Anima 使用 20 步、CFG 4，MOSS 使用 50 步、CFG 3。不同模型的 loss 和耗时不用于横向质量判断，六步也不验证长期收敛。

主环境回归运行 230 项：225 项通过，2 项因 Windows 符号链接权限跳过，3 项 MOSS 测试在同一主环境的新进程中补跑并全部通过，共计 228 项通过、2 项权限跳过。包括 Krea2 的真实编码器缓存、NF4/FP8 反向、LoRA 保存重载和小模型续训，Anima 的完整小型 DiT 反向与保存重载，以及 MOSS 原生目标层、冻结底模、梯度检查点和保存重载。原 Krea2 测试中针对 Transformers 4 的 RoPE 配置转换已改为兼容新版直接保存的字段。

`pip check`、十个训练器入口导入、Flash/SDPA/AdamW8bit GPU 检查和重启后的前后端健康检查均通过。其他训练器只有入口和既有回归覆盖，并未逐一做完整底模短训；本次不代表所有高级参数组合都已验证。

证据汇总：`logs/shared-env-validation.json`（任务、指标、包版本、配置、模型/产物哈希）；基础环境检查：`logs/shared-env-check.json`；回归日志：`logs/shared-env-backend-tests.log`、`logs/shared-env-moss-tests.log`。升级安装没有修改 Torch、CUDA、Flash Attention 或其他应用环境。

复现训练可通过任务中心复用对应配置，或在项目根目录执行：

```powershell
.\venv\Scripts\python.exe -m accelerate.commands.launch --num_cpu_threads_per_process 8 --num_processes 1 scripts/krea2_train_network.py --config_file config/runs/20261022-171800-fc19c9/train.toml
.\venv\Scripts\python.exe -m accelerate.commands.launch --num_cpu_threads_per_process 8 --num_processes 1 scripts/anima_train_network.py --config_file config/runs/20261022-171800-dada54/train.toml
.\venv\Scripts\python.exe -m accelerate.commands.launch --num_cpu_threads_per_process 8 --num_processes 1 scripts/moss_image_train_network.py --config_file config/runs/20261022-171800-c3e615/train.toml
```

如需保留本次验收产物，重跑前先在配置里更换输出目录及输出名称。

## 历史验证范围（2026-10-11）

| 检查 | 结果及范围 |
| --- | --- |
| 环境依赖 | `pip check` 通过，九个训练器导入和真实参数解析通过 |
| SDXL | SDPA/Flash 自注意力与交叉注意力输出、输入梯度对比；梯度检查点下实际 LoRA 连续更新、冻结底模保持不变 |
| Anima | 小规模完整 DiT，SDPA/Flash 反向与 LoRA 更新、梯度检查点、Safetensors 保存及重载 |
| Krea2 | 小型真实模型 NF4/FP8、缓存、掩码 GQA、LoRA 保存重载、SDPA 精确续训、采样；完整用户底模 FP8 + Flash 六步训练和 CFG 预览 |
| 其他训练路径 | 九个入口导入，FLUX 小型真实模型 SDPA 反向，AdamW8bit 更新；未进行其他完整底模训练 |
| 回归 | 后端 219 项，217 通过、2 项既有条件跳过；前端 8 项、TypeScript 检查和 Vite 构建通过 |
| 界面 | Krea2 的 Flash、FP8 可选，xFormers 显示「未安装」且禁用，沿用现有组件和样式 |

Anima 测试暴露出 AdaLN 中 FP32 时间嵌入与 BF16 权重直接相乘的问题。项目入口将这些小型调制投影的计算保持为 FP32，保留权重存储精度和 LoRA 参数名；修复对 SDPA 和 Flash 都生效。SDXL 入口只替换原有 CrossAttention 的计算方法，保持参数路径和 LoRA 保存格式。没有修改 vendored sd-scripts。

本地仅提供了 Krea2 完整底模，SDXL、Anima 目前通过真实小模型/模块训练验证，尚未完成各自完整预训练权重的端到端短训。短训通过也不代表验证了长期收敛、全部优化器或所有高级组合。

复现回归：

```powershell
$env:KREA2_GPU_TESTS = '1'
.\venv\Scripts\python.exe -m unittest discover -s mikazuki/tests -p 'test_*.py'
Remove-Item Env:KREA2_GPU_TESTS
.\venv\Scripts\python.exe -m mikazuki.scripts.benchmark_krea2_attention --output logs/attention-benchmark.json
```

当前 Python 3.12 环境证据：`logs/python312-migration/environment.json`；汇总：`logs/python312-migration/validation.json`。正式环境重新通过 219 项回归及 21 项页面/API 检查，并通过工作台实际任务入口完成 Krea2 FP8 + Flash 六步短训，LoRA、训练状态和 30 个指标点均验证通过。此前 Torch/Flash 升级证据保留在 `logs/attention-upgrade-validation.json`，界面截图为 `logs/attention-upgrade-ui.png`。

## 性能与版本取舍

2026-10-10 升级 Torch/Flash 时的同机 BF16 GQA 基准（Python 3.10、Batch 1、4096 Token、48 个 Q 头、12 个 K/V 头、Head Dim 128，前向加反向）：Torch SDPA efficient attention 为 37.20ms，独立 Flash 为 10.99ms，二者峰值 allocated 均约 0.59GiB。Profiler 确认 `flash_attn::_flash_attn_forward` 和 `_flash_attn_backward`，见 `logs/krea2-attention-torch214-flash.json`。这约 3.38 倍的算子差异不能套用于整个训练。

完整 Krea2 FP8 + Flash 六步短训，在 880×1168、Batch 1、Rank 16 下，第 2–6 步平均 3.05 秒，峰值 allocated 16.04GiB（包括一次 512×512、4 步 CFG 预览）。此前 Torch 2.7 / xFormers 0.0.30 的同类短训约 2.66 秒/步；升级目标是移除 xFormers 后兼容三项重点训练，不能声称新版在所有任务上更快。不同运行的 GPU 状态也影响耗时。

历史候选 xFormers 0.0.35 和用户指定的 0.0.35.dev1147 均要求 Torch 2.10+。dev1147 改为转发 MSLK；实测官方 Windows MSLK 1.3.0+cu130 配合 Triton 后 GQA 推理可用，但 RTX 5090 训练缺少反向算子，FA2 也未构建，因此没有采用。报告 `logs/krea2-dev1147-validation.json` 保留这次独立候选测试。此结论不外推到 Linux 或自行构建的包。
