# SDXL、Anima、Krea2 训练环境

2026-10-11 正式环境已迁移到 Windows x64 / Python 3.12.15 / RTX 5090（SM 12.0、32GB、驱动 610.62）。当前采用 PyTorch 2.14.1 + CUDA 13.0、torchvision 0.29.1 和独立 Flash Attention 2。xFormers 已从项目虚拟环境卸载，训练不依赖 Triton 或 torch.compile。

工作台及其训练子进程均已使用项目 `venv` 中的 Python 3.12.15，基础解释器位于持久的 `runtime/python/cpython-3.12.15-windows-x86_64-none`，不依赖临时测试目录。迁移保持全部 105 个软件包版本一致；升级前的性能对照、正式验收和旧环境回退位置见 [Python 升级评估与迁移记录](python-upgrade-assessment.md)。

## 使用 Attention 选项

在工作台的「显存与性能 → Attention 实现/计算方式」中选择：

| 模型 | SDPA 选项 | Flash 选项 | 项目入口 |
| --- | --- | --- | --- |
| SDXL | `sdpa`（默认） | `flash` | `scripts/sdxl_train_network.py` |
| Anima | `torch`（默认），兼容 `sdpa` | `flash` | `scripts/anima_train_network.py` |
| Krea2 | `sdpa`（CLI 默认） | `flash`（工作台默认） | `scripts/krea2_train_network.py` |

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
| diffusers / transformers | 0.39.0 / 4.57.6 |
| accelerate / peft | 1.15.0 / 0.18.1 |
| bitsandbytes / safetensors | 0.50.2 / 0.8.0 |
| huggingface-hub | 0.36.2 |
| numpy / protobuf | 1.26.4 / 3.20.3 |

保留 Transformers 4、Hub 0.x、NumPy 1.x 和 Protobuf 3，避免同时迁移现有训练器与 Caption 依赖。不要直接用 vendored `scripts/dev/requirements.txt` 覆盖本项目环境。其他 Python/平台需自行选择匹配构建，并用同一 GPU 检查脚本验证；本次兼容结论限于上述机器。

## 已验证范围

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
