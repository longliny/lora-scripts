# Python 版本升级评估

2026-10-11 在本机 Windows x64、Ryzen 9 9950X3D、RTX 5090 32GB 上完成隔离对照，随后按用户要求完成正式迁移。**当前项目和工作台均使用 Python 3.12.15**，旧的 3.10.20 环境保留用于回退，系统 Python 未改变。下面保留升级前的对照数据，正式迁移记录见后文。

**本次迁移用于获得后续维护支持，实测没有训练提速收益。** 升级前相同依赖下，完整 Krea2 FP8 + Flash 短训从平均 2.53 秒/步变为 2.58 秒/步，未观察到提速。SDXL、Anima 的小模型/模块训练和项目回归在候选环境和正式环境中均通过检查。

## 选择 3.12 的依据

| 版本 | 维护状态与项目适配 | 判断 |
| --- | --- | --- |
| 3.10.20（迁移前） | Python 3.10 已于 2026-10-01 停止维护；已有训练仍可运行 | 保留环境备份用于回退 |
| 3.11 | 安全维护至 2027-10，比 3.12 少一年 | 本次没有另建环境测试 |
| 3.12.15（当前） | 安全维护至 2028-10；现有 105 个软件包均能保持原版本安装 | 已验证并迁移到正式环境 |
| 3.13 及以上 | 当前 NumPy 1.26.4 仅支持 Python 3.9–3.12，需要另外评估 NumPy 2 及相关依赖迁移 | 不作为这次只升级 Python 的目标 |

维护状态来自 [Python 官方版本表](https://devguide.python.org/versions/)。NumPy 上限见 [NumPy 1.26.4 发布说明](https://numpy.org/doc/stable/release/1.26.4-notes.html)。这不意味着新 Python 无法训练，而是它不适合保持本项目现有依赖组合不变的迁移。

[Python 3.12.15](https://www.python.org/downloads/release/python-31215/) 是 2026-09-30 发布的安全更新。官方对 3.12 的新补丁只提供源码；本次使用 uv 管理的 Astral 便携构建，没有安装系统级 Python 或修改系统 PATH。候选环境位于 `temp/python-upgrade/`；正式迁移另行安装持久解释器，并在原 `venv` 路径重新生成环境与命令启动器。

## 兼容性结果

两个环境均使用 Torch 2.14.1+cu130、torchvision 0.29.1+cu130、Flash Attention 2.8.3+cu130torch2.14、Diffusers 0.39.0、Transformers 4.57.6、bitsandbytes 0.50.2、NumPy 1.26.4。全部 **105 个已安装软件包的版本一致**，仅使用对应 Python ABI 的二进制构建；均未安装 xFormers。

| 检查 | Python 3.12.15 结果 |
| --- | --- |
| 依赖 | `pip check` 通过 |
| 训练入口 | 九个训练器导入及真实参数解析通过 |
| GPU 基础 | 独立 Flash Attention 前后向、SDPA、AdamW8bit 更新通过 |
| SDXL | 实际注意力模块的自注意力/交叉注意力、Flash 梯度、梯度检查点和 LoRA 更新通过 |
| Anima | 小规模完整 DiT 的 Flash/LoRA 训练、梯度检查点、保存与重载通过 |
| Krea2 | NF4/FP8 小模型回归通过；用户提供的完整底模和训练集在 FP8 + Flash 下两轮各 12 步训练通过 |
| 项目回归 | 219 项：217 项通过，2 项按原有条件跳过 |

SDXL、Anima 未提供完整预训练权重用于本次对照，因此不将小模型通过等同于完整底模端到端训练验证。其余模型只完成入口检查及现有回归覆盖范围，未逐一做完整训练。短训也不能验证长期收敛。

Flash 使用与当前环境相同的 [mjun0812 v0.10.2 社区构建](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.10.2)，对应 Python 3.12 的文件为 `flash_attn-2.8.3+cu130torch2.14-cp312-cp312-win_amd64.whl`，发布资产 SHA256 为：

```text
090e78a61613e6c27a948b14808ba358fe77e250ff701b3c7fe1be42569e7134
```

## 训练速度对照

使用 `models/Krea-2-Raw` 和 `train/bi` 中的 6 张图片。分桶后的实际尺寸均为 **880×1168**，Batch 1、LoRA Rank/Alpha 16、FP8 冻结底模、BF16 计算、Flash Attention、AdamW8bit、梯度检查点、种子 1337。两侧 CPU 线程数均设置为 8；使用同一份已生成的 latent/text 缓存。

按 **3.10 → 3.12 → 3.12 → 3.10** 顺序串行运行，每轮 12 步，去掉前 2 步，保留各版本 20 个稳定步样本。每轮启动前均检查项目训练、标注和 ComfyUI 队列空闲。按用户已有授权在 ComfyUI 空闲时卸载了其驻留模型，没有关闭 ComfyUI。

| Python | 第一次平均步耗时 | 第二次平均步耗时 | 合并平均步耗时 | 峰值 allocated / reserved |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.523 秒 | 2.544 秒 | **2.534 秒** | 16.04 / 17.10 GiB |
| 3.12.15 | 2.587 秒 | 2.577 秒 | **2.582 秒** | 16.04 / 17.10 GiB |

本次候选环境平均步耗时增加 **1.90%**，没有显示出训练提速；不据此断言所有训练配置或长期任务都会慢同样比例。Python 升级不会自动改变相同版本 CUDA 算子的计算能力，可能受益的是 Python 调度和数据处理部分。[Faster CPython 的标准基准提速](https://docs.python.org/3.11/whatsnew/3.11.html#faster-cpython)不能直接套用为 GPU 训练提速比例。

计时采用训练器原有 `step_seconds`，包含训练前后向和优化步骤的主机侧计时，不包含模型加载、FP8 量化和大部分数据加载；没有额外插入每步结束的 CUDA 同步。整次任务计时另行保留：3.10 两轮为 94.43 / 67.58 秒，3.12 为 71.15 / 69.34 秒。首轮加载受操作系统文件缓存影响明显，不能将这些总时长直接解释为 Python 升级收益。

四轮均正常保存 LoRA，每份 448 个 FP32 张量全部有限，224 个原始零初始化的 LoRA B 张量均已有非零更新。各轮平均损失约 0.0642；这说明短训正常完成，不代表长期质量验证。

## 正式迁移记录

2026-10-11 08:54（北京时间）完成服务切换。先在服务运行期间完成持久解释器安装和离线依赖预检，再在训练、标注队列为空时停止本项目开发服务，备份旧环境并重建正式 `venv`，最后重启前后端。模型、数据、已有训练产物和系统 Python 均保持原位。

| 项目 | 迁移结果 |
| --- | --- |
| 主解释器 | `venv/Scripts/python.exe`，Python 3.12.15 |
| 基础解释器 | `runtime/python/cpython-3.12.15-windows-x86_64-none/python.exe` |
| 旧环境备份 | `runtime/backups/venv-py310-20261011-085332` |
| 安装配置 | `.python-version` 固定 3.12.15，Flash 增加带 SHA256 的 cp312 分支，Windows 安装脚本优先选择项目持久解释器 |
| 依赖 | 105 个软件包版本与已验证候选一致；`pip check` 通过 |
| 工作台 | 8004 前端代理与 28002 后端均报告 Python 3.12.15；21 项页面/API 检查通过 |
| 正式环境回归 | 219 项：217 通过，2 项条件跳过；九个训练入口导入和 GPU 检查通过 |
| 实际任务 | `job_20261011-085713-58a30b`，通过工作台 → Accelerate → 项目 Krea2 入口完成 FP8 + Flash 六步训练 |
| 保存与指标 | 448 个 LoRA 张量全部有限、224 个 LoRA B 张量均有更新；训练状态已保存且权重哈希匹配；5 组指标共 30 个点正常读取 |

完整验收记录为 `logs/python312-migration/validation.json`，任务产物为 `output/krea2-validation/python312-migration/`。该次验收第 2–6 步均值为 2.48 秒，峰值 allocated 16.09 GiB；这是部署后的功能短训，不能与前面不同时间运行的对照直接比较为升级收益。

需要回退时，应先在项目队列空闲后停止开发服务，将当前 `venv` 另存备份，再把上述 3.10 备份恢复到原 `venv` 路径并重新启动。旧环境的命令启动器包含原路径，不能直接把备份目录作为日常启动目录。原 Python 3.10 基础解释器仍保留在 `runtime/python/` 中。

正式环境不依赖 `temp/python-upgrade/` 或 `temp/python-migration/`。依赖快照为 `runtime/installed-requirements.txt`，保持版本的安装清单为 `runtime/python312-dependency-lock.txt`，Torch/Flash 的具体构建仍由根目录对应需求文件指定。未来进一步提速需要针对具体模型测量 Attention、缓存、精度和批量设置的收益。

## 本地证据与复查

| 文件 | 内容 |
| --- | --- |
| `logs/python-upgrade-assessment.json` | 汇总、性能统计、验证范围和 LoRA 文件检查 |
| `logs/python-upgrade-package-comparison.json` | 两个环境的完整版本清单，版本差异为空 |
| `logs/python312-environment.json` | pip、九个训练入口和 GPU 检查 |
| `logs/python-upgrade-benchmark.json` | 四轮实际命令、逐步耗时、整程耗时、队列及显卡状态 |
| `temp/python-upgrade/tests312.log` | 219 项回归完整输出 |
| `temp/python-upgrade/compare.py` | 串行对照脚本；已有输出时拒绝覆盖 |
| `output/krea2-validation/python-comparison/` | 四轮完整模型短训输出 |

在项目根目录复查当前正式环境：

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m mikazuki.scripts.check_training_environment --gpu --output logs/training-environment-current.json
$env:KREA2_GPU_TESTS = '1'
.\venv\Scripts\python.exe -m unittest discover -s mikazuki/tests -p 'test_*.py'
Remove-Item Env:KREA2_GPU_TESTS
```

GPU 检查应在其他 GPU 任务空闲时执行。测试结果与源代码对应本次工作区状态；临时环境清理后仍可使用正式环境进行上述检查，重跑旧版对照则需重新准备隔离环境。
