# sd-scripts 本地集成源码

本目录同步自 [kohya-ss/sd-scripts v0.12.2](https://github.com/kohya-ss/sd-scripts/commit/11d8cedd0c67b93a10f8b2d75f43554f6f12c794)。
2026-10-23 在现有目录内从 v0.12.0 升级。来源记录见 `UPSTREAM.json`，参数清单生成器将其写入工作台 manifest。

## 同步范围

- 同步前逐文件对照官方旧提交 `690ea7f96c23182352ec63def76d431c6120bd2f`，规范化 CRLF 后无本地修改。
- 新增 13 个文件，更新 54 个文件，没有删除上游文件。
- 沿用原 vendoring 范围，排除 `.ai/`、`.github/`、`tests/`，不纳入 Python 字节码。
- 上游代码原样集成；项目 Flash Attention、Anima AdaLN 精度和 MOSS 界面参数适配仍在 `mikazuki/training`。
- MOSS、Anima、SDXL 等直接导入这里的 `library`、`networks` 和 `train_network`，不再另装 MOSS 源码或切换同名模块。
- 旧 `runtime/moss-sd-scripts-v0.12.2` 已移至忽略的 `temp/upstream-integration/previous-moss-runtime`，仅供追溯，不参与执行。

## 依赖与验证

实际环境由根目录 `requirements.txt` 与 `requirements-cu130.txt` 管理，勿用本目录的上游依赖文件覆盖项目配置。
三种模型的完整底模测试及验证范围见[项目训练环境说明](../../docs/training-environment.md)。

重新生成参数清单：

```powershell
.\venv\Scripts\python.exe -m mikazuki.catalog.inspector generate --scripts-root scripts/dev
```

后续同步前先核对当前源码与 `UPSTREAM.json` 所指版本，保留项目适配，再更新来源与 manifest。
