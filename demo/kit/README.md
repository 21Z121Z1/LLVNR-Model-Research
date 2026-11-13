# LLVNR Kit Builder

这是 LLVNR Demo 的本地模型资产包工具。它不下载 Apple 固件，不提取/解密固件，也不附带任何模型权重。

## 工具

- `export_research_assets.py`：调用你本地的研究导出器，生成 portable runtime assets。默认模型集是本 Demo 的 12 个 V63 后摄配置。
- `extract_ipsw_assets.py`：调用本机 `ipsw` CLI，从用户提供的本地 IPSW/OTA 中提取 `H19` 的 `model.hwx` 和 `LLVTuningParameters.plist`。它不内置密钥、不下载固件，也不把产物上传到网络。
- `build_kit.py`：把本地运行资产和 catalog 打包成 `.llvnrkit`。
- `verify_kit.py`：在导入前检查目录或 ZIP 的路径、符号链接、文件清单、哈希、模型身份和 catalog。
- `schema/kit.schema.json`：`kit.json` 的结构说明。

## 快速流程

```bash
# 可选：从本机 IPSW/OTA 提取模型和 tuning 到私有目录
python3 extract_ipsw_assets.py /path/to/local.aea \
  --variant V63 --output /path/to/private/llvnr-extracted

# 可选：在你的私有研究树中生成运行资产
python3 export_research_assets.py \
  --research-root /path/to/private/llvnr_port \
  --assets-out /path/to/private/model-assets

# 打包；默认同时生成同名 .zip
python3 build_kit.py \
  --assets /path/to/private/model-assets \
  --catalog /path/to/private/catalog \
  --output /path/to/private/LLVNR-local.llvnrkit \
  --license /path/to/LICENSE-ANEForge.txt

# 导入前验证
python3 verify_kit.py /path/to/private/LLVNR-local.llvnrkit
python3 verify_kit.py /path/to/private/LLVNR-local.llvnrkit.zip
```

App 的 `File → 导入模型资产包…` 同时接受未解压目录和 ZIP。安装目录为：

```text
~/Library/Application Support/LLVNRDemo/CurrentModelKit
```

## 输入要求

`--assets` 的每个模型目录必须包含：

- `manifest.json`
- `model.mil`
- `weights.bin`
- `source_graph.json`
- `export_validation.json`
- manifest 声明的 `precision/` 和 `evidence/` 文件

`--catalog` 必须包含 `catalog.json` 和它声明的 tuning 文件。builder 只复制运行需要的文件，拒绝原始固件后缀，不复制稀疏 `.npy` 中间文件。

## 法律边界

builder 是本地打包工具，不是对 Apple 固件、模型权重、tuning 或衍生模型的再分发授权。生成的 kit 只能用于你有权使用的素材；Apache/MIT 等代码许可证不覆盖第三方模型数据。不要把生成的 `.llvnrkit` 上传到公开仓库，除非你已单独确认相应素材的授权。
