# LLVNR Kit Builder

这是 LLVNR Demo 的本地模型资产包工具链。它不下载 Apple 固件，不实现
AEA 解密，也不附带任何模型权重、tuning、`CameraSetup.plist` 或可运行的
`.llvnrkit`。所有生成物都必须留在使用者的私有目录。

## 工具

| 工具 | 作用 |
| --- | --- |
| `extract_ipsw_assets.py` | 调用本机 `ipsw` CLI，从用户提供的本地 IPSW/OTA 中严格提取 H19 `model.hwx` 和 `LLVTuningParameters.plist`。 |
| `recover_portable_sources.py` | 从本地提取目录中的 `model.hwx` 恢复 portable source graph 和系数；使用仓库内 sanitized synthetic layout metadata，不读取私有基准权重。 |
| `export_runtime_assets.py` | 从 portable source graph + 本地系数数组生成 `model.mil`、`weights.bin`、manifest，并执行 CPU/ANE 验证。ANEForge 编译/运行子集已随仓库提供，不再依赖私有 Python 包。 |
| `build_catalog.py` | 从提取目录生成 version-2 `catalog.json`，自动绑定 `ModelInfo.Name`、tuning、画布、模型 SHA-256 和运行资产。 |
| `build_kit.py` | 把本地运行资产和 catalog 打包成 `.llvnrkit`。 |
| `verify_kit.py` | 在导入前检查路径、ZIP 成员、符号链接、文件清单、哈希、模型身份和 catalog。 |
| `convert_local_model.py` | 一条命令串联 extract → recover → export → catalog → kit → verify。 |
| `export_research_assets.py` | 兼容旧私有研究树的桥接入口；新流程优先使用 `export_runtime_assets.py`。 |
| `schema/kit.schema.json` | `kit.json` 的结构说明。 |

## 快速流程

### 从本地提取目录直接生成 kit

```bash
python3 demo/kit/convert_local_model.py \
  --extracted /path/to/private/llvnr-extracted \
  --work /path/to/private/llvnr-work \
  --output /path/to/private/LLVNR-local.llvnrkit
```

也可以直接传本地 OTA：

```bash
python3 demo/kit/convert_local_model.py \
  --ota /path/to/local.aea \
  --models LLV-HD-Wide-2x-v0.3.0_2304x1296 \
  --work /path/to/private/llvnr-work \
  --output /path/to/private/LLVNR-local.llvnrkit
```

内部会先运行 `recover_portable_sources.py`，再调用 exporter。`--sources` 仍可用
于复用已经恢复好的私有图；`portable-sources/<MODEL_ID>/source_graph.json` 中的
系数路径必须相对于 source 目录、coefficient root 或 coefficient root 下的模型目录。
转换器会把它们规范化到私有输出树的 `coefficients/` 下，并重新计算哈希。

### 已有运行资产

```bash
python3 demo/kit/build_catalog.py \
  --extracted /path/to/private/llvnr-extracted \
  --assets /path/to/private/model-assets \
  --output /path/to/private/catalog

python3 demo/kit/build_kit.py \
  --assets /path/to/private/model-assets \
  --catalog /path/to/private/catalog \
  --output /path/to/private/LLVNR-local.llvnrkit \
  --license third_party/ANEForge/LICENSE

python3 demo/kit/verify_kit.py /path/to/private/LLVNR-local.llvnrkit
python3 demo/kit/verify_kit.py /path/to/private/LLVNR-local.llvnrkit.zip
```

也可以只从本地 OTA/IPSW 提取：

```bash
python3 demo/kit/extract_ipsw_assets.py /path/to/local.aea \
  --variant V63 \
  --output /path/to/private/llvnr-extracted
```

## 公开转换链的边界

公开恢复器使用本地 `model.hwx` 和仓库内的 sanitized structural metadata 生成
portable source graph；仓库不包含模型权重、探针 HWX、tuning 或现成图。exporter
只接受已经生成的 source graph，会拒绝缺失系数、错误哈希、未使用的输入端口、
缺失输出端口和不支持的画布。

`--validation ane` 会在 Apple Silicon 上编译并执行导出的 MIL，把 CPU 与
ANE 的 synthetic-input 结果写入每个模型的 `export_validation.json`。导出
manifest 还会记录本机检测到的 `target_family` / `target_arch`；App 会拒绝
与当前主机不匹配的新 manifest，避免把 host-target fallback 误当成跨代支持。
`--validation cpu` 只生成 CPU 报告，`build_kit.py` 会拒绝把它当作可运行模型；
`--validation none` 只用于结构调试。H13 之外的目标支持情况见
[`docs/ANE-SUPPORT.md`](../../docs/ANE-SUPPORT.md)。

## 输入要求

`--assets` 的每个模型目录必须包含：

- `manifest.json`
- `model.mil`
- `weights.bin`
- `source_graph.json`
- `export_validation.json`
- manifest 声明的 `precision/` 和 `evidence/` 文件

`--catalog` 必须包含 `catalog.json` 和它声明的 tuning 文件。builder 只复制
运行需要的文件，拒绝原始固件后缀，不复制稀疏 `.npy` 中间文件。

App 的 `File → 导入模型资产包…` 同时接受未解压目录和 ZIP。安装目录为：

```text
~/Library/Application Support/LLVNRDemo/CurrentModelKit
```

## 隐私

App 不上传模型、tuning、相机帧、视频或报告。运行报告的本地路径和分享前
脱敏方式见 [`PRIVACY.md`](../../PRIVACY.md)。发布源码或二进制前至少运行：

```bash
python3 tools/audit_release.py .
python3 tools/check_readme_links.py .
```

## 法律边界

builder 和 exporter 是本地研究工具，不是对 Apple 固件、模型权重、tuning
或衍生模型的再分发授权。生成的 kit 只能用于你有权使用的素材；Apache/MIT
等代码许可证不覆盖第三方模型数据。不要把生成的 `.llvnrkit` 上传到公开
仓库，除非你已单独确认相应素材的授权。
