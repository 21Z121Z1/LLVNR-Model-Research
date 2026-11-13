# LLVNR Demo 0.4.0 · build 6 · 无模型权重测试分发版

这个包只发布运行器、界面和本地模型资产包工具。它**不包含** LLVNR 模型、权重、tuning、`CameraSetup.plist`，也不包含任何可直接运行的 `.llvnrkit`。首次使用前，必须由使用者在本地从自己有权使用的材料生成资产包并导入。

> 本地转换工具不是 Apple 素材的再分发许可。是否可以把生成的模型、权重或 tuning 提供给其他人，仍取决于素材本身的授权。

## 分发方式

这个方案参考了 [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) 的工程边界：公开仓库发布转换/运行代码，让使用者在本机提供原始 Apple 文件并生成所需产物；仓库本身不托管 Apple 权重或转换后的模型。AppleNeuralHash2ONNX 只提供这条工程模式的参考，不构成法律意见，也不代表 Apple 授权了 LLVNR 的衍生分发。

本包采用同样的边界：

- App 只读取 `~/Library/Application Support/LLVNRDemo/CurrentModelKit`。
- 公开 App 不镜像固件，不下载 IPSW，不自动解密或提取固件。
- 包内 recovery/builder 工具处理使用者本地的 HWX、tuning 和运行资产；公开包本身不携带这些材料。
- 模型、tuning 和哈希都留在本机；App 不上传资产包。
- `.llvnrkit` 仍可能包含受 Apple 权利约束的派生数据，不应因为“格式变了”就公开上传。

## 包内文件

```text
LLVNR Demo Weightless.app
README.md
LICENSE-ANEForge.txt
KIT-DISTRIBUTION-NOTICE.md
kit/
  README.md
  PRIVACY.md
  extract_ipsw_assets.py
  recover_portable_sources.py
  export_runtime_assets.py
  export_research_assets.py
  build_catalog.py
  build_kit.py
  verify_kit.py
  convert_local_model.py
  sanitize_run_report.py
  audit_release.py
  check_readme_links.py
  docs/
  common/
  research/
  third_party/ANEForge/
  schema/kit.schema.json
PACKAGE-VERIFICATION.json
```

`kit/` 是本机转换与打包工具，不包含研究树中的模型目录。

## 本地生成流程

### 1. 准备本地研究资产

包内 `recover_portable_sources.py` 可以从本机提取目录的 `model.hwx` 生成
portable source graph 和系数；它使用包内 sanitized synthetic layout metadata，
不读取或分发模型权重。先生成私有 source graph：

```bash
python3 kit/recover_portable_sources.py \
  --extracted /path/to/private/llvnr-extracted \
  --output /path/to/private/portable-sources
```

也可以直接使用已经通过数值验证的 portable runtime assets。每个模型目录至少包含：
  - `manifest.json`
  - `model.mil`
  - `weights.bin`
  - `source_graph.json`
  - `export_validation.json`
  - 清单声明的 `precision/` 和 `evidence/` 文件
- 一个本机生成的 catalog 目录：
  - `catalog.json`
  - `Tunings/`
  - 可选的 `CameraSetup.plist`

这些文件应由你自己的研究流程生成；公开 kit 不负责从固件取得它们。

如果已有 portable source graph 和系数数组，公开 exporter 可以直接生成运行资产：

```bash
python3 kit/export_runtime_assets.py \
  --sources /path/to/private/portable-sources \
  --coefficient-root /path/to/private/coefficients \
  --output /path/to/private/model-assets \
  --validation ane
```

它会使用包内 vendored ANEForge 源码生成 `model.mil` / `weights.bin`，并在 Apple Silicon 上执行 CPU/ANE 验证。旧私有研究树仍可使用兼容桥接：

```bash
python3 kit/export_research_assets.py \
  --research-root /path/to/private/llvnr_port \
  --assets-out /path/to/private/model-assets
```

`recover_portable_sources.py` 负责 `model.hwx` → portable source graph；
`export_runtime_assets.py` 只接受已经生成的 source graph，不会把未验证的图当作可运行模型。

### 2. 生成 catalog 并构建 `.llvnrkit`

```bash
python3 kit/build_catalog.py \
  --extracted /path/to/private/llvnr-extracted \
  --assets /path/to/private/model-assets \
  --output /path/to/private/catalog

python3 kit/build_kit.py \
  --assets /path/to/private/model-assets \
  --catalog /path/to/private/catalog \
  --output /path/to/private/LLVNR-local.llvnrkit \
  --license /path/to/LICENSE-ANEForge.txt
```

builder 会：

- 校验每个模型的 manifest、源模型 SHA-256、CPU/ANE 验证结果、输入端口、输出端口和画布；
- 校验 catalog、tuning 和文件哈希；
- 只复制 App 运行所需的文件，不复制稀疏 `.npy` 中间文件；
- 拒绝 `.hwx`、`.net`、`.weights`、`.shape`、`.espressif` 等原始固件文件；
- 生成 `kit.json`，默认同时生成同名 ZIP。

可用 `--allow-partial` 构建只含部分模型的资产包。默认要求 catalog 中的每个模型都存在。也可以把上面的步骤交给：

```bash
python3 kit/convert_local_model.py \
  --extracted /path/to/private/llvnr-extracted \
  --work /path/to/private/llvnr-work \
  --output /path/to/private/LLVNR-local.llvnrkit
# 或直接给本地 OTA：--ota /path/to/local.aea
```

### 3. 先验证，再导入

```bash
python3 kit/verify_kit.py /path/to/private/LLVNR-local.llvnrkit
python3 kit/verify_kit.py /path/to/private/LLVNR-local.llvnrkit.zip
```

验证器会检查：

- `kit.json`、catalog、模型 manifest 和运行文件哈希；
- 模型 ID 与 catalog 的一致性；
- 文件清单之外没有夹带文件；
- ZIP 路径穿越、符号链接和超量展开；
- 原始固件文件是否被误带入。

只有输出 `"passed": true` 后，才把资产包导入 App。

### 4. 导入 App

打开 `LLVNR Demo Weightless.app`，选择菜单：

```text
File → 导入模型资产包…
```

可以选择未解压的 `.llvnrkit` 文件夹或 builder 生成的 `.zip`。安装采用事务式替换：校验失败时保留当前已安装资产包。

也可以使用无界面的安装入口：

```bash
"/Applications/LLVNR Demo Weightless.app/Contents/MacOS/LLVNRDemo" \
  --install-kit /path/to/private/LLVNR-local.llvnrkit.zip \
  --install-report /tmp/llvnr-install.json
```

生成安装后的目录完整性报告：

```bash
"/Applications/LLVNR Demo Weightless.app/Contents/MacOS/LLVNRDemo" \
  --check-catalog /tmp/llvnr-catalog.json
```

## 资产包格式

```text
LLVNR-local.llvnrkit/
  kit.json
  README.md
  NOTICE.md
  LICENSE-ANEForge.txt        # 可选
  Catalog/
    catalog.json
    Tunings/*.plist
    CameraSetup.plist         # 可选、仅本地
  Models/<MODEL_ID>/
    manifest.json
    model.mil
    weights.bin
    source_graph.json
    export_validation.json
    precision/
    evidence/
```

`kit.json` 的 schema 位于 `kit/schema/kit.schema.json`。不要手工修改模型 manifest 或 `kit.json`；任何字节变化都会导致安装失败。

## 兼容性与签名

- 仅适用于 Apple Silicon Mac，运行时依赖 ANE、Metal 和 macOS 私有 Espresso/E5RT 接口。
- 最低系统版本由 App 的 `Info.plist` 声明；其他 macOS 版本、芯片和相机仍需实机验证。
- 本构建是 ad-hoc 测试签名，不是 Developer ID 签名，也没有 Apple 公证。
- 网络下载后，Gatekeeper 可能要求先尝试打开，再到“系统设置 → 隐私与安全性”允许打开。
- 相机权限和本地文件访问由目标 Mac 管理。
- H13 之外的 ANE 代际尚未在目标硬件上验证；不要把 compile-only 的 `rc=0` 当作支持证据。
- 新导出的 manifest 会记录 `target_family` / `target_arch`；App 会在目标代际不匹配时拒绝加载。
- 分享运行报告前使用 `kit/sanitize_run_report.py`；详见包内 `kit/PRIVACY.md`。

## 不应进入公开包的素材

- `weights.bin`
- `model.mil`
- 模型 `manifest.json`
- `export_validation.json` 或任何含模型内容的证据包
- `Tunings/*.plist`
- `CameraSetup.plist`
- `.hwx`、`.net`、`.weights`、`.shape`、`.espressif`
- 任何 `.llvnrkit` 或 `.llvnrkit.zip`

允许公开的是 App 运行器代码、kit 工具源码、schema、文档、运行时许可证和验证报告。

## 从本地 IPSW/OTA 提取模型和 tuning

`kit/extract_ipsw_assets.py` 是一个显式的本地提取包装器。它只接受用户已经拥有的 IPSW/OTA，调用本机 `ipsw` CLI，并严格限制匹配 `ISPKit.framework/<V63|V64|V64s>/LowLightVideo` 下的：

- `LLV-HD-*.bundle/H19.bundle/main/main_ane/model.hwx`
- `*LLVTuningParameters.plist`

示例：

```bash
python3 kit/extract_ipsw_assets.py /path/to/local.aea \
  --variant V63 \
  --output /path/to/private/llvnr-extracted
```

脚本不内置固件密钥、不下载模型、不执行自己的解密实现；`ipsw` 的 AEA 密钥查询行为由外部工具决定。默认要求 V63 的 12 个模型和 6 个 tuning 全部找到；探索其他 OTA 时可以使用 `--allow-partial`，但不要把它当作完整发布包。提取结果只放在私有目录，不要提交。

生成的 `EXTRACT-MANIFEST.json` 只记录相对路径、大小、SHA-256 和计数，不记录输入绝对路径。
