# 可复现检查

本仓库的公开 smoke test不依赖模型权重、相机或 ANE：

```bash
python3 -m unittest discover -s tests -v
```

App 源码的编译检查需要 macOS：

```bash
swift build -c release --product LLVNRDemo
```

研究代码的完整运行需要使用者提供本地原料。仓库中保留了相对目录约定：

```text
research/work/
  h19_*.py
  llvnr_*.py
  llvnr_port/
    ota_extract/          # 本地私有，不提交
    model_assets/         # 本地私有，不提交
    catalog_ports/        # 本地私有，不提交
    graph/
```

不要把上述本地目录或生成的 `.llvnrkit` 提交到 Git。发布前的身份/二进制扫描见 `tools/audit_release.py`。

如果要从本地 IPSW/OTA 重新生成原料，先运行：

```bash
python3 demo/kit/extract_ipsw_assets.py /path/to/local.aea \
  --variant V63 --output /path/to/private/llvnr-extracted
```

该命令会生成 `EXTRACT-MANIFEST.json`，默认要求 V63 的 12 个模型和 6 个 tuning。提取目录属于私有研究资产，不能提交。
