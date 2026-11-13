# 架构与数据流

## 1. 输入层

使用者从自己有权使用的本地材料准备模型目录。仓库不执行固件下载、解密或镜像提取。公开路径先使用 `demo/kit/recover_portable_sources.py` 从本地 HWX 恢复 portable source graph，再由 `demo/kit/export_runtime_assets.py` 生成运行资产；`demo/kit/export_research_assets.py` 只作为旧私有研究树的兼容桥接。

模型资产目录的运行时边界：

```text
MODEL_ID/
  manifest.json
  model.mil
  weights.bin
  source_graph.json
  export_validation.json
  precision/       # 可选，精度敏感模型
  evidence/        # 可选，本地数值证据
```

这些文件属于本地资产，**不应上传到公开仓库**。

## 2. 解析与恢复

`research/work/` 下的模块按以下层次工作：

- `h19_weightlib.py`：读取 HWX 的 Mach-O section、kernel groups 和 symbols；
- `h19_dense_codec.py`：FP16 dense block、lane packing、bias 的编码/解码；
- `h19_sparse_codec.py`、`h19_sparse_extended.py`、`h19_sparse_stream.py`：稀疏 lane、mask、tile stream；
- `h19_statewalk.py`：task state 和寄存器 walk；
- `llvnr_ir.py`、`llvnr_taskflow.py`：task 端口、存储视图和数据依赖；
- `recover_catalog.py`：通用 coefficient codec 和 bounded recovery；
- `graph/recovered_reference.py`：从本地 HWX + recovery 目录构造 task reference；
- `recovery_profiles.json.gz`：sanitized synthetic compiler layout metadata，不含权重；
- `parse_dump.py`：文本 task dump 转结构化 JSON。

解析代码本身不等于模型推理验收。coefficient round-trip、输出数值和硬件执行是不同门槛。

## 3. 源图

`research/work/llvnr_port/graph/` 把恢复的系数和 task 语义组合成通用算子图：

```text
his_frame + cur_frame + dnr_zoom_map + fusion_zoom_map
                         │
                         ▼
              portable source graph
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
    NumPy CPU reference            ANE/Metal adapter
          │                             │
          └──────────────┬──────────────┘
                         ▼
          out_yuv / state_yuv / state_weight_band0
```

`portable_ops.py` 是不依赖硬件和编译器的算术参考。`portable_graph.py` 负责图构建、系数绑定、输入检查和 H13/ANE 适配。`wide1_graph.py`、`superwide05_graph.py`、`tele4_graph.py`、`tele8_graph.py` 和 `large_canvas_graph.py` 负责各模型族的图绑定。`run_portable.py` 负责递归状态、reset 和逐帧报告。`native_reference.py` 保留 task-level 对照路径。

## 4. App

`demo/Sources/LLVNRDemo` 包含：

- `ModelStore.swift`：目录与 ZIP 导入、事务式安装和完整性检查；
- `ModelRuntime.swift`：MIL 与权重运行时；
- `PipelineEngine.swift`：串行 FIFO、时间历史和显示输出；
- `VideoPreviewSession.swift`、`VideoFrameProcessor.swift`：视频预览与反向 seek 处理；
- `MetalProcessor.swift` 和 `Resources/*.metal`：YUV 预处理/显示路径。

App 运行时从 `~/Library/Application Support/LLVNRDemo/CurrentModelKit` 读取已验证的本地 kit，不从网络拉取模型。

## 5. 验收分层

1. **格式**：字节区间、mask、lane packing、round-trip。
2. **图**：节点、端口、形状、状态回边和 CPU 执行。
3. **运行时**：ANE/Metal 编译、输入输出、递归状态和显示池。
4. **体验**：视频反向拖动、HDR 显示、相机和功耗，需要目标机器和显示器实机验证。

发布文档不得把前两层的通过自动写成后两层已经通过。
