# 来源与清理记录

本仓库的内容来自一个本地 LLVNR/H19 研究树，经过**白名单导出**而不是直接复制整个工作区。

## 保留

- 独立编写的解析、解码、图执行、kit builder 和 macOS App 源代码；
- 无权重 App 的 Metal/ANE 运行时桥接；
- H19 task/寄存器的研究摘要和 535-task 导航表；
- ANEForge 的 MIT 许可证/NOTICE。

## 删除

以下内容没有进入导出树：

- 原始和派生的照片、视频、房间测试画面、人脸截图、并排对照图；
- `capture/`、`integration/`、`demo/Validation/`、`demo/dist/`、`demo/.build/`；
- `iphone_oracle/`（按用户要求不分发）；
- `model_assets_v2/`、`catalog_ports/`、`dense_weights/`、`special_weights/`、`deconv_weights/`；
- 原始 HWX、IPSW、tuning、`CameraSetup.plist` 和 `.llvnrkit`；
- `source_graph.json`、模型 manifest、权重数组和数值 evidence JSON；
- 证书、provisioning、私钥、Team ID、UDID 或序列号。

## 路径与身份

复制后的源码中不含本机用户名、绝对用户目录路径、主机名、邮箱、Team ID、设备 UDID 或 provisioning 字段。发布审计脚本会再次扫描文本和二进制文件；扫描结果只记录文件名和规则名，不输出命中的个人值。

## 签名

这里的“签名”指 macOS 代码签名，不是手写签名：

- 研究 App 使用 ad-hoc 签名用于本机测试；
- 无权重 Release 包没有 Developer ID 签名，也没有 Apple 公证；
- 仓库不包含 `.p12`、`.mobileprovision`、Team ID、私钥或 Apple 账户凭据。
