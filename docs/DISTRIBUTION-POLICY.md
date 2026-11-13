# 分发边界

## 采用的工程模式

本项目参考 [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) 的项目组织方式：

1. 公开解析、转换、图重写、运行时和验证代码；
2. 原始素材由使用者在本地准备；
3. 仓库不托管权重、tuning、模型文件或转换后的 `.llvnrkit`；
4. 代码许可证不扩大第三方素材的权利；
5. App 只消费用户本地生成并验证过的资产包。

这是软件分发边界，不是对 Apple 固件、模型权重或衍生数据的法律授权。AppleNeuralHash2ONNX 只作为工程模式参考，不代表它认可或授权 LLVNR 项目。

## 公开

- H19/H13 解析器、权重解码器和字节级 round-trip 检查；
- 任务表、寄存器说明、图重建方法和数值验证代码；
- `.llvnrkit` builder、verifier、schema 和合成测试；
- 无权重 App 源码、打包脚本、签名兼容性说明；
- 不含路径和设备的性能/数值摘要。

## 不公开

- 原始固件、IPSW、解密中间产物；
- `.hwx`、`.net`、`.weights`、`.shape`、`.espressif` 等模型资源；
- `weights.bin`、`model.mil`、模型 manifest、tuning plist、`CameraSetup.plist`；
- 任何 `.llvnrkit` 或包含模型内容的 ZIP；
- 实拍照片、视频、预览帧、对照蒙太奇、签名、证书和设备标识。

## Release asset

Release asset 只提供**无权重 App**和工具源码包。它不因为 App 通过 `codesign --sign -` 做 ad-hoc 签名而变成 Developer ID 构建，也没有 Apple 公证。

## 本地提取器

`demo/kit/extract_ipsw_assets.py` 只包装外部 `ipsw` CLI，接受使用者提供的本地 IPSW/OTA。它不携带密钥、不下载固件、不修改系统密钥数据库，也不把提取结果发到网络。使用者的本地 `ipsw` 版本和 AEA 密钥查询行为仍需自行确认。
