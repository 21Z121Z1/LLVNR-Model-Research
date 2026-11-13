# Security

## Reporting

请通过 GitHub 的私密安全报告功能联系仓库维护者，不要在公开 Issue 中粘贴固件、模型、tuning、设备标识、证书、私钥或个人媒体。

## Repository boundaries

- 不要提交 `.hwx`、`.net`、`.weights`、`.shape`、`.espressif`、`.dat`、`.fp16`、`.npy`、`.npz`、`.bvn`、`.bgra`、`.llvnrkit` 或模型缓存。
- 不要提交 provisioning profile、证书、私钥、Team ID、设备 UDID、序列号、IPSW 或解密后的固件。
- 不要提交含人脸、房间、签名或设备信息的测试照片/视频。
- `.llvnrkit` builder 采用本地输入；不要把生成的资产包作为仓库内容上传。
- 发布构建若需要 Developer ID，请在本机 CI secret 或 Apple 账户流程中完成；仓库不保存签名材料。

## Known limitation

仓库中的 App 使用私有 E5RT 接口，属于研究性兼容层。它可能在 macOS 更新后失效；这不是安全边界或生产运行保证。
