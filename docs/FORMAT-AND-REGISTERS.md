# H19 格式与寄存器笔记

这是一份面向移植和审计的索引，不是完整规范，也不包含权重。

## 地址单位

- TD packet 通常以 32-bit word 索引寄存器；
- `docs/TASK-TABLE.md` 的地址使用 HW byte address：`byte = word * 4`；
- task table 的 `D`、`L`、`P` 分别表示 DRAM/BSS、L2 和外部端口。

## 常用记录

| 记录 | 作用 |
|---|---|
| Common | task 类型、输出维度和通用标志 |
| MacCfg | 卷积/池化/PE 的 MAC 模式 |
| NE KernelCfg | 卷积核、稀疏、palette、激活和 packing 线索 |
| PE | element-wise、affine、LUT、比较和 clamp |
| TileDMA | 源/目标空间、base、stride、window 和 crop |
| KernelDMA | kernel group 地址与 section 偏移 |

## 代码入口

```text
research/work/h19_statewalk.py       task state、section、word walk
research/work/h19_weightlib.py       kernel groups / symbols / section
research/work/h19_dense_codec.py     dense FP16 block and bias
research/work/h19_sparse_codec.py    sparse 2x2/3x3 masks and payload
research/work/llvnr_ir.py            model export
research/work/llvnr_taskflow.py      storage views and dependency trace
research/work/llvnr_port/graph/execute_numpy.py
                                     task-level NumPy executor
```

## 解码原则

- 先验证 descriptor/shape/address，再做 coefficient decode；
- 解码结果必须能按原格式重新编码并得到相同字节；
- 不把“能被解码”误写成“能被 H19 硬件执行”；
- 稀疏、dense、palette、embedded-LUT 等模式要分开验证；
- 对未实现的模式 fail closed，不静默替换成随机权重。

## 寄存器/任务表的用途

`docs/TASK-TABLE.md` 是恢复后的调度导航表，用于定位 task、端口、空间和 kernel group。它是研究元数据，不包含模型权重或可执行模型。表中出现的地址和 group hash 只描述本地分析结果。
