# ML-KEM 与 Saber 的二值 NTT/INTT

[English](README.md)

本仓库包含两套完整的软件实现：ML-KEM-512/768/1024 和
LightSaber/Saber/FireSaber。内容包括 KeyGen、Encaps、Decaps，
NTT/INTT 权重生成、二进制位平面、整数重构、模约减和官方测试验证。
两个目录可以分别运行。

| 算法家族 | 参数配置 | 变换映射 |
| --- | --- | --- |
| ML-KEM | ML-KEM-512、ML-KEM-768、ML-KEM-1024 | 模 3329，两条 128 系数通道，每个算子使用 12 个二值位平面 |
| Saber | LightSaber、Saber、FireSaber | 辅助模数 3329、7681，每个模数四条 64 系数通道，使用 12、13 个二值位平面 |

二值权重位平面的元素严格为 `0/1`，不包含 RRAM 电导值。数学源矩阵保存的是
拆分前的模整数系数。二值后端计算权重与输入位的部分和，再进行数字重构。这是理想整数
计算模型；仓库不包含 RRAM 板卡驱动、器件测量、ADC 仿真、RTL 或论文延时模型。

## 安装与运行

需要 Python 3.10 或以上版本、NumPy。在仓库根目录执行：

```sh
git clone https://github.com/wangsongq/rram-pqc-transforms.git
cd rram-pqc-transforms
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 快速检查两个算法家族
python scripts/verify.py --suite smoke

# 完整官方用例及算术核查
python scripts/verify.py --suite full
```

运行完整的密钥生成、封装和解封装示例：

```sh
python mlkem_full_flow/example.py --parameter-set ML-KEM-768
python saber_full_flow/example.py
```

安装依赖后，Python 验证不需要联网。官方测试输入和预期输出已随仓库保存，
运行时会核对来源和哈希。失败时返回非零退出码；统一运行日志写入
`generated/verification/`，各模块的详细结果写入各自的生成目录，
这些运行产物已加入 `.gitignore`。

只检查其中一个家族：

```sh
python scripts/verify.py --family mlkem --suite full
python scripts/verify.py --family saber --suite full
```

更多命令、接口和权重生成步骤见
[ML-KEM 说明](mlkem_full_flow/README.md)、
[Saber 说明](saber_full_flow/README.md) 和
[复现说明](docs/REPRODUCIBILITY.md)。

## 推荐使用的后端

- **ML-KEM：** `rram_transform_only.py` 中的 `TransformOnlyMLKEM`，
  建议显式设置 `signed_cbd_inputs=False`。仅替换 NTT/INTT，其他数字运算保持不变。
  INTT 权重融合归一化与 Montgomery 缩放，输出恢复为普通余数。
- **Saber：** `saber.py` 中的 `Saber(..., backend="rram_binary")`。
  使用模 3329 和模 7681 的辅助变换，每个模数包含四条 64 系数通道，
  通过模四次多项式的基乘积和中心化 CRT 完成多项式运算。

参考后端用于核对计算结果，模型中的算术计数不能直接解释为硬件时延。

## 重新生成权重

矩阵和二值权重已经随仓库保存，也可以从数学定义重新生成。在仓库根目录执行：

```sh
python mlkem_full_flow/matrices/generate_matrices.py
python mlkem_full_flow/rram_binary/generate_weights.py
python mlkem_full_flow/rram_intt_montgomery_weights/generate_weights.py

python saber_full_flow/matrices/source_build_weights.py
python saber_full_flow/generate_auxiliary_weights.py
```

CSV 保存二值位平面，行对应输入、列对应输出；NPZ 保存对应数组和布局元数据。
最低有效位编号为 0。数学源矩阵中的整数是拆分前的模系数。
已有权重与生成结果不一致时会报错，不会直接覆盖。

## 验证口径

ML-KEM 有 **435 个独立官方 ACVP 用例**，两种输入配置各运行一次，共
870 次用例执行。Saber 在二值后端上验证 **300 个独立官方 Round-3 KAT 用例**。
重复运行不会增加独立测试用例数量。

当前整理版本的上述检查全部通过，Saber 共完成 1500 项输出比较。
详细环境、算术检查和官方 C 参考验证结果见
[验证记录](docs/VALIDATION.md)。

这些结果用于软件功能验证，不代表 NIST 认证或物理阵列读出正确率。Python
实现便于阅读和追踪，不保证恒定时间。附带的密钥和随机种子均为公开测试数据；
启用 trace 会记录密钥、随机数和中间值。

## 可选功能

重新编译官方 Saber C 参考实现，需要 C 编译器和 OpenSSL 开发库：

```sh
# Debian/Ubuntu 的依赖包：build-essential、libssl-dev
python scripts/verify.py --family saber --suite reference
```

这一步不是运行 Python 官方用例检查的前提。

## 许可证

本项目自有代码与文档采用 [MIT](LICENSE)。官方测试数据和第三方参考代码
保留各自原有声明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
