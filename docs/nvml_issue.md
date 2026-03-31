# NVML 驱动不兼容问题报告

## 日期：2026-03-31

## 问题概述

在执行 Step 3（训练 Critic）和 Step 4（训练 Actor）时，PyTorch 调用 NVML 函数失败，导致 DPO 训练无法在 GPU 上运行。

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | 16x V100-SXM3-32GB |
| NVIDIA 驱动 | 450.142.00（2021 年发布） |
| CUDA 版本 | 11.0 |
| 内核版本 | 4.15.0-159-generic |
| 操作系统 | Ubuntu 18.04 |
| PyTorch | 2.10.0+cu128 |
| 系统内存 | ~1540 GB 可用 |

## 错误详情

PyTorch 2.10.0+cu128 在初始化 CUDA 时调用 `nvmlDeviceGetNvLinkRemoteDeviceType`，该函数在 NVIDIA 驱动 450.142.00 中不存在，导致程序崩溃：

```
AttributeError: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1: undefined symbol: nvmlDeviceGetNvLinkRemoteDeviceType
```

## 根本原因

PyTorch 2.10+ 与旧版 NVIDIA 驱动（< 470 系列）不兼容。PyTorch 内部在 `dlopen("libnvidia-ml.so.1")` 后通过 `dlsym` 查找 `nvmlDeviceGetNvLinkRemoteDeviceType` 符号，而旧驱动没有导出该符号。

## 尝试过的方案（均失败）

### 1. 设置 CUDA_VISIBLE_DEVICES=0
- **结果**: 失败
- **原因**: PyTorch 仍然加载 NVML 库并调用该函数，环境变量无法阻止

### 2. 设置 NCCL_P2P_DISABLE=1
- **结果**: 失败
- **原因**: 该变量只影响 NCCL 的 P2P 通信，不影响 PyTorch 自身的 NVML 调用

### 3. LD_PRELOAD stub 库
- **结果**: 失败
- **原因**: PyTorch 使用 `dlsym(handle, ...)` 在特定 `dlopen` 句柄上查找符号，不经过 `RTLD_DEFAULT`，LD_PRELOAD 无法拦截

### 4. objcopy 注入符号到库副本
- **结果**: 未完成
- **原因**: ELF 修改复杂度高，且可能引入其他兼容性问题

### 5. CPU 训练回退
- **结果**: 可行但极慢
- **原因**: Gemma-2-2B 在 CPU + float32 下训练速度极低，实际不可接受

## 当前管线状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| Step 1: 生成轨迹 | 已完成 | 3 个样本 → 22 个偏好对 |
| Step 2: 构建偏好数据 | 已完成 | actor_preferences + critic_preferences |
| Step 3: 训练 Critic | **阻塞** | NVML 不兼容，无法 GPU 训练 |
| Step 4: 训练 Actor | **阻塞** | 同上 |
| Step 5: 评估 | **阻塞** | 依赖 Step 3/4 的模型 |

注意：Step 1 和 Step 2 使用 vLLM 推理（不经过 PyTorch CUDA 直接调用），所以不受此问题影响。

## 建议方案

### 方案 A：升级 NVIDIA 驱动（推荐）
- 将驱动升级到 >= 470 版本
- 需要管理员权限
- 根本解决问题，所有后续步骤可正常进行

### 方案 B：降级 PyTorch
- 使用 PyTorch <= 2.3 + CUDA 11.8 的组合
- 不需要管理员权限
- 需要确认 trl、transformers 等依赖的兼容性
- 可能需要调整部分 API 调用（如 DPOConfig 等）

### 方案 C：使用其他兼容环境
- 在有兼容驱动的机器上运行
- 需要迁移数据和代码

## 附录：关键文件

- DPO 训练入口：`src/training/dpo_trainer.py`
- Step 3 脚本：`scripts/03_train_critic.py`
- Step 4 脚本：`scripts/04_train_actor.py`
- vLLM 推理（不受影响）：`src/inference/vllm_server.py`
