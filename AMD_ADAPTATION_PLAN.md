# nano-vllm AMD 平台适配计划

## 环境信息
- CPU: AMD Ryzen 7 5800U (8C/16T, Zen3)
- GPU: 集成 Radeon Vega 8 (无ROCm支持，WSL2无GPU设备暴露)
- RAM: ~7.6 GB
- OS: WSL2 Ubuntu
- Python: 3.12.3

## 适配目标
1. 支持 AMD ROCm GPU (独立显卡如RX 7900, MI300等)
2. 支持 CPU 回退模式 (用于当前集成GPU环境)
3. 保持 NVIDIA CUDA 兼容性

## 需要修改的文件

### 1. 新增: `nanovllm/utils/device.py`
- 设备检测: CUDA / ROCm / CPU
- 统一内存管理 API
- 统一张量设备转移 API

### 2. 修改: `nanovllm/layers/attention.py`
- 将 flash-attn 改为可选依赖, 提供 PyTorch SDPA 回退
- 将 Triton kernel 改为可选, 提供纯 PyTorch 回退
- 自动选择最优后端

### 3. 修改: `nanovllm/engine/model_runner.py`
- 替换硬编码 "cuda" 为自动检测设备
- 分布式后端: nccl(CUDA/ROCm) / gloo(CPU)
- CUDA graph: 仅在 CUDA 上启用
- 内存管理: 适配 CPU 模式
- pin_memory + .cuda() → 统一设备转移

### 4. 修改: `nanovllm/config.py`
- 添加 device 字段

### 5. 修改: `pyproject.toml`
- flash-attn 和 triton 改为可选依赖

## 技术方案

### Attention 后端优先级
1. flash-attn (如果安装且设备支持)
2. PyTorch SDPA (F.scaled_dot_product_attention)

### KV Cache Store 后端优先级
1. Triton kernel (如果安装且设备支持)
2. 纯 PyTorch index 操作

### 设备检测逻辑
```
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip:
        device_type = "rocm"   # AMD ROCm (torch.cuda API 可透明使用)
    else:
        device_type = "cuda"   # NVIDIA CUDA
else:
    device_type = "cpu"
```

## 状态
- [x] 代码分析
- [x] 设备工具模块 (`nanovllm/utils/device.py`)
- [x] Attention 改造 (`nanovllm/layers/attention.py`)
- [x] ModelRunner 改造 (`nanovllm/engine/model_runner.py`)
- [x] Config 更新 (`nanovllm/config.py`)
- [x] 依赖更新 (`pyproject.toml`)
- [x] RotaryEmbedding 兼容性修复 (`nanovllm/layers/rotary_embedding.py`)
- [x] CPU 模式端到端测试通过 (Qwen3-0.6B, ~8 tok/s decode)

## 测试结果
- PyTorch 2.6.0+cpu, Python 3.12.3
- Qwen3-0.6B 模型, CPU 推理成功
- Decode 速度: ~7-8 tok/s (AMD Ryzen 7 5800U)
