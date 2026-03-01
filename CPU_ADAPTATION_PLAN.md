# nano-vllm CPU 适配计划

## 概述
将 nano-vllm 从仅支持 NVIDIA GPU 适配为同时支持 CPU 平台。  
核心思路：通过 device 自动检测，条件分发 GPU/CPU 代码路径。

## GPU 依赖分析

### 1. `nanovllm/engine/model_runner.py` (最重要)
- `dist.init_process_group("nccl", ...)` → CPU 需用 "gloo"
- `torch.cuda.set_device(rank)` → CPU 不需要
- `torch.set_default_device("cuda")` → CPU 改为 "cpu"
- CUDA Graph 捕获 → CPU 不支持，强制 eager
- `torch.cuda.synchronize/empty_cache/mem_get_info/memory_stats` → CPU 无对应
- `.cuda(non_blocking=True)` → CPU 去掉
- `pin_memory=True` → CPU 不需要

### 2. `nanovllm/layers/attention.py`
- `flash_attn` (flash_attn_varlen_func, flash_attn_with_kvcache) → GPU only
- `triton` kernel (store_kvcache_kernel) → GPU only
- CPU 替代方案：
  - 使用 `F.scaled_dot_product_attention` (PyTorch 2.0+)
  - KV cache 存储用纯 PyTorch 索引操作

### 3. `nanovllm/config.py`
- `gpu_memory_utilization` → CPU 需要基于系统 RAM 分配

### 4. `pyproject.toml`
- `triton>=3.0.0` 和 `flash-attn` → 仅 GPU 需要，改为可选

## 修改方案

### Step 1: config.py
- 添加 `device: str = "auto"` 字段
- `__post_init__` 中解析 auto → "cuda" if available else "cpu"
- CPU 时强制 `enforce_eager = True`

### Step 2: attention.py
- 条件导入 flash_attn / triton
- 添加 CPU 版本：
  - `store_kvcache_cpu()`: 纯 PyTorch 索引赋值
  - `gather_kv_from_blocks()`: 从分页 KV cache 收集
  - `prefill_attention_cpu()`: 循环 + SDPA
  - `decode_attention_cpu()`: 循环 + SDPA
- Attention.forward() 中根据 device.type 分发

### Step 3: model_runner.py
- 根据 device 选择 distributed backend ("gloo" vs "nccl")
- CPU: 跳过 CUDA Graph, cuda.empty_cache 等
- CPU: 内存分配用 /proc/meminfo 获取可用 RAM
- CPU: 去掉 pin_memory 和 .cuda() 调用

### Step 4: pyproject.toml
- flash-attn 和 triton 移到 optional dependencies [gpu]

### Step 5: 测试
- 下载 Qwen3-0.6B
- 运行 example.py 验证 CPU 推理
