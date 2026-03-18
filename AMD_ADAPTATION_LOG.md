# nano-vllm AMD 平台适配完整记录

## 1. 环境分析

### 硬件信息
- **CPU:** AMD Ryzen 7 5800U (8核16线程, Zen3 架构, 3.8GHz)
- **GPU:** 集成 Radeon Vega 8（非独立显卡，WSL2 中无 `/dev/kfd`、`/dev/dri` 设备暴露）
- **RAM:** ~7.6 GB（与 GPU 共享）
- **OS:** WSL2 Ubuntu (Linux 6.6.87, x86_64)
- **Python:** 3.12.3

### 关键发现
1. AMD Ryzen 7 5800U 是 APU（集成 GPU），ROCm 官方仅支持独立 AMD GPU（RX 7900、MI300 等）
2. WSL2 环境中未暴露 GPU 设备节点，无法直接使用 GPU 计算
3. 系统总 RAM 仅 7.6GB，测试时需使用小模型

### 原始项目分析

原始 nano-vllm 仅支持 NVIDIA GPU，依赖以下 NVIDIA 特有技术：

| 依赖 | 位置 | 用途 |
|------|------|------|
| `flash-attn` | `layers/attention.py` | Flash Attention 加速（prefill + decode） |
| `triton` | `layers/attention.py` | KV Cache 存储的 GPU kernel |
| CUDA graphs | `engine/model_runner.py` | decode 阶段图捕获加速 |
| `nccl` | `engine/model_runner.py` | 分布式通信后端 |
| `torch.cuda.*` | `engine/model_runner.py` | 设备管理、内存管理 |

---

## 2. 适配方案

### 设计原则
- **三层后端**: NVIDIA CUDA → AMD ROCm → CPU，自动检测并选择最优后端
- **向后兼容**: 保持原有 NVIDIA CUDA 功能不变
- **优雅降级**: 每个 GPU 专有功能都有 CPU 回退路径

### 设备检测逻辑
```python
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip:
        device_type = "rocm"   # AMD ROCm (torch.cuda API 透明可用)
    else:
        device_type = "cuda"   # NVIDIA CUDA
else:
    device_type = "cpu"        # CPU 回退
```

### Attention 后端优先级
1. **flash-attn** — 如果安装且设备为 GPU
2. **PyTorch SDPA** (`F.scaled_dot_product_attention`) — 通用回退，支持所有平台

### KV Cache Store 后端优先级
1. **Triton kernel** — 如果安装且设备为 GPU
2. **纯 PyTorch index 操作** — 通用回退

---

## 3. 修改的文件

### 3.1 新增: `nanovllm/utils/device.py`

设备检测与抽象工具模块，提供统一 API：

- `detect_device_type()` → `"cuda"` / `"rocm"` / `"cpu"`
- `get_device()` → `torch.device` 对象
- `get_dist_backend()` → `"nccl"` (GPU) / `"gloo"` (CPU)
- `supports_cuda_graphs()` → 仅 NVIDIA CUDA 返回 True
- `get_memory_info()` → GPU 用 `torch.cuda.mem_get_info()`，CPU 读 `/proc/meminfo`
- `to_device()` → 统一张量设备转移（GPU 用 pin_memory + non_blocking，CPU 直接创建）

### 3.2 修改: `nanovllm/layers/attention.py`

**核心变更** — flash-attn 和 triton 改为可选，提供 SDPA 回退：

- `import triton` / `import flash_attn` 改为 `try/except`，设置 `HAS_TRITON` / `HAS_FLASH_ATTN` 标志
- 新增 `_store_kvcache_pytorch()`：纯 PyTorch 实现 KV Cache 存储，使用 index 操作替代 Triton kernel
- 新增 `_sdpa_varlen_prefill()`：使用 `F.scaled_dot_product_attention` 实现变长序列 prefill
  - 逐序列循环处理（因为 SDPA 不支持 packed/varlen 格式）
  - 支持 prefix cache（从 paged KV cache 中按 block_table 收集 K/V）
  - 处理 GQA（通过 `repeat_interleave` 扩展 KV heads）
  - 自动构建 causal mask（prefix cache 场景下 q_len ≠ k_len）
- 新增 `_sdpa_decode()`：使用 SDPA 实现 decode 阶段注意力
  - 从 paged KV cache 按 block_table 收集 K/V
  - 单 token query 无需 causal mask
- `Attention.forward()` 自动分发：GPU 有 flash-attn 用 flash 路径，否则用 SDPA 路径

### 3.3 修改: `nanovllm/engine/model_runner.py`

设备无关化改造：

| 原始代码 | 修改后 |
|---------|--------|
| `dist.init_process_group("nccl", ...)` | `dist.init_process_group(dev.get_dist_backend(), ...)` |
| `torch.cuda.set_device(rank)` | `dev.set_device(rank)` |
| `torch.set_default_device("cuda")` | `torch.set_default_device(dev.get_torch_device_str())` |
| `torch.cuda.empty_cache()` | `dev.empty_cache()` |
| `torch.cuda.synchronize()` | `dev.device_synchronize()` |
| `torch.cuda.mem_get_info()` | `dev.get_memory_info()` |
| `torch.cuda.memory_stats()` | `dev.get_memory_stats()` |
| `torch.tensor(..., pin_memory=True).cuda(non_blocking=True)` | `dev.to_device(data, dtype, self.device)` |
| `if not self.enforce_eager: self.capture_cudagraph()` | `if self.use_cuda_graphs: self.capture_cudagraph()` |

CPU 模式 KV Cache 分配策略：使用可用内存的 50% × `gpu_memory_utilization` 比例。

### 3.4 修改: `nanovllm/config.py`

- 新增 `device_type` 字段，`__post_init__` 中自动检测
- CPU 和 ROCm 模式自动强制 `enforce_eager = True`（禁用 CUDA graphs）

### 3.5 修改: `nanovllm/layers/rotary_embedding.py`

- 移除 `@lru_cache(1)` 装饰器（因为 transformers 5.x 将 `rope_scaling` 从 `None` 变为 dict，dict 不可哈希）
- 改用手动 dict 缓存 `_rope_cache`

### 3.6 修改: `pyproject.toml`

- 版本升级 `0.2.0` → `0.3.0`
- `triton` 和 `flash-attn` 从必选依赖移至可选依赖
- 新增 `[project.optional-dependencies]`：
  - `cuda` extras: `triton>=3.0.0`, `flash-attn`
  - `rocm` extras: `triton>=3.0.0`
- 新增 `safetensors` 为必选依赖（之前漏掉）

---

## 4. 遇到的问题及解决

### Issue 1: PyTorch 2.10.0+cpu 与 Python 3.12 的 torch.compile 不兼容

- **错误**: `TypeError: Too few arguments for <class 'torch._inductor.codegen.common.CSE'>; actual 1, expected 2`
- **原因**: PyTorch 2.10.0 的 inductor CuteDSL 代码有 typing 兼容性问题
- **解决**: 降级到 PyTorch 2.6.0+cpu

### Issue 2: transformers 5.2.0 将 rope_scaling=null 转为 dict

- **错误**: `TypeError: unhashable type: 'dict'` in `get_rope` with `@lru_cache`
- **原因**: transformers 5.x 将 config.json 中 `"rope_scaling": null` 解析为 `{'rope_theta': ..., 'rope_type': 'default'}`
- **解决**: 移除 `@lru_cache`，改用手动 dict 缓存

---

## 5. 测试结果

### 环境
- PyTorch 2.6.0+cpu
- transformers 5.2.0
- Python 3.12.3
- 无 GPU，纯 CPU 模式

### 单元测试
| 测试项 | 结果 |
|--------|------|
| 设备检测 (CPU 模式) | ✅ 通过 |
| `store_kvcache` PyTorch 回退 | ✅ 通过 |
| SDPA varlen prefill (无 prefix cache) | ✅ 通过 |
| SDPA decode (paged KV cache) | ✅ 通过 |
| Attention 模块 prefill | ✅ 通过 |
| Attention 模块 decode | ✅ 通过 |

### 端到端推理测试
- **模型**: Qwen3-0.6B (1.5GB, bfloat16)
- **配置**: `max_model_len=512, max_num_seqs=4, enforce_eager=True`
- **Prompt**: `"Hello, who are you?"`
- **输出**: `'<think>\nOkay, the user asked, "Hello, who are you?" I need to respond appropriately...'`
- **Decode 速度**: ~7-8 tok/s
- **结果**: ✅ 成功

---

## 6. 使用指南

### 安装

```bash
# CPU 模式 (无 GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# NVIDIA CUDA 模式
pip install torch
pip install -e ".[cuda]"

# AMD ROCm 模式
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e ".[rocm]"
```

### 运行

```python
from nanovllm import LLM, SamplingParams

llm = LLM("path/to/model", enforce_eager=True)  # CPU 模式自动 enforce_eager
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello!"], sampling_params)
```

CPU 模式会自动：
- 使用 `gloo` 分布式后端
- 使用 SDPA 替代 Flash Attention
- 使用 PyTorch index 操作替代 Triton kernel
- 禁用 CUDA graphs
- 从 `/proc/meminfo` 读取可用内存

---

## 7. 算子融合 (Operator Fusion) 实验

### 背景

尝试理解算子融合对性能的影响，编写 benchmark 脚本 `fusion.py` 进行实验。

### 第一版：NumPy 实现（失败）

最初使用 NumPy 实现"融合"vs"未融合"的 sigmoid × y 操作对比：

```python
# "未融合"
def un_fused_sigmoid_mul(x, y):
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * y

# "融合"
def fused_sigmoid_mul(x, y):
    return (1 / (1 + np.exp(-x))) * y
```

**结果**：性能无差异，甚至出现负收益。

**原因分析**：NumPy 不存在真正的算子融合机制。将代码写成一行只是语法层面的合并，底层依然逐步调用独立的 C kernel（`neg → exp → add → div → mul`），每一步都会：
1. 分配临时数组
2. 遍历整个数组读写内存
3. 产生额外的内存带宽开销

NumPy 没有 JIT 编译器，不会分析表达式树来合并 kernel，因此无论怎么改写 Python 代码，执行路径完全相同。

### 第二版：PyTorch 实现（成功）

改用 PyTorch 实现，对比三种方案：

1. **Unfused (eager)** — 标准 PyTorch eager 执行，每个算子独立 launch
2. **torch.compile** — 使用 Inductor 后端自动融合（CPU 上用 C++/OpenMP，GPU 上自动生成 Triton kernel）
3. **Triton kernel** — 手写融合 kernel（仅 CUDA GPU 可用）

```python
# Unfused eager
def unfused_sigmoid_mul(x, y):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * y

# torch.compile 自动融合
@torch.compile(mode="max-autotune")
def compiled_sigmoid_mul(x, y):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * y

# Triton 手写融合 kernel（GPU only）
@triton.jit
def _fused_sigmoid_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = tl.sigmoid(x) * y
    tl.store(out_ptr + offsets, result, mask=mask)
```

### Benchmark 结果（CPU 模式，Tensor size: 1,000,000）

| Method | P50 (ms) | P90 (ms) | P99 (ms) | Mean (ms) | Speedup |
|---|---|---|---|---|---|
| Unfused (eager) | 3.3008 | 3.9144 | 4.9159 | 3.3818 | 1.00x |
| torch.compile | 0.3089 | 0.4457 | 0.6860 | 0.3323 | **10.18x** |

### 关键结论

`torch.compile` 通过 Inductor 后端实现了真正的算子融合，性能提升约 **10x**：

1. **追踪计算图** — 识别出 `sigmoid → mul` 的子图
2. **融合成单个 kernel** — 一次遍历数据完成所有计算
3. **消除中间 tensor 分配** — 不再需要存储 sigmoid 的中间结果
4. **减少内存带宽压力** — 这类逐元素操作是 memory-bound 的，减少内存访问次数是最大收益来源

这也说明了为什么 nano-vllm 在 GPU 上使用 Flash Attention、Triton kernel 等融合算子能获得显著加速 — 本质上都是同一个优化思路：**减少 kernel launch 次数和中间内存分配，最大化计算/访存比**。

---

## 8. 手搓 Triton Kernel 替代 torch.compile

### 动机

项目中有三处使用 `@torch.compile` 做自动融合：

| 位置 | 操作 | 文件 |
|------|------|------|
| `SiluAndMul.forward` | `silu(gate) * up` | `layers/activation.py` |
| `RMSNorm.rms_forward` / `add_rms_forward` | RMSNorm ± 残差加法 | `layers/layernorm.py` |
| `RotaryEmbedding.forward` | RoPE 旋转位置编码 | `layers/rotary_embedding.py` |

手搓的好处：**零编译开销、确定性行为、可精细控制 block size / 共享内存 / 数据类型转换**。缺点是维护成本高。

### 8.1 Fused SiluAndMul

原始代码（`layers/activation.py`）：
```python
@torch.compile
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

手搓 Triton kernel：input 形状为 `(N, 2*D)`，将其看作 `gate[:, :D]` 和 `up[:, D:]`，在一个 kernel 里完成 `silu(gate) * up`：

```python
@triton.jit
def _silu_and_mul_kernel(
    input_ptr, output_ptr,
    stride_row,          # input 每行的 stride（== 2*D）
    D: tl.constexpr,     # 输出维度（input 维度的一半）
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)

    # 一次 load 两段
    gate = tl.load(input_ptr + row * stride_row + cols).to(tl.float32)
    up   = tl.load(input_ptr + row * stride_row + D + cols).to(tl.float32)

    # silu(x) = x * sigmoid(x)
    result = gate * tl.sigmoid(gate) * up

    tl.store(output_ptr + row * (stride_row // 2) + cols, result)

def fused_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    D = x.shape[-1] // 2
    out = torch.empty(N, D, device=x.device, dtype=x.dtype)
    _silu_and_mul_kernel[(N,)](x, out, x.stride(0), D=D)
    return out
```

> **注意**：`D` 作为 `tl.constexpr`，Triton 会为每个不同的 `D` 值编译一次。对于固定模型（如 Qwen3 的 `intermediate_size`），这不是问题。但如果 `D` 很大（比如 > 几万），需要将其拆成多个 block 处理。

### 8.2 Fused Add + RMSNorm

原始代码（`layers/layernorm.py`）：
```python
@torch.compile
def add_rms_forward(self, x, residual):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

手搓 Triton kernel（以 fused add + RMSNorm 为例，这是最有价值的那个）：

```python
@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr, new_residual_ptr,
    stride,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    offset = row * stride + cols

    x = tl.load(x_ptr + offset).to(tl.float32)
    res = tl.load(residual_ptr + offset).to(tl.float32)

    # fused add
    x = x + res

    # 写回 new residual（bf16/fp16）
    tl.store(new_residual_ptr + offset, x)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cols).to(tl.float32)
    result = x * rrms * w

    tl.store(output_ptr + offset, result)
```

**核心收益**：`x + residual` → `x²` → `mean` → `rsqrt` → `x * rrms * weight` 这一连串操作，在 eager 模式下需要 5+ 次 HBM 读写，融合后只需 **2 次读 + 2 次写**。

### 8.3 Fused Rotary Embedding

原始代码（`layers/rotary_embedding.py`）：
```python
@torch.compile
def forward(self, positions, query, key):
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key
```

手搓 Triton kernel：对 Q 和 K 的每一行，从 `cos_sin_cache` 取 cos/sin，然后就地旋转：

```python
@triton.jit
def _rotary_kernel(
    q_ptr, k_ptr, cos_sin_cache_ptr, positions_ptr,
    q_stride, k_stride, cache_stride,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    pos = tl.load(positions_ptr + row)
    half = tl.arange(0, HALF_DIM)

    cos = tl.load(cos_sin_cache_ptr + pos * cache_stride + half)
    sin = tl.load(cos_sin_cache_ptr + pos * cache_stride + HALF_DIM + half)

    # 处理 Q heads
    if head < num_q_heads:
        base = row * q_stride + head * HEAD_DIM
        x1 = tl.load(q_ptr + base + half).to(tl.float32)
        x2 = tl.load(q_ptr + base + HALF_DIM + half).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        tl.store(q_ptr + base + half, y1)
        tl.store(q_ptr + base + HALF_DIM + half, y2)
    # 处理 K heads
    else:
        kh = head - num_q_heads
        base = row * k_stride + kh * HEAD_DIM
        x1 = tl.load(k_ptr + base + half).to(tl.float32)
        x2 = tl.load(k_ptr + base + HALF_DIM + half).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        tl.store(k_ptr + base + half, y1)
        tl.store(k_ptr + base + HALF_DIM + half, y2)
```

grid 设为 `(N_tokens, num_q_heads + num_k_heads)`，每个 program 只处理一个 token 的一个 head。

### 8.4 其他手搓路径

#### CUDA C++ Extension

如果对性能要求极致（比 Triton 再快 10-20%），可以用 `torch.utils.cpp_extension` 加载自定义 CUDA kernel，但需要：
- 处理 half/bfloat16 的类型转换
- 手动管理 shared memory、warp shuffle（做 reduction 时）
- RMSNorm 的 `mean` 需要 warp-level reduction，复杂度显著上升

#### 调用已有高性能库

生产级推理引擎通常不完全手搓，而是调用已有的优化库：

| 操作 | 常用库 |
|------|--------|
| RMSNorm (fused add + norm) | `flash-attn` 的 `flash_attn.ops.rms_norm`，或 `vllm._custom_ops` |
| SiluAndMul | `vllm._custom_ops.silu_and_mul`（CUDA kernel） |
| Rotary Embedding | `flash-attn` 的 `apply_rotary_emb`，或 `flash-infer` |
| Fused QKV + RoPE + KV cache | `flashinfer` 一体化 kernel |

### 8.5 方案对比

| 方案 | 开发成本 | 性能 | 推荐场景 |
|------|---------|------|---------|
| `torch.compile` | 最低 | 很好（~90% 最优） | 快速原型、教学 |
| Triton 手搓 | 中等 | 优秀 | 学习 kernel fusion 原理、定制化需求 |
| CUDA C++ 手搓 | 最高 | 极致 | 生产级、极端性能需求 |
| 调用 flash-attn/vllm ops | 低 | 极致 | 工程实践首选 |

---

## 9. 项目架构总览

### 9.1 TUI 架构图

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              nano-vllm 整体架构                                     ║
║                         (~1200 行的极简 vLLM 推理引擎)                                ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

  example.py / bench.py
        │
        │  from nanovllm import LLM, SamplingParams
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  nanovllm/llm.py  ──  LLM(LLMEngine)                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │  nanovllm/engine/llm_engine.py  ──  LLMEngine                                 │  │
│  │                                                                                │  │
│  │   generate(prompts, sampling_params)                                           │  │
│  │     │                                                                          │  │
│  │     ├─ 1. add_request()  ──► Sequence(token_ids, sampling_params)              │  │
│  │     │                                                                          │  │
│  │     └─ 2. loop step() until finished:                                          │  │
│  │          │                                                                     │  │
│  │          ├──────────────────────────┐                                           │  │
│  │          ▼                          ▼                                           │  │
│  │  ┌─────────────────┐    ┌──────────────────────┐                               │  │
│  │  │   Scheduler      │    │   ModelRunner         │                               │  │
│  │  │  (scheduler.py)  │    │  (model_runner.py)    │                               │  │
│  │  └────────┬─────────┘    └──────────┬───────────┘                               │  │
│  │           │                         │                                           │  │
│  │           ▼                         ▼                                           │  │
│  │     schedule()              run(seqs, is_prefill)                               │  │
│  │       │                       │                                                 │  │
│  │       │                       ├─ prepare_prefill / prepare_decode               │  │
│  │       │                       ├─ set_context(...)                               │  │
│  │       │                       ├─ run_model() ──► model.forward()                │  │
│  │       │                       │     ├─ eager (prefill)                           │  │
│  │       │                       │     └─ CUDA Graph replay (decode)               │  │
│  │       │                       └─ sampler(logits, temps) ──► token_ids           │  │
│  │       │                                                                         │  │
│  │       ▼                                                                         │  │
│  │  postprocess(seqs, token_ids)                                                   │  │
│  │    └─ append token / check EOS / deallocate blocks                              │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │  两大核心子系统 │
                        └──────┬───────┘
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│       调度系统                │   │              模型执行系统                         │
│                             │   │                                                 │
│  ┌───────────────────────┐  │   │  ┌───────────────────────────────────────────┐  │
│  │  Scheduler            │  │   │  │  ModelRunner                              │  │
│  │  (scheduler.py)       │  │   │  │  (model_runner.py)                        │  │
│  │                       │  │   │  │                                           │  │
│  │  waiting_queue ─────┐ │  │   │  │  ┌─ load_model()                         │  │
│  │  running_queue ─┐   │ │  │   │  │  ├─ init_kv_cache()                      │  │
│  │                 │   │ │  │   │  │  ├─ init_cuda_graphs()                    │  │
│  │  schedule():    │   │ │  │   │  │  │                                        │  │
│  │   prefill ◄─────┘   │ │  │   │  │  │  Tensor Parallelism:                  │  │
│  │     └─ allocate      │ │  │   │  │  │   rank 0: write_shm(cmd)             │  │
│  │   decode ◄───────────┘ │  │   │  │  │   rank N: read_shm() + execute       │  │
│  │     └─ append          │  │   │  │  └───────────────────────────────────────│  │
│  └───────────┬───────────┘  │   │  └───────────────────────────────────────────┘  │
│              │              │   │                                                 │
│              ▼              │   │  ┌───────────────────────────────────────────┐  │
│  ┌───────────────────────┐  │   │  │  Context (utils/context.py)               │  │
│  │  BlockManager         │  │   │  │  线程局部元数据，模型前向传播时读取          │  │
│  │  (block_manager.py)   │  │   │  │                                           │  │
│  │                       │  │   │  │  is_prefill, cu_seqlens_q/k,              │  │
│  │  ┌─ can_allocate()    │  │   │  │  slot_mapping, block_tables,              │  │
│  │  ├─ allocate()        │  │   │  │  context_lens, max_seqlen_q/k            │  │
│  │  ├─ can_append()      │  │   │  └───────────────────────────────────────────┘  │
│  │  ├─ append()          │  │   │                                                 │
│  │  └─ deallocate()      │  │   │  ┌───────────────────────────────────────────┐  │
│  │                       │  │   │  │  Loader (utils/loader.py)                 │  │
│  │  Prefix Caching:      │  │   │  │  load safetensors → packed_modules_mapping│  │
│  │   xxhash(token_ids)   │  │   │  └───────────────────────────────────────────┘  │
│  │   → 复用已有 block     │  │   │                                                 │
│  └───────────────────────┘  │   └─────────────────────────────────────────────────┘
└─────────────────────────────┘
```

### 9.2 模型前向传播

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                        模型层 (nanovllm/models/qwen3.py)                              │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Qwen3ForCausalLM                                                              │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Qwen3Model                                                              │  │  │
│  │  │                                                                          │  │  │
│  │  │  embed_tokens ──► hidden_states                                          │  │  │
│  │  │       │                                                                  │  │  │
│  │  │       ▼                                                                  │  │  │
│  │  │  ┌─ DecoderLayer × N ──────────────────────────────────────────────┐     │  │  │
│  │  │  │                                                                  │     │  │  │
│  │  │  │   input_layernorm (RMSNorm + residual)                          │     │  │  │
│  │  │  │        │                                                         │     │  │  │
│  │  │  │        ▼                                                         │     │  │  │
│  │  │  │   ┌─ Qwen3Attention ──────────────────────────────────────┐     │     │  │  │
│  │  │  │   │  qkv_proj ──► q, k, v                                 │     │     │  │  │
│  │  │  │   │  q_norm / k_norm (RMSNorm, if no bias)                │     │     │  │  │
│  │  │  │   │  rotary_emb(positions, q, k)                          │     │     │  │  │
│  │  │  │   │  attn(q, k, v)  ◄── FlashAttention + KV Cache        │     │     │  │  │
│  │  │  │   │  o_proj                                               │     │     │  │  │
│  │  │  │   └───────────────────────────────────────────────────────┘     │     │  │  │
│  │  │  │        │                                                         │     │  │  │
│  │  │  │        ▼                                                         │     │  │  │
│  │  │  │   post_attention_layernorm (RMSNorm + residual)                 │     │  │  │
│  │  │  │        │                                                         │     │  │  │
│  │  │  │        ▼                                                         │     │  │  │
│  │  │  │   ┌─ Qwen3MLP ───────────────────────────────────────────┐     │     │  │  │
│  │  │  │   │  gate_up_proj ──► SiluAndMul ──► down_proj            │     │     │  │  │
│  │  │  │   └───────────────────────────────────────────────────────┘     │     │  │  │
│  │  │  │        │                                                         │     │  │  │
│  │  │  └────────┼─────────────────────────────────────── (loop) ─────────┘     │  │  │
│  │  │           ▼                                                              │  │  │
│  │  │      final_norm (RMSNorm)                                                │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘  │  │
│  │       │                                                                        │  │
│  │       ▼                                                                        │  │
│  │  lm_head ──► logits                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 算子层

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                     算子层 (nanovllm/layers/)                                         │
│                                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  linear.py        │  │  layernorm.py     │  │  activation.py   │                   │
│  │                   │  │                   │  │                   │                   │
│  │  ColumnParallel   │  │  RMSNorm          │  │  SiluAndMul       │                   │
│  │  RowParallel      │  │   rms_forward     │  │  silu(x) * y      │                   │
│  │  QKVParallel      │  │   add_rms_forward │  │                   │                   │
│  │  MergedColumn     │  │                   │  │  @torch.compile   │                   │
│  │                   │  │  @torch.compile   │  └──────────────────┘                   │
│  │  all_reduce /     │  └──────────────────┘                                         │
│  │  all_gather       │                                                                │
│  └──────────────────┘  ┌──────────────────┐  ┌──────────────────┐                   │
│                        │  attention.py     │  │  rotary_embed.py  │                   │
│  ┌──────────────────┐  │                   │  │                   │                   │
│  │  embed_head.py    │  │  Attention        │  │  RotaryEmbedding  │                   │
│  │                   │  │   prefill: varlen │  │  apply_rotary_emb │                   │
│  │  VocabParallel    │  │   decode: kvcache │  │                   │                   │
│  │  ParallelLMHead   │  │                   │  │  @torch.compile   │                   │
│  └──────────────────┘  │  store_kvcache     │  └──────────────────┘                   │
│                        │  (Triton kernel)   │                                         │
│  ┌──────────────────┐  └──────────────────┘  ┌──────────────────┐                   │
│  │  sampler.py       │                        │  context.py       │                   │
│  │                   │                        │                   │                   │
│  │  temp scaling +   │                        │  set/get/reset    │                   │
│  │  gumbel-max       │                        │  线程局部 Context  │                   │
│  └──────────────────┘                        └──────────────────┘                   │
│                                                                                      │
│  外部依赖: torch · triton · flash-attn · transformers · safetensors · xxhash         │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 请求生命周期

```
  User Prompt
      │
      ▼
  ┌─────────┐    tokenize     ┌──────────┐   add to    ┌───────────┐
  │  LLM    │ ──────────────► │ Sequence │ ──────────► │ Scheduler │
  │.generate│                 │ WAITING  │             │  waiting  │
  └────┬────┘                 └──────────┘             │  queue    │
       │                                               └─────┬─────┘
       │  step() loop                                        │
       ▼                                                     ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                        Scheduler.schedule()                      │
  │                                                                  │
  │   WAITING ──► can_allocate? ──► allocate blocks ──► PREFILL      │
  │   RUNNING ──► can_append?  ──► append slot     ──► DECODE        │
  │                                                                  │
  │           ┌─────────────────────────────────────┐                │
  │           │  BlockManager                       │                │
  │           │                                     │                │
  │           │  blocks: [B0][B1][B2]...[Bn]        │                │
  │           │          ▲                          │                │
  │           │          │ prefix cache (xxhash)    │                │
  │           │          └─ 相同前缀复用已有 block    │                │
  │           └─────────────────────────────────────┘                │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ scheduled seqs
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     ModelRunner.run()                             │
  │                                                                  │
  │   prepare_prefill / prepare_decode                               │
  │        │                                                         │
  │        ▼                                                         │
  │   set_context(cu_seqlens, slot_mapping, block_tables, ...)       │
  │        │                                                         │
  │        ▼                                                         │
  │   ┌─────────────────────────────────────────────────────────┐    │
  │   │  Qwen3ForCausalLM.forward(input_ids, positions)         │    │
  │   │                                                         │    │
  │   │   embed ──► [DecoderLayer × N] ──► norm ──► lm_head     │    │
  │   │                     │                                    │    │
  │   │                     ├── Attention: FlashAttn + KV cache  │    │
  │   │                     ├── MLP: gate_up → SiLU*mul → down   │    │
  │   │                     └── RMSNorm + residual stream        │    │
  │   └─────────────────────────────────────────────────────────┘    │
  │        │                                                         │
  │        ▼                                                         │
  │   Sampler(logits, temperatures) ──► next_token_ids               │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                   Scheduler.postprocess()                        │
  │                                                                  │
  │   seq.append_token(token_id)                                     │
  │     │                                                            │
  │     ├─ EOS or max_tokens? ──► FINISHED ──► deallocate blocks     │
  │     │                                                            │
  │     └─ otherwise ──► continue RUNNING ──► next step()            │
  └──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                          detokenize
                                 │
                                 ▼
                         Output Text ✓
```

### 9.5 架构要点

1. **分层清晰** — 4 层架构：用户 API → 引擎（调度 + 执行） → 模型 → 算子
2. **调度与执行解耦** — `Scheduler` 管"谁跑"（block 分配、队列管理），`ModelRunner` 管"怎么跑"（前向推理、CUDA Graph）
3. **Continuous Batching** — prefill 和 decode 分开调度，通过 `Context` 线程局部变量传递元数据给模型层
4. **Prefix Caching** — `BlockManager` 通过 xxhash 对 token 序列做指纹，相同前缀复用已分配的 KV cache block
5. **Tensor Parallelism** — rank 0 做调度，其它 rank 通过共享内存同步执行模型前向
6. **三处 `@torch.compile`** — `RMSNorm`、`SiluAndMul`、`RotaryEmbedding` 用编译器自动融合 elementwise 算子

---

## 10. 知识检验

### 10.1 Kernel Fusion 基础

**Q1. unfused 版本比 fused 版本慢的核心瓶颈是什么？**

- ✗ Python 解释器开销
- **✓ 多次 kernel launch + 中间 tensor 的 HBM 读写（memory-bound）**

> elementwise 算子几乎都是 memory-bound，计算量极小但需要读写整个 tensor。融合的核心收益是减少中间 tensor 的 HBM 读写，不是减少 Python 开销。

**Q2. torch.compile 在这个项目中的本质作用是什么？**

- **✓ 自动将多个 elementwise 算子融合为单个 Triton kernel，减少显存读写**

**Q3. Triton 中 tl.constexpr 参数（如 BLOCK_SIZE）的作用是什么？**

- ✗ 限制 GPU 线程数量
- **✓ 编译期常量，Triton 会为每个不同值编译一个特化 kernel**

> Triton 的 JIT 编译器会为每组 constexpr 值生成不同的 PTX 代码，可以展开循环、优化内存访问模式。

### 10.2 nano-vllm 架构

**Q4. Scheduler 和 ModelRunner 的职责分工？**

- **✓ Scheduler 决定"谁跑"（队列管理、block 分配），ModelRunner 决定"怎么跑"（前向推理、采样）**

**Q5. Context (utils/context.py) 扮演什么角色？**

- **✓ 线程局部元数据，让模型层（Attention/LMHead）无需显式传参就能获取 prefill/decode 的调度信息**

**Q6. BlockManager 的 prefix caching 怎么工作？**

- **✓ 对 token 序列做 xxhash 指纹，相同前缀复用已计算的 KV cache block，避免重复计算**

### 10.3 模型前向 & 推理优化

**Q7. Qwen3DecoderLayer 中的 residual stream 设计？**

- **答案：** 第一层 `residual=None`，用纯 `rms_forward`；后续层 `residual` 从上层传入，用 fused `add_rms_forward`。残差不经过 norm 而是直接累加传递（Pre-Norm Transformer 设计）。

**Q8. prefill 和 decode 分别使用什么执行方式？**

- **答案：** Prefill 用 eager（动态 shape），decode 用 CUDA Graph replay（固定 batch size，减少 launch 开销）。

**Q9. store_kvcache 为什么用 Triton kernel 实现？**

- **答案：** 需要按 `slot_mapping` 做散列写入，单个 Triton kernel 一次搞定 K 和 V 的写入，比 PyTorch 的 `index_copy_` 或花式索引更高效。

**Q10. 手搓 RMSNorm 的 Triton kernel 最大的难点是什么？**

- **答案：** `mean(x²)` 需要行内 reduction（`tl.sum`），不是纯 elementwise 操作，比 SiluAndMul 和 RotaryEmbedding 复杂。
