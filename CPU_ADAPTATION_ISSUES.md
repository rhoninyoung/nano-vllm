# CPU 适配 - 问题记录

## 已知注意事项

1. **torch.compile 在 CPU 上的首次编译较慢** - 首次 forward pass 会触发 Inductor 编译，需要 C++ 编译器。WSL 环境应已有 gcc。如果出问题可设置 `TORCH_COMPILE_DISABLE=1`。

2. **prefix caching 的 causal mask** - CPU SDPA 使用 `is_causal=True` 时，当 q_len < k_len，PyTorch 2.2+ 会正确处理 bottom-right aligned causal mask。项目要求 torch>=2.4.0，无问题。

3. **GQA (Grouped Query Attention)** - Qwen3 使用 GQA，num_kv_heads < num_attention_heads。CPU SDPA 需要手动 repeat_interleave KV heads。

4. **内存计算** - CPU 上没有 `torch.cuda.mem_get_info`，改用 `/proc/meminfo` 读取可用内存。仅适用于 Linux/WSL。

5. **TP (Tensor Parallelism)** - CPU TP 使用 gloo backend。TP>1 在 CPU 上性能不佳，建议 TP=1。

---

## 遇到的问题及解决方案

### Issue #1: `get_rope()` 的 `@lru_cache` 与 dict 类型 rope_scaling 不兼容
- **文件**: `nanovllm/layers/rotary_embedding.py`
- **错误**: `TypeError: unhashable type: 'dict'`
- **原因**: Qwen3 config 的 `rope_scaling` 是 dict `{'rope_theta': 1000000, 'rope_type': 'default'}`，但 `@lru_cache` 要求所有参数可哈希
- **解决**: 改用手动 dict 缓存；同时处理 `rope_type=="default"` 的情况（视为无缩放）

### Issue #2: `torch.compile` Inductor 后端需要 Python.h
- **错误**: `CppCompileError: fatal error: Python.h: No such file or directory`
- **原因**: `torch.compile` 在 CPU 上使用 Inductor 后端，会生成并编译 C++ 代码
- **解决**: `sudo apt-get install python3.12-dev`（WSL/Ubuntu）

## 测试结果

**CPU 推理测试通过！** (2026-03-01)
- 模型: Qwen3-0.6B (bfloat16)
- 设备: CPU (torch 2.10.0+cpu, WSL2)
- Prefill: ~1 tok/s（含 torch.compile 编译开销）
- Decode: ~5-10 tok/s
- 两个 prompt 均正确生成了回复
- 总运行时间约 11 分钟（大部分是 warmup 的 torch.compile 编译时间，后续运行会因缓存而快得多）
