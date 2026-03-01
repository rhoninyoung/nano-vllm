# AMD 适配问题记录

## 已知限制
1. WSL2 中 AMD 集成GPU (Vega 8) 无 ROCm 支持 → 使用 CPU 模式测试
2. 系统 RAM 仅 7.6GB → 只能测试小模型
3. pip3 未安装 → 需要先安装

## 遇到的问题

### Issue 1: PyTorch 2.10.0+cpu 与 Python 3.12 的 torch.compile 不兼容
- 错误: `TypeError: Too few arguments for <class 'torch._inductor.codegen.common.CSE'>; actual 1, expected 2`
- 原因: PyTorch 2.10.0 的 inductor CuteDSL 代码有 typing 兼容性问题
- 解决: 降级到 PyTorch 2.6.0 或使 torch.compile 在失败时优雅降级

### Issue 2: transformers 5.2.0 将 rope_scaling=null 转为 dict
- 错误: `TypeError: unhashable type: 'dict'` in `get_rope` with `@lru_cache`
- 原因: 新版 transformers 将 config.json 中 `"rope_scaling": null` 解析为 `{'rope_theta': ..., 'rope_type': 'default'}` 
- 解决: 移除 `@lru_cache`, 改用手动 dict 缓存, 忽略 rope_scaling 参数
