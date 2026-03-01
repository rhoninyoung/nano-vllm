# 测试里程碑：按阶段与 Feature 的用例清单

> 与 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 对应。每完成一个「参考开发 / 独立开发」的 feature，可用本页中对应测试用例验证实现是否正确。  
> 建议测试框架：**pytest**（单元/集成），需 GPU 的用例可加 `@pytest.mark.gpu` 或单独脚本运行。

---

## 使用说明

- **用例类型**：`单元`（不依赖模型/GPU）、`集成`（多模块协作）、`E2E`（端到端，需模型或小模型）。
- **运行示例**：若用例写在 `tests/` 下，可用 `pytest tests/ -v`；仅跑某阶段可用 `pytest tests/ -v -k "phase1"`（需在用例名或 marker 中体现阶段）。
- **模型依赖**：标注「需模型」的用例需要本地已有 Qwen3-0.6B（或你实现的等价小模型路径），否则可跳过或 mock。

---

## 阶段 0：环境与依赖

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 0.1 | 环境与依赖安装 | 所有依赖可导入 | 单元 | `import torch`, `import transformers`, `import flash_attn`, `import triton`, `import xxhash` 不报错 |
| 0.2 | 模型可下载/可读 | 指定目录下存在模型文件 | 单元 | 模型目录存在 `config.json` 及至少一个 `.safetensors` |
| 0.3 | example 跑通 | 官方 example 能完成一次生成 | E2E | 运行 `example.py` 无异常，输出为非空字符串 |
| 0.4 | bench 跑通 | 官方 bench 能跑完并输出吞吐 | E2E | 运行 `bench.py` 无异常，打印 Total/Time/Throughput |

**建议**：阶段 0 通过后再开始阶段 1 的开发与测试。

---

## 阶段 1：单条序列、无 KV Cache、无批处理

### 1.1 Config + SamplingParams

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.1.1 | Config 合法路径 | 有效模型目录能构造 Config | 单元 | `Config(valid_model_dir)` 不抛异常，`hf_config` 非 None |
| 1.1.2 | Config 非法路径 | 非法路径或非目录应报错 | 单元 | `Config("/nonexistent")` 或 `Config("/tmp")`（无 config.json）抛出 AssertionError 或等价异常 |
| 1.1.3 | SamplingParams 合法 | temperature>0, max_tokens>0 可构造 | 单元 | `SamplingParams(temperature=0.6, max_tokens=64)` 正常 |
| 1.1.4 | SamplingParams 非法 | temperature 过小视为贪婪应报错 | 单元 | `SamplingParams(temperature=0)` 或 `1e-11` 在 `__post_init__` 中触发断言/异常 |

### 1.2 Tokenizer

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.2.1 | encode/decode 往返 | 文本 → token_ids → 文本 无损（或可接受差异） | 单元 | 对若干短句 `decode(encode(text)) == text` 或与原文语义一致（某些 tokenizer 会加空格） |
| 1.2.2 | EOS 存在 | tokenizer 有 eos_token_id | 单元 | `tokenizer.eos_token_id` 存在且为 int |

### 1.3 基础层（可先不依赖模型权重）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.3.1 | RMSNorm forward | 输出形状、数值稳定 | 单元 | 随机输入 `(B, L, D)`，输出形状相同；输入全零时输出接近零（除 scale） |
| 1.3.2 | RMSNorm add residual | pre-norm 形式 x+residual 再 norm | 单元 | 与「先加 residual 再调用 rms_forward」结果一致 |
| 1.3.3 | RoPE forward | 相同位置相同输入得到相同输出 | 单元 | 同一 positions、同一 q/k 两次 forward 结果一致；不同 position 结果不同 |
| 1.3.4 | SiluAndMul | gate 结构：chunk(2) 后 silu(x)*y | 单元 | 与手写 `F.silu(x1)*x2` 结果一致 |
| 1.3.5 | Sampler temperature | 高 temperature 更随机、低 temperature 更确定 | 单元 | 固定 seed，同一 logits：temperature 小则多次采样相同；temperature 大则允许不同（可做统计） |
| 1.3.6 | Sampler 输出形状 | 输入 (B, V)，输出 (B,) | 单元 | `sampler(logits, temps).shape == (batch_size,)`，dtype 整数 |

### 1.4 线性层 / Embedding（无 TP 时）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.4.1 | ReplicatedLinear / 普通 Linear | 前向形状正确 | 单元 | 输入 (B, L, I)，输出 (B, L, O)；与 PyTorch `nn.Linear` 同输入同权重结果一致 |
| 1.4.2 | Embedding | 查表形状正确 | 单元 | 输入 token_ids (B, L)，输出 (B, L, D)；越界 id 可约定报错或按实现约定 |

### 1.5 Attention（无 KV Cache、无 FlashAttn 的简易版）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.5.1 | 单步 attention 形状 | 输入 q,k,v 输出 o，形状一致 | 单元 | (B, L, H, D) 的 q,k,v → o 为 (B, L, H, D) 或 (B, L, H*D) 视实现 |
| 1.5.2 | Causal mask | 位置 i 只看到 0..i | 单元 | 将位置 L-1 的 k 置零，仅位置 L-1 的 q 对应输出应变化；或与参考 causal attention 数值接近 |

### 1.6 权重加载

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.6.1 | 加载后权重非零 | 从 safetensors 加载到模型后，主要参数非全零 | 集成 | 加载 Qwen3（或你的实现）后，至少检查 embed 和一层 linear 的 weight 范数 > 0 |
| 1.6.2 | 加载后前向可跑 | 加载权重后一次 forward 不报错 | 集成 | `model(input_ids, positions)` 不抛异常，输出形状符合预期（需模型） |

### 1.7 单条序列自回归生成（无 KV、无 batch）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 1.7.1 | 生成长度受 max_tokens 限制 | 不超过 max_tokens | E2E | 设 max_tokens=5，ignore_eos=True，completion 长度恰好 5（需模型） |
| 1.7.2 | 遇到 EOS 停止 | 采样到 eos_token_id 时停止 | E2E | 若 prompt 易触发短回复，completion 在 EOS 处截断，长度 ≤ max_tokens（需模型） |
| 1.7.3 | 输出为合法 token 序列 | 所有 token 在 vocab 内 | E2E | 所有 completion token_ids 在 [0, vocab_size)（需模型） |

---

## 阶段 2：引入 KV Cache

### 2.1 KV Cache 语义（单层或全模型）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 2.1.1 | Prefill 写满 cache | 一次 prefill 后，cache 中前 len(prompt) 个 slot 被写入 | 集成 | Prefill 后检查 k_cache/v_cache 对应 slot 非零（或与手工算的 K/V 一致） |
| 2.1.2 | Decode 只追加不覆盖 | 每次 decode 只在「当前 slot」写入，历史 slot 不变 | 集成 | Prefill 后记下某历史 slot 的值；再 decode 一步，该 slot 不变 |
| 2.1.3 | Prefill + 多步 Decode 与 无 Cache 全量重算一致 | 数值一致性（同一 seed） | 集成 | 同一 prompt，两种方式：(A) prefill + N 步 decode；(B) 每步用整段 input_ids 做一次 forward。在相同采样 seed 下，每步采样出的 token 一致（需模型） |

### 2.2 Prefill / Decode 分支

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 2.2.1 | Prefill 输入长度 = prompt 长度 | 传入的 input_ids/positions 长度等于该序列 prompt 长度 | 单元/集成 | 在 prepare_prefill 或等价处，对单条 seq，len(input_ids)==len(seq) |
| 2.2.2 | Decode 输入长度 = batch 大小 | 每条序列只 1 个 token | 单元/集成 | 在 prepare_decode 或等价处，len(input_ids)==len(seqs)，且每个 position 为 len(seq)-1 |

### 2.3 单条 + 简单批 Decode

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 2.3.1 | 多条序列 decode 互不干扰 | 两条不同 prompt，各自生成序列独立 | 集成 | 两条 prompt A、B，分别单独生成得 out_a、out_b；再 batch 同时生成得 out_a'、out_b'，在相同 seed 下 out_a==out_a'、out_b==out_b'（需模型） |

---

## 阶段 3：多序列调度与 Batch

### 3.1 Sequence 与状态

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 3.1.1 | 创建后为 WAITING | 新 Sequence 状态为 WAITING | 单元 | `Sequence([1,2,3], sp).status == SequenceStatus.WAITING` |
| 3.1.2 | append_token 更新 last_token 与 num_tokens | 每 append 一次长度+1 | 单元 | 初始 len(seq)==3，append_token(5) 后 len(seq)==4，last_token==5 |
| 3.1.3 | num_completion_tokens / num_prompt_tokens | 与 token_ids 划分一致 | 单元 | prompt 长 4，append 2 次后 num_completion_tokens==2，completion_token_ids 长度为 2 |
| 3.1.4 | is_finished | 状态为 FINISHED 时 is_finished 为 True | 单元 | 将 status 设为 FINISHED，assert seq.is_finished |

### 3.2 Scheduler

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 3.2.1 | 空队列 is_finished | 无请求时 is_finished 为 True | 单元 | 新 Scheduler，assert is_finished() |
| 3.2.2 | 有等待请求时先 prefill | 若干 WAITING 加入后，第一次 schedule 返回 is_prefill=True | 单元 | add 2 个短 seq，schedule() 返回 (seqs, True)，且 len(seqs)>=1 |
| 3.2.3 | 无等待时 decode | waiting 空、running 非空时，schedule 返回 is_prefill=False | 单元 | 先 add 再 schedule 一次（prefill），不 postprocess 完成；再 schedule 应返回 is_prefill=False |
| 3.2.4 | 完成序列被移出 running | postprocess 中标记 FINISHED 并 remove | 单元 | 模拟 postprocess(seqs, [eos_id]*len(seqs))，running 中不再包含这些 seq |
| 3.2.5 | max_num_seqs / max_num_batched_tokens 限制 | 单次 schedule 不超过限制 | 单元 | 加入多条长 seq，schedule 返回的 seqs 数量及总 token 数满足配置上限 |

### 3.3 Context

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 3.3.1 | set_context / get_context | 设置后能正确取出 | 单元 | set_context(True, ...)，get_context().is_prefill 为 True；reset 后为默认 |
| 3.3.2 | Prefill 时 cu_seqlens 与 slot_mapping 长度一致 | 与 input 长度匹配 | 集成 | 在 prepare_prefill 后，context 中 cu_seqlens_q[-1] == 总新 token 数，slot_mapping 长度与之一致 |

### 3.4 LLMEngine / generate

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 3.4.1 | generate 返回数量等于 prompt 数量 | 每个 prompt 对应一个输出 | E2E | generate(["a","b","c"], sp) 返回 len(outputs)==3（需模型） |
| 3.4.2 | 返回顺序与 prompt 顺序一致 | outputs[i] 对应 prompts[i] | E2E | 同上，顺序与输入一致（按 seq_id 排序后与 prompts 顺序一致） |
| 3.4.3 | 单条与多条中单条结果一致 | 相同 prompt、相同 seed，单条生成与批量生成中该条一致 | E2E | 固定 seed，单条 generate([p], sp)[0] 与 generate([p,q], sp)[0] 的 text 或 token_ids 一致（需模型） |
| 3.4.4 | 不同 SamplingParams 生效 | max_tokens 不同则长度不同 | E2E | 同一 prompt，max_tokens=2 与 max_tokens=10，completion 长度不同（需模型） |

---

## 阶段 4：PagedAttention 与 BlockManager

### 4.1 Block 与 BlockManager（不依赖 GPU）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 4.1.1 | 初始化 free_block_ids 数量 | 等于 num_blocks | 单元 | BlockManager(10, 256)，len(free_block_ids)==10，used 为空 |
| 4.1.2 | can_allocate | 需要块数 ≤ free 数量时 True | 单元 | 构造一 seq 长度为 256*2，num_blocks==2；can_allocate(seq) 为 True；若 free 仅 1 块则为 False |
| 4.1.3 | allocate 后 block_table 与 free 减少 | 分配后 seq.block_table 长度正确，free 减少对应数量 | 单元 | allocate(seq)，len(seq.block_table)==seq.num_blocks，len(free_block_ids)==初始-占用数 |
| 4.1.4 | deallocate 后块回收到 free | 释放后 used 减少、free 增加 | 单元 | allocate(seq) 后 deallocate(seq)，free 恢复为初始数量，seq.block_table 清空 |
| 4.1.5 | can_append | 最后一块未满或可新分配一块时为 True | 单元 | 满块时（len(seq)%block_size==0）再 append 需新块，can_append 取决于是否还有 free |
| 4.1.6 | may_append 在满块时分配新块 | 当 len(seq)%block_size==1 时 block_table 增加 1 | 单元 | 从 256 到 257 token 时，may_append 后 len(block_table) 增加 1 |

### 4.2 Prefix Caching（BlockManager hash 逻辑）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 4.2.1 | 相同 token 块复用同一 block | 两条 seq 前缀相同，第二条 allocate 时前缀块 ref_count>1 | 单元 | 两条 seq 前 256 个 token 相同，先 allocate(seq1)，再 allocate(seq2)，seq2 的前缀 block 与 seq1 共用，对应 block ref_count==2 |
| 4.2.2 | 不同 token 块不复用 | 内容不同则不同 block_id | 单元 | 两 seq 仅最后一格不同，第二块 block_id 不同 |
| 4.2.3 | compute_hash 确定性 | 相同输入相同 hash | 单元 | BlockManager.compute_hash([1,2,3]) 两次调用结果相同 |

### 4.3 Slot mapping 与 Scheduler 集成

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 4.3.1 | Prefill slot_mapping 与 block_table 一致 | 每个 token 的 slot 落在其对应 block 内 | 集成 | prepare_prefill 得到的 slot_mapping，与根据 block_table 和 block_size 推算的 slot 一致 |
| 4.3.2 | Decode slot_mapping 为当前步写入位置 | 每条 seq 一个 slot，为最后一块的当前偏移 | 集成 | prepare_decode 中，slot_mapping[i] 对应 seqs[i] 的「下一格」写入位置 |
| 4.3.3 | 块不足时 preempt | 无 free 块时某 running seq 被挤回 waiting 并 deallocate | 单元/集成 | 占满所有块后，再 schedule decode 且需要新块时，发生 preempt，该 seq 回到 waiting |

### 4.4 E2E：带 Block 的 generate

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 4.4.1 | 多请求带块管理完整跑通 | 与阶段 3 行为一致，仅底层为块管理 | E2E | generate 多条 prompt，输出条数、顺序、每条长度约束与阶段 3 一致（需模型） |
| 4.4.2 | 长序列不 OOM | 总 token 超过单段连续 cache 时仍能跑 | E2E | 较长 prompt + 较长 max_tokens，不因显存崩溃（需模型 + 足够块数） |

---

## 阶段 5：性能优化（可选）

### 5.1 CUDA Graph

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 5.1.1 | Replay 与 Eager 结果一致 | 同一输入，graph.replay() 与直接 forward 数值一致 | 集成 | 对固定 bs 的 decode，capture 后 replay 的 logits 与 enforce_eager 下同输入 logits 一致（需模型） |
| 5.1.2 | 仅 decode 使用 graph | Prefill 或大 batch 不走 graph | 集成 | 代码路径上 prefill 或 bs>512 时调用非 graph 分支（可读代码或打 log 验证） |

### 5.2 FlashAttention

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 5.2.1 | Prefill 与参考 attention 数值接近 | 小规模下与 PyTorch SDPA causal 对比 | 集成 | 短序列、小 head，flash_attn_varlen_func 与 torch 实现结果接近（允许少量数值误差） |
| 5.2.2 | Decode with_kvcache 与逐 token 结果一致 | 与「不用 kvcache、整段重算」在 decode 步一致 | 集成 | 同 2.1.3，但使用 FlashAttention + KV cache 实现 |

### 5.3 张量并行（多卡）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 5.3.1 | 2 卡与 1 卡结果一致 | 相同 prompt、seed，tp=2 与 tp=1 生成相同 | E2E | 需 2 张 GPU；固定 seed，compare generate(..., tensor_parallel_size=1) 与 2（需模型） |
| 5.3.2 | 多进程与 shm 通信 | rank>0 能收到 rank0 的调用并执行 | 集成 | 可打 log 或简单 echo 测试：rank0 write_shm，rank1 read_shm 并执行一次 run（需多卡环境） |

### 5.4 吞吐 / 显存（可选）

| # | Feature | 用例描述 | 类型 | 如何验证 / 断言 |
|---|--------|----------|------|------------------|
| 5.4.1 | 吞吐不低于参考 | 与 README/bench 中给出的 tok/s 同量级 | E2E | 在相同硬件与配置下，bench 脚本的 throughput 不低于文档示例的约 80%（需模型 + 相同环境） |
| 5.4.2 | 显存不超配置 | 使用 gpu_memory_utilization 时，峰值显存在预期内 | E2E | 监控 GPU 显存，峰值不超过 total * gpu_memory_utilization + 一定余量 |

---

## 如何组织测试代码（建议）

- 在项目根目录建 `tests/`。
- 按阶段分子目录或文件名，例如：  
  `tests/test_phase1_config.py`, `tests/test_phase1_layers.py`, `tests/test_phase2_kvcache.py`, `tests/test_phase3_scheduler.py`, `tests/test_phase4_block_manager.py`, `tests/test_phase5_cudagraph.py`。
- 需要 GPU/模型的用例可加：  
  `@pytest.mark.gpu` 或 `@pytest.mark.requires_model`，用 `pytest -m "not requires_model"` 在无模型环境只跑单元测试。
- 模型路径可从环境变量读取，例如：  
  `MODEL_PATH = os.environ.get("NANOVLLM_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/"))`，无路径时跳过需模型的用例。

完成一个 feature 后，在本文档对应表格里打勾或注明通过情况，便于追踪进度。
