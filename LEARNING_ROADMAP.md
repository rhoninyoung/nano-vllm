# Nano-vLLM 项目说明与学习路线图

> 面向 LLM 零基础、希望通过「重写整个项目」来学习的同学。  
> 建议按阶段顺序推进，每阶段先理解概念再动手写代码。

---

## 一、这个项目做了什么？

**Nano-vLLM** 是一个**轻量级的大模型推理引擎**（约 1200 行 Python），功能上类似 [vLLM](https://github.com/vllm-project/vllm)，但代码更短、更易读，适合学习和二次开发。

### 1.1 核心功能

- **离线推理**：给定一段文本（prompt），模型按 token 逐个生成后续内容（completion），直到达到长度或结束符。
- **批量请求**：同时处理多条 prompt，通过调度器把「预填（prefill）」和「逐 token 解码（decode）」批在一起，提高 GPU 利用率。
- **高性能**：通过 KV Cache、PagedAttention（块化管理）、Prefix Caching、CUDA Graph、张量并行等优化，在 8GB 显存上达到与 vLLM 相当的吞吐。

### 1.2 对外接口（和 vLLM 类似）

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["你的问题"], sampling_params)
# outputs[0]["text"] 即为模型生成的回答
```

用户只需：**加载模型 → 设置采样参数 → 调用 `generate`**，内部完成 tokenize、调度、前向、采样、decode 等全部流程。

### 1.3 项目结构概览

```
nanovllm/
├── __init__.py          # 对外暴露 LLM, SamplingParams
├── llm.py                # LLM 类（薄封装 LLMEngine）
├── config.py             # 配置：模型路径、batch 上限、显存比例、KV block 等
├── sampling_params.py    # 采样参数：temperature, max_tokens, ignore_eos
├── engine/
│   ├── llm_engine.py     # 引擎：tokenizer、scheduler、model_runner、add_request/step/generate
│   ├── scheduler.py      # 调度：prefill/decode 选择、等待队列、运行队列、块分配
│   ├── block_manager.py  # KV Cache 块管理 + Prefix Caching（xxhash）
│   ├── sequence.py       # 单条请求：token 列表、block_table、状态、采样参数
│   └── model_runner.py   # 模型运行：加载 Qwen3、分配 KV、prefill/decode 数据准备、CUDA Graph、TP
├── models/
│   └── qwen3.py         # Qwen3：Embedding + DecoderLayer × N + Norm + LM Head
├── layers/
│   ├── attention.py     # 注意力 + FlashAttention + Triton 写 KV Cache
│   ├── rotary_embedding.py  # RoPE
│   ├── linear.py        # 线性层 + 张量并行（Column/Row/QKV/LMHead）
│   ├── layernorm.py     # RMSNorm（含 fused residual）
│   ├── activation.py    # SiLU (Swish) + gate
│   ├── embed_head.py    # 词嵌入 + 并行 LM Head
│   └── sampler.py       # 温度采样（softmax + 随机采样）
└── utils/
    ├── context.py       # 全局推理上下文：prefill/decode、cu_seqlens、slot_mapping、block_tables
    └── loader.py        # 从 safetensors 加载权重（含 packed 映射）
```

数据流可以简化为：

1. 用户调用 `generate(prompts, sampling_params)`  
2. 每条 prompt 被 tokenize 后变成 `Sequence`，加入 Scheduler 的等待队列  
3. 每轮 `step()`：Scheduler 决定本轮是「prefill」还是「decode」，选出一批 `Sequence`  
4. ModelRunner 根据 prefill/decode 准备 input_ids、positions、slot_mapping、block_tables 等，调用模型  
5. 模型前向得到 logits，Sampler 按 temperature 采样得到下一个 token  
6. 每个 Sequence 追加 token，更新 KV 块；若达到 EOS 或 max_tokens 则标记完成并回收块  
7. 重复 3–6 直到所有请求完成，最后把 completion token 转成文本返回  

---

## 二、学习前需要掌握的基础（按优先级）

| 优先级 | 内容 | 说明 |
|--------|------|------|
| 必须 | Python 基础 + 多进程/多线程概念 | 至少能写类、读 dataclass、知道进程间通信大致做什么 |
| 必须 | PyTorch 基础 | Tensor、nn.Module、前向传播、device、dtype |
| 必须 | Transformer 解码器 | 自回归、Masked Self-Attention、KV Cache 的作用、一层里 Attention + FFN 的结构 |
| 建议 | 词元化（Tokenization） | 文本 → token ids；vocab、BOS/EOS、chat template 有个印象即可 |
| 建议 | CUDA/GPU 内存 | 显存占用、为什么 decode 要专门优化（小 batch、重复 kernel 启动） |
| 可选 | Triton / Flash Attention | 本项目用 FlashAttention 做 attention，用 Triton 写 KV 写入 kernel，可后补 |

---

## 三、从零重写项目的路线图（分阶段）

下面按「先跑通最小闭环，再逐步加功能」的方式，给出一个可执行的 roadmap。每一阶段都建议：先读懂 Nano-vLLM 里对应文件，再自己实现一版（可先不追求性能，只求逻辑一致）。

---

### 阶段 0：环境与依赖（约 0.5 天）

- 目标：能跑通项目自带的 `example.py` 和 `bench.py`。
- 步骤：
  1. 创建虚拟环境，安装：`torch`、`triton`、`transformers`、`flash-attn`、`xxhash`（见 `pyproject.toml`）。
  2. 用 `huggingface-cli download` 下载 Qwen3-0.6B 到本地目录。
  3. 修改 `example.py` 中的模型路径，运行并看到正常生成结果。
- 产出：本地可复现的推理环境 + 对「输入 prompt → 输出文本」的直观感受。

---

### 阶段 1：单条序列、无 KV Cache、无批处理（约 2–3 天）

- 目标：实现「单条 prompt → 模型自回归生成 N 个 token」，不搞调度、不搞块管理。
- 建议顺序：
  1. **Config + SamplingParams**  
     - 读懂 `config.py`、`sampling_params.py`，自己写一份最小配置（如模型路径、max_tokens、temperature）。
  2. **Tokenizer**  
     - 用 `transformers.AutoTokenizer` 把字符串转成 `input_ids`，生成结束后再 `decode` 回文本。
  3. **模型结构（单卡、无 TP）**  
     - 只实现「能跑」的 Qwen3：  
       - `nanovllm/models/qwen3.py`：Embedding → 多层 DecoderLayer → Norm → LM Head。  
     - 先不接 engine，直接写一个脚本：`input_ids = [prompt_ids]`，循环：`logits = model(input_ids)`，取最后一个位置的 logits，用 temperature 采样得到 `next_id`，拼到 `input_ids`，直到 EOS 或达到 max_tokens。
  4. **基础层（先不优化）**  
     - `layers/embed_head.py`：普通 Embedding + 普通 Linear 做 LM Head（可先不做 TP）。  
     - `layers/linear.py`：普通 Linear（或只保留 ReplicatedLinear）。  
     - `layers/attention.py`：先不用 FlashAttention，用 PyTorch 的 `scaled_dot_product_attention` 或手写 attention，**不**写 KV Cache，每步整段序列重算（先保证正确性）。  
     - `layers/rotary_embedding.py`：RoPE 按位置编码 Q、K。  
     - `layers/layernorm.py`：RMSNorm。  
     - `layers/activation.py`：SiLU + gate。  
     - `layers/sampler.py`：temperature scaling + softmax + 随机采样下一个 token。
  5. **权重加载**  
     - 读 `utils/loader.py`，支持从 HuggingFace 格式的 safetensors 加载到当前模型（可先不支持 packed，只做简单 name → parameter 映射）。
- 产出：一个「单条序列、无 KV、无 batch」的推理脚本，能从头生成一段话。此时你对 Transformer 解码器和自回归生成已经亲手实现过一遍。

---

### 阶段 2：引入 KV Cache（单条 + 简单批 decode）（约 2–3 天）

- 目标：decode 阶段不再重算历史 token，只算新 token 的 Q，K/V 从缓存读并追加写入。
- 建议顺序：
  1. **KV Cache 的语义**  
     - 对每个 layer：维护 `k_cache`, `v_cache`，形状例如 `[max_len, num_kv_heads, head_dim]`（或按你实现的 layout）。  
     - Prefill：整段 prompt 算一次，把 K、V 写入 cache，并得到当前步的 logits（取最后一个位置）。  
     - Decode：只输入「当前一个 token」，算出 Q；K、V 只算当前步并 append 到 cache；attention 用「当前 Q」和「全量 K、V」做。
  2. **在 Attention 里接上 Cache**  
     - 读 `nanovllm/layers/attention.py` 和 `utils/context.py`：本项目用全局 Context 传 `slot_mapping`、`block_tables` 等；你先做「连续一整块」的 KV cache 版本即可（即一个 sequence 占一段连续 slot）。  
  3. **区分 Prefill / Decode**  
     - 在 ModelRunner 或等价逻辑里：第一次对某条序列算整段 prompt → prefill；之后每步只送 1 个 token → decode。  
  4. **可选：接 FlashAttention**  
     - 用 `flash_attn_with_kvcache` 做 decode 步的 attention，并按要求构造 `cache_seqlens`、`block_table`（若仍用连续 layout，可先简单传 seq_len）。  
- 产出：单条序列「prefill 一次 + 多步 decode」，每步只算一个 token，显存与时间明显优于「每步全量重算」。

---

### 阶段 3：多序列调度与 Batch（约 3–4 天）

- 目标：支持多条请求同时进行，调度器决定每轮是「一批 prefill」还是「一批 decode」。
- 建议顺序：
  1. **Sequence 与状态**  
     - 读 `engine/sequence.py`：每条请求有 token 列表、状态（WAITING / RUNNING / FINISHED）、采样参数、长度等。自己实现一个简化版（可先不做 block_table，只记 token_ids 和 length）。
  2. **Scheduler 逻辑**  
     - 读 `engine/scheduler.py`：  
       - 维护 `waiting` 和 `running` 队列。  
       - `schedule()`：若 waiting 非空，优先做 prefill（按 `max_num_batched_tokens`、`max_num_seqs` 限制拉一批）；否则对 running 做 decode（一次取一批序列，每序列 1 个 token）。  
     - 先实现「无块管理」版本：假设每条序列独占一段连续 KV 空间，不回收、不换块。
  3. **ModelRunner 的 prefill/decode 数据准备**  
     - 读 `model_runner.py` 的 `prepare_prefill` / `prepare_decode`：  
       - Prefill：把多条序列的 token 拼成 `input_ids`，并构造 `positions`、`cu_seqlens_q/k`（变长）、以及你当前 layout 下的 `slot_mapping`。  
       - Decode：每条序列只取最后一个 token，构造 `input_ids`、`positions`、每条的 `context_lens` 和 `slot_mapping`。  
  4. **Context**  
     - 读 `utils/context.py`：用全局 Context 在 prefill/decode 之间传递 `is_prefill`、`cu_seqlens_*`、`slot_mapping`、`context_lens` 等，避免在每层传一大堆参数。
  5. **LLMEngine 主循环**  
     - 读 `engine/llm_engine.py`：  
       - `add_request`：tokenize → 建 Sequence → 入 waiting。  
       - `step()`：`schedule()` 得到本轮的 seqs 和 is_prefill → `model_runner.run(seqs, is_prefill)` → 用返回的 token 更新每条 seq，若完成则移出 running、标记 FINISHED。  
       - `generate()`：把所有 prompts 依次 add_request，然后 while 循环 step 直到 is_finished，最后按 seq_id 收集 completion 并 decode 成文本。
- 产出：多请求并发的 `generate([...], sampling_params)`，内部是规范的 prefill / decode 批处理。

---

### 阶段 4：PagedAttention 与 BlockManager（约 3–4 天）

- 目标：KV 不再按「每条序列一段连续内存」，而是切成固定大小的 block，由 BlockManager 分配/回收，支持 prefix caching。
- 建议顺序：
  1. **块的概念**  
     - 读 `engine/block_manager.py` 和 `engine/sequence.py`：  
       - 每个 block 存固定长度 token 的 K/V（如 256）。  
       - 每条 Sequence 持有一个 `block_table`：`[block_id_0, block_id_1, ...]`，表示该序列的 KV 分布在哪些块里。
  2. **BlockManager**  
     - 维护 free/used block 列表；`allocate(seq)`：按 seq 当前长度算出需要的 block 数，从 free 里分配，并写入 seq.block_table；`deallocate(seq)`：把这些 block 还回 free；`can_append(seq)`：当前最后一个 block 是否还能写一个 token，不能则再分配一块。
  3. **Slot mapping**  
     - 读 `model_runner.prepare_prefill` / `prepare_decode`：每个 token 的 K/V 要写到哪一块的哪个 slot，用 `slot_mapping` 表示；attention 里根据 `block_table` + `slot_mapping` 读写 KV cache（本项目用 FlashAttention 的 block_table 接口 + Triton 写 cache）。
  4. **Prefix Caching（可选但推荐）**  
     - BlockManager 里用 xxhash 对「整块 token」算 hash；若新请求的某块与已有块内容相同，则复用该 block（ref_count++），避免重复计算和存储。读 `block_manager.py` 中 `allocate` 的 hash 逻辑。
  5. **Scheduler 与块**  
     - `schedule()` 里：prefill 前检查 `block_manager.can_allocate(seq)`；decode 前检查 `can_append(seq)`，必要时 preempt（把某条序列挤回 waiting、deallocate）再分配。
- 产出：与当前 Nano-vLLM 行为一致的块式 KV 管理 + 可选的 prefix caching，显存利用更稳、支持更长上下文。

---

### 阶段 5：性能优化（约 2–4 天，可选）

- 目标：接近当前 Nano-vLLM 的推理速度与吞吐。
- 建议顺序：
  1. **CUDA Graph（decode）**  
     - 读 `model_runner.capture_cudagraph`：对固定 batch size 的 decode 步，录制一次 CUDA graph，之后每步 replay，减少 kernel 启动开销。注意：只对 decode、且 batch size 在已录制的集合内使用 graph。
  2. **FlashAttention 全接入**  
     - Prefill：`flash_attn_varlen_func`（变长）；Decode：`flash_attn_with_kvcache`；Triton kernel 只负责把 K、V 按 `slot_mapping` 写入共享的 `k_cache`/`v_cache`。
  3. **张量并行（多卡）**  
     - 读 `model_runner.py` 多进程 + SharedMemory + `dist.init_process_group`：rank 0 发命令，其他 rank 循环读 shm 执行。  
     - 读 `layers/linear.py`、`embed_head.py`：ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding / ParallelLMHead 的切分与 all_reduce/gather。
  4. **其他**  
     - `torch.compile` 用在 RMSNorm、Sampler、Rotary 等（本项目已有）；warmup、pin_memory、non_blocking 等细节可对照 `model_runner.py` 补齐。
- 产出：在 8GB 卡上跑 Qwen3-0.6B，吞吐与官方 bench 接近。

---

## 四、建议学习顺序小结

| 阶段 | 内容 | 建议时间 |
|------|------|----------|
| 0 | 环境 + 跑通 example/bench | 0.5 天 |
| 1 | 单序列、无 KV、无 batch：模型 + 采样 + 加载 | 2–3 天 |
| 2 | KV Cache：prefill + decode，单条/简单批 | 2–3 天 |
| 3 | 多序列调度：Sequence、Scheduler、Engine、generate 循环 | 3–4 天 |
| 4 | PagedAttention + BlockManager + Prefix Caching | 3–4 天 |
| 5 | CUDA Graph、FlashAttention、张量并行等优化 | 2–4 天 |

整体约 **2–3 周**（视基础而定）。每阶段都以「先读 Nano-vLLM 对应模块，再自己实现一版」为主，这样既理解项目在做什么，又能在重写中建立从零实现推理引擎的直觉。

---

## 五、推荐阅读与参考

- **Transformer / KV Cache**：  
  - [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)  
  - [vLLM 博客：PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- **vLLM 架构**：  
  - [vLLM 论文 / 文档](https://docs.vllm.ai/) 中关于 scheduling、block manager 的部分。
- **Flash Attention**：  
  - [FlashAttention-2 论文/文档](https://github.com/Dao-AILab/flash-attention)（理解变长与 with_kvcache 的接口即可）。

---

## 六、本项目中的关键概念速查

| 概念 | 在本项目中的位置 | 一句话 |
|------|------------------|--------|
| Prefill | 第一次处理整段 prompt | 一次前向算出整段 K/V 并写入 cache，得到最后一个位置的 logits |
| Decode | 自回归每一步 | 只输入当前 1 个 token，Q 新算，K/V 追加写 cache，用全量 K/V 做 attention |
| KV Cache | 每层 Attention 的 k_cache / v_cache | 存历史 K、V，避免重复计算 |
| Block / PagedAttention | BlockManager + Sequence.block_table | KV 按块存储，块可复用、可回收，便于长序列与 prefix cache |
| Prefix Caching | BlockManager 的 xxhash + hash_to_block_id | 相同前缀的 token 块共用一个 block，省算力省显存 |
| Slot mapping | Context.slot_mapping | 每个 token 的 K/V 写入哪一块的哪个 slot（线性索引） |
| Scheduler | 选 prefill 还是 decode、选哪些 seq | 提高 GPU 利用率，控制显存与延迟 |
| Tensor Parallelism | 多进程 + Column/Row Parallel Linear | 把大矩阵切到多卡，用 all_reduce/gather 拼回 |

你可以把本文档当作「项目说明书 + 自学 roadmap」：先通读第一节弄清项目在做什么，再按阶段 0 → 1 → … → 5 边看代码边重写，遇到概念就查第六节和对应源码。祝学习顺利。
