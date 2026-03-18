已完成对整个仓库的 review，下面是用 TUI 线框图绘制的项目整体架构：

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
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          外部依赖                                                     │
│                                                                                      │
│   torch  ·  triton  ·  flash-attn  ·  transformers  ·  safetensors  ·  xxhash       │
└──────────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════
                         请求生命周期 (Request Lifecycle)
═══════════════════════════════════════════════════════════════════════════════════════

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

**架构要点总结**：

1. **分层清晰** — 4 层架构：用户 API → 引擎（调度 + 执行） → 模型 → 算子
2. **调度与执行解耦** — `Scheduler` 管"谁跑"（block 分配、队列管理），`ModelRunner` 管"怎么跑"（前向推理、CUDA Graph）
3. **Continuous Batching** — prefill 和 decode 分开调度，通过 `Context` 线程局部变量传递元数据给模型层
4. **Prefix Caching** — `BlockManager` 通过 xxhash 对 token 序列做指纹，相同前缀复用已分配的 KV cache block
5. **Tensor Parallelism** — rank 0 做调度，其它 rank 通过共享内存同步执行模型前向
6. **三处 `@torch.compile`** — `RMSNorm`、`SiluAndMul`、`RotaryEmbedding` 用编译器自动融合 elementwise 算子，这就是上次讨论"手搓替代"的目标