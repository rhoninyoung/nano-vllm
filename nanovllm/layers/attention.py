import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from nanovllm.utils.context import get_context


# --------------- GPU path (Triton + FlashAttention) ---------------

if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_gpu(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# --------------- CPU path (pure PyTorch) ---------------

def store_kvcache_cpu(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Store K/V into paged cache using PyTorch indexing (CPU-compatible)."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    valid = slot_mapping >= 0
    if not valid.any():
        return
    valid_slots = slot_mapping[valid]
    k_cache.view(-1, D)[valid_slots] = key[valid].reshape(-1, D)
    v_cache.view(-1, D)[valid_slots] = value[valid].reshape(-1, D)


def gather_kv_from_blocks(k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int, block_size: int):
    """Gather K/V from paged block cache into a contiguous (seq_len, heads, dim) tensor."""
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    block_ids = block_table[:num_blocks_needed].long()
    k = k_cache[block_ids].reshape(-1, k_cache.size(-2), k_cache.size(-1))[:seq_len]
    v = v_cache[block_ids].reshape(-1, v_cache.size(-2), v_cache.size(-1))[:seq_len]
    return k, v


def _expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, num_q_heads: int):
    """Repeat KV heads to match Q heads for GQA."""
    num_kv_heads = k.size(-3)
    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=-3)
        v = v.repeat_interleave(repeat, dim=-3)
    return k, v


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _forward_gpu(self, q, k, v, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache_gpu(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _forward_cpu(self, q, k, v, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache_cpu(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self._prefill_cpu(q, k, v, k_cache, v_cache, context)
        else:
            return self._decode_cpu(q, k_cache, v_cache, context)

    def _prefill_cpu(self, q, k, v, k_cache, v_cache, context):
        cu_seqlens_q = context.cu_seqlens_q
        cu_seqlens_k = context.cu_seqlens_k
        block_tables = context.block_tables
        block_size = k_cache.size(1) if k_cache.numel() else 256
        batch_size = cu_seqlens_q.size(0) - 1
        outputs = []

        for i in range(batch_size):
            q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            q_i = q[q_s:q_e]

            if block_tables is not None:
                k_len = cu_seqlens_k[i + 1].item()
                k_i, v_i = gather_kv_from_blocks(k_cache, v_cache, block_tables[i], k_len, block_size)
            else:
                k_s, k_e = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
                k_i = k[k_s:k_e]
                v_i = v[k_s:k_e]

            # (seq, heads, dim) -> (1, heads, seq, dim) for SDPA
            q_i = q_i.transpose(0, 1).unsqueeze(0)
            k_i = k_i.transpose(0, 1).unsqueeze(0)
            v_i = v_i.transpose(0, 1).unsqueeze(0)
            k_i, v_i = _expand_kv_for_gqa(k_i, v_i, self.num_heads)

            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True, scale=self.scale)
            outputs.append(o_i.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)

    def _decode_cpu(self, q, k_cache, v_cache, context):
        context_lens = context.context_lens
        block_tables = context.block_tables
        block_size = k_cache.size(1)
        batch_size = q.size(0)
        outputs = []

        for i in range(batch_size):
            seq_len = context_lens[i].item()
            k_i, v_i = gather_kv_from_blocks(k_cache, v_cache, block_tables[i], seq_len, block_size)

            # q_i: (num_heads, 1, head_dim), k/v: (num_kv_heads, seq_len, head_dim)
            q_i = q[i].unsqueeze(1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)
            k_i, v_i = _expand_kv_for_gqa(k_i, v_i, self.num_heads)

            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False, scale=self.scale)
            outputs.append(o_i.squeeze(1))

        # (batch, num_heads, head_dim) -> (batch, 1, num_heads, head_dim)
        return torch.stack(outputs).unsqueeze(1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if q.is_cuda and HAS_FLASH_ATTN:
            return self._forward_gpu(q, k, v, context)
        return self._forward_cpu(q, k, v, context)
