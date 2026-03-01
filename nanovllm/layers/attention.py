import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context

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


# ==================== Triton KV Cache Store ====================

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


def _store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    valid = slot_mapping >= 0
    if not valid.any():
        return
    valid_idx = valid.nonzero(as_tuple=True)[0]
    valid_slots = slot_mapping[valid_idx]
    k_flat = k_cache.reshape(-1, D)
    v_flat = v_cache.reshape(-1, D)
    k_flat[valid_slots] = key[valid_idx].reshape(-1, D)
    v_flat[valid_slots] = value[valid_idx].reshape(-1, D)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    if HAS_TRITON and key.is_cuda:
        _store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        _store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping)


# ==================== SDPA Attention Fallback ====================

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count for GQA."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _sdpa_varlen_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, scale, block_tables,
                         k_cache, v_cache, num_heads, num_kv_heads, block_size):
    """SDPA-based variable-length prefill attention."""
    num_seqs = cu_seqlens_q.shape[0] - 1
    n_rep = num_heads // num_kv_heads
    head_dim = q.shape[-1]
    outputs = []

    for i in range(num_seqs):
        q_s = cu_seqlens_q[i].item()
        q_e = cu_seqlens_q[i + 1].item()
        qi = q[q_s:q_e]  # [seq_q, num_heads, head_dim]

        if block_tables is not None:
            k_len = cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()
            num_blocks = (k_len + block_size - 1) // block_size
            block_ids = block_tables[i, :num_blocks]
            ki = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:k_len]
            vi = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:k_len]
        else:
            k_s = cu_seqlens_k[i].item()
            k_e = cu_seqlens_k[i + 1].item()
            ki = k[k_s:k_e]
            vi = v[k_s:k_e]

        # [1, heads, seq, head_dim]
        qi = qi.permute(1, 0, 2).unsqueeze(0)
        ki = ki.permute(1, 0, 2).unsqueeze(0)
        vi = vi.permute(1, 0, 2).unsqueeze(0)

        ki = _repeat_kv(ki, n_rep)
        vi = _repeat_kv(vi, n_rep)

        seq_q = qi.shape[2]
        seq_k = ki.shape[2]

        if seq_q == seq_k:
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=True)
        else:
            attn_mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=qi.device).tril(diagonal=seq_k - seq_q)
            oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask, scale=scale)

        oi = oi.squeeze(0).permute(1, 0, 2)  # [seq_q, num_heads, head_dim]
        outputs.append(oi)

    return torch.cat(outputs, dim=0)


def _sdpa_decode(q, k_cache, v_cache, context_lens, block_tables,
                 scale, num_heads, num_kv_heads, block_size):
    """SDPA-based decode attention with paged KV cache."""
    batch_size = q.shape[0]
    n_rep = num_heads // num_kv_heads
    head_dim = q.shape[-1]
    outputs = []

    for i in range(batch_size):
        qi = q[i:i + 1]  # [1, 1, num_heads, head_dim]
        seq_len = context_lens[i].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_tables[i, :num_blocks]

        ki = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        vi = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]

        # qi: [1, 1, num_heads, head_dim] → [1, num_heads, 1, head_dim]
        qi = qi.permute(0, 2, 1, 3)
        # ki: [seq_len, num_kv_heads, head_dim] → [1, num_kv_heads, seq_len, head_dim]
        ki = ki.permute(1, 0, 2).unsqueeze(0)
        vi = vi.permute(1, 0, 2).unsqueeze(0)

        ki = _repeat_kv(ki, n_rep)
        vi = _repeat_kv(vi, n_rep)

        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False)
        oi = oi.permute(0, 2, 1, 3)  # [1, 1, num_heads, head_dim]
        outputs.append(oi)

    return torch.cat(outputs, dim=0)  # [batch, 1, num_heads, head_dim]


# ==================== Attention Module ====================

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

    def _forward_flash(self, q, k, v, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
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

    def _forward_sdpa(self, q, k, v, context):
        k_cache, v_cache = self.k_cache, self.v_cache
        block_size = k_cache.shape[1] if k_cache.ndim == 4 else 256
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                o = _sdpa_varlen_prefill(
                    q, None, None,
                    context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    self.scale, context.block_tables,
                    k_cache, v_cache, self.num_heads, self.num_kv_heads, block_size)
            else:
                o = _sdpa_varlen_prefill(
                    q, k, v,
                    context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    self.scale, None,
                    None, None, self.num_heads, self.num_kv_heads, block_size)
        else:
            o = _sdpa_decode(
                q.unsqueeze(1), k_cache, v_cache,
                context.context_lens, context.block_tables,
                self.scale, self.num_heads, self.num_kv_heads, block_size)
        return o

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if HAS_FLASH_ATTN and q.is_cuda:
            return self._forward_flash(q, k, v, context)
        return self._forward_sdpa(q, k, v, context)
