import torch
import time
import statistics
import logging
from typing import List, Tuple, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_tensors(size: int = 1_000_000) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    return x, y


# ── 1. Unfused: 每个算子独立 launch，产生中间 tensor ──────────────────────
def unfused_sigmoid_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * y


# ── 2. torch.compile 自动融合 (Inductor / Triton backend) ─────────────────
@torch.compile(mode="max-autotune")
def compiled_sigmoid_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * y


# ── 3. 手写 Triton kernel —— 真正的单 kernel 融合 ─────────────────────────
HAS_TRITON = False
if DEVICE == "cuda":
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _fused_sigmoid_mul_kernel(
            x_ptr, y_ptr, out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)

            sigmoid_x = tl.sigmoid(x)
            result = sigmoid_x * y

            tl.store(out_ptr + offsets, result, mask=mask)

        def triton_sigmoid_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            BLOCK_SIZE = 1024
            grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _fused_sigmoid_mul_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
            return out

        HAS_TRITON = True
    except ImportError:
        logger.warning("Triton not available, skipping Triton kernel benchmark")


# ── Benchmark 工具 ────────────────────────────────────────────────────────
def benchmark_cuda(
    func: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_iters: int = 200,
    benchmark_iters: int = 1000,
) -> List[float]:
    """使用 CUDA events 精确计时"""
    for _ in range(warmup_iters):
        func(x, y)
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(benchmark_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func(x, y)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def benchmark_cpu(
    func: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_iters: int = 50,
    benchmark_iters: int = 5000,
) -> List[float]:
    for _ in range(warmup_iters):
        func(x, y)

    times: List[float] = []
    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        func(x, y)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return times


def calc_stats(times: List[float]) -> dict:
    s = sorted(times)
    n = len(s)
    return {
        "p50": statistics.median(s),
        "p90": s[int(n * 0.90)],
        "p99": s[int(n * 0.99)],
        "mean": statistics.mean(s),
        "std": statistics.stdev(s),
    }


def print_table(results: dict[str, dict]):
    names = list(results.keys())
    baseline = results[names[0]]

    header = f"{'Method':<25} | {'P50 (ms)':>10} | {'P90 (ms)':>10} | {'P99 (ms)':>10} | {'Mean (ms)':>10} | {'Speedup':>8}"
    logger.info("")
    logger.info(header)
    logger.info("-" * len(header))
    for name, stats in results.items():
        speedup = baseline["mean"] / stats["mean"]
        logger.info(
            f"{name:<25} | {stats['p50']:>10.4f} | {stats['p90']:>10.4f} | "
            f"{stats['p99']:>10.4f} | {stats['mean']:>10.4f} | {speedup:>7.2f}x"
        )


def plot_results(results: dict[str, List[float]]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    data = [results[n] for n in names]

    axes[0].boxplot(data, tick_labels=names, showfliers=False)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency Distribution")

    stats = {n: calc_stats(results[n]) for n in names}
    percentiles = ["p50", "p90", "p99"]
    x = range(len(percentiles))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [stats[name][p] for p in percentiles]
        offset = (i - len(names) / 2 + 0.5) * width
        axes[1].bar([xi + offset for xi in x], vals, width, label=name)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([p.upper() for p in percentiles])
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Percentile Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    plt.close()
    logger.info("Plot saved to benchmark_results.png")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    logger.info(f"Device: {DEVICE}")
    x, y = create_tensors()
    logger.info(f"Tensor size: {x.numel():,}  dtype: {x.dtype}")

    benchmark = benchmark_cuda if DEVICE == "cuda" else benchmark_cpu

    methods = {"Unfused (eager)": unfused_sigmoid_mul}

    # torch.compile warmup: 首次调用触发编译
    logger.info("Compiling with torch.compile (this may take a moment)...")
    compiled_sigmoid_mul(x, y)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    methods["torch.compile"] = compiled_sigmoid_mul

    if HAS_TRITON:
        methods["Triton kernel"] = triton_sigmoid_mul

    all_times: dict[str, List[float]] = {}
    all_stats: dict[str, dict] = {}
    for name, func in methods.items():
        logger.info(f"Benchmarking: {name}")
        times = benchmark(func, x, y)
        all_times[name] = times
        all_stats[name] = calc_stats(times)

    print_table(all_stats)

    # 正确性验证
    ref = unfused_sigmoid_mul(x, y)
    for name, func in methods.items():
        if name == "Unfused (eager)":
            continue
        out = func(x, y)
        ok = torch.allclose(ref, out, atol=1e-5)
        logger.info(f"Correctness [{name}]: {'PASS' if ok else 'FAIL'}")

    plot_results(all_times)


if __name__ == "__main__":
    main()
