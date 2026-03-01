"""
CPU benchmark for nano-vllm, with torch.compile comparison.

Usage:
    python bench_cpu.py                        # default with torch.compile
    python bench_cpu.py --no-compile           # without torch.compile (eager)
    python bench_cpu.py --compare              # run both modes and show speedup
    python bench_cpu.py --compare --num_seqs 8 --max_output 128
"""
import os
import sys
import json
import argparse
import subprocess
import time
from random import randint, seed


def build_workload(num_seqs, max_input, max_output, rng_seed):
    seed(rng_seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(max_input // 2, max_input))]
        for _ in range(num_seqs)
    ]
    from nanovllm import SamplingParams
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=randint(max_output // 2, max_output))
        for _ in range(num_seqs)
    ]
    total_input = sum(len(p) for p in prompt_token_ids)
    total_output = sum(sp.max_tokens for sp in sampling_params)
    return prompt_token_ids, sampling_params, total_input, total_output


def run_single(args):
    """Run a single benchmark and return results dict."""
    compile_mode = "compile" if not args.no_compile else "eager"

    if args.no_compile:
        import torch._dynamo
        torch._dynamo.config.disable = True

    from nanovllm import LLM, SamplingParams

    prompt_token_ids, sampling_params, total_input, total_output = \
        build_workload(args.num_seqs, args.max_input, args.max_output, args.seed)

    # --- Phase 1: Model Loading ---
    t0 = time.time()
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1)
    load_time = time.time() - t0

    # --- Phase 2: Warmup (triggers torch.compile JIT) ---
    t0 = time.time()
    llm.generate(["warmup"], SamplingParams(max_tokens=8))
    warmup_time = time.time() - t0

    # --- Phase 3: Benchmark ---
    t0 = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    bench_time = time.time() - t0

    throughput = total_output / bench_time

    results = dict(
        mode=compile_mode,
        total_input=total_input,
        total_output=total_output,
        load_time=round(load_time, 2),
        warmup_time=round(warmup_time, 2),
        bench_time=round(bench_time, 2),
        throughput=round(throughput, 2),
        latency_ms=round(1000 / throughput, 1),
    )

    if args.json_output:
        print(json.dumps(results))
    else:
        print(f"\n{'='*45}")
        print(f"  Mode: {'torch.compile' if compile_mode == 'compile' else 'eager (no compile)'}")
        print(f"{'='*45}")
        print(f"  Input tokens:    {total_input}")
        print(f"  Output tokens:   {total_output}")
        print(f"  Model load:      {results['load_time']:.2f}s")
        print(f"  Warmup (1st gen): {results['warmup_time']:.2f}s")
        print(f"  Benchmark:       {results['bench_time']:.2f}s")
        print(f"  Throughput:      {results['throughput']:.2f} tok/s")
        print(f"  Token latency:   {results['latency_ms']:.1f} ms")
        print(f"{'='*45}")

    return results


def run_compare(args):
    """Run both compile and eager modes via subprocesses, then compare."""
    print(f"{'='*55}")
    print(f"  nano-vllm CPU Benchmark — torch.compile comparison")
    print(f"{'='*55}")
    print(f"  Model:       {args.model}")
    print(f"  Sequences:   {args.num_seqs}")
    print(f"  Max input:   {args.max_input} tokens")
    print(f"  Max output:  {args.max_output} tokens")
    print(f"{'='*55}\n")

    base_cmd = [
        sys.executable, __file__,
        "--model", args.model,
        "--num_seqs", str(args.num_seqs),
        "--max_input", str(args.max_input),
        "--max_output", str(args.max_output),
        "--seed", str(args.seed),
        "--json-output",
    ]

    results = {}
    for label, extra_flags in [("eager", ["--no-compile"]), ("compile", [])]:
        print(f"[{label}] Running... ", end="", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            base_cmd + extra_flags,
            capture_output=True, text=True,
        )
        wall = time.time() - t0
        print(f"done ({wall:.0f}s wall)")

        if proc.returncode != 0:
            print(f"[{label}] FAILED (exit code {proc.returncode})")
            print(proc.stderr[-2000:] if proc.stderr else "(no stderr)")
            return

        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                results[label] = json.loads(line)
                break
        else:
            print(f"[{label}] Could not parse JSON output")
            print(proc.stdout[-1000:])
            return

    eager = results["eager"]
    compiled = results["compile"]
    speedup = compiled["throughput"] / eager["throughput"] if eager["throughput"] > 0 else float("inf")

    print(f"\n{'='*55}")
    print(f"  {'Metric':<25} {'Eager':>10} {'Compile':>10} {'Ratio':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Model load (s)':<25} {eager['load_time']:>10.2f} {compiled['load_time']:>10.2f}")
    print(f"  {'Warmup / 1st gen (s)':<25} {eager['warmup_time']:>10.2f} {compiled['warmup_time']:>10.2f}")
    print(f"  {'Benchmark time (s)':<25} {eager['bench_time']:>10.2f} {compiled['bench_time']:>10.2f} {eager['bench_time']/compiled['bench_time']:>7.2f}x")
    print(f"  {'Throughput (tok/s)':<25} {eager['throughput']:>10.2f} {compiled['throughput']:>10.2f} {speedup:>7.2f}x")
    print(f"  {'Token latency (ms)':<25} {eager['latency_ms']:>10.1f} {compiled['latency_ms']:>10.1f}")
    print(f"{'='*55}")
    print(f"\n  torch.compile speedup: {speedup:.2f}x")
    print(f"  torch.compile overhead (warmup): +{compiled['warmup_time'] - eager['warmup_time']:.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="nano-vllm CPU benchmark")
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num_seqs", type=int, default=4)
    parser.add_argument("--max_input", type=int, default=64)
    parser.add_argument("--max_output", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (eager mode)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both modes and show speedup comparison")
    parser.add_argument("--json-output", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
