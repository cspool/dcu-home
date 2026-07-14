import json
import math
import os
import statistics
import time

import torch

from vllm import _custom_ops as ops


K = 5120
SHAPES = (14336, 16384, 34816)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 20260712)
RESULT_DIR = os.environ.get(
    "RESULT_DIR",
    "/public/home/tangyu408/testdata/goal_runs/20260712_h10_8_runtime_validation",
)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def run_pair(weight: torch.Tensor, x: torch.Tensor):
    y320 = ops.LLMM1Strided(weight, x, 4, 320)
    y640 = ops.LLMM1Strided(weight, x, 4, 640)
    torch.cuda.synchronize()
    return y320, y640


def event_bench(weight: torch.Tensor, x: torch.Tensor, threads: int, calls: int):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        ops.LLMM1Strided(weight, x, 4, threads)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / calls


def expect_reject(name, fn, output):
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        output[name] = {"rejected": True, "message": str(exc).splitlines()[0]}
        return
    output[name] = {"rejected": False, "message": "unexpected success"}
    raise AssertionError(f"negative case unexpectedly succeeded: {name}")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    results = {
        "device": torch.cuda.get_device_name(0),
        "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "shapes": {},
        "negative_cases": {},
        "special_values": {},
    }

    for m in SHAPES:
        shape_result = {"seeds": [], "benchmark": {}}
        print(f"correctness M={m}", flush=True)
        for seed in SEEDS:
            torch.manual_seed(seed)
            weight = torch.randn((m, K), device=device, dtype=torch.bfloat16)
            x = torch.randn((1, K), device=device, dtype=torch.bfloat16)
            y320, y640 = run_pair(weight, x)
            y640_repeat = ops.LLMM1Strided(weight, x, 4, 640)
            reference = torch.nn.functional.linear(x, weight)
            torch.cuda.synchronize()
            diff = (y640.float() - reference.float()).abs()
            seed_result = {
                "seed": seed,
                "bitwise_vs_t320": bitwise_equal(y320, y640),
                "repeat_bitwise": bitwise_equal(y640, y640_repeat),
                "torch_max_abs": float(diff.max().item()),
                "torch_mean_abs": float(diff.mean().item()),
            }
            if not seed_result["bitwise_vs_t320"] or not seed_result["repeat_bitwise"]:
                raise AssertionError((m, seed, seed_result))
            shape_result["seeds"].append(seed_result)
            del weight, x, y320, y640, y640_repeat, reference, diff
            torch.cuda.empty_cache()

        torch.manual_seed(1000 + m)
        weight = torch.randn((m, K), device=device, dtype=torch.bfloat16)
        x = torch.randn((1, K), device=device, dtype=torch.bfloat16)
        for threads in (320, 640):
            for _ in range(100):
                ops.LLMM1Strided(weight, x, 4, threads)
        torch.cuda.synchronize()
        t320 = []
        t640 = []
        for group in range(31):
            order = (320, 640) if group % 2 == 0 else (640, 320)
            measured = {threads: event_bench(weight, x, threads, 50) for threads in order}
            t320.append(measured[320])
            t640.append(measured[640])
        median320 = statistics.median(t320)
        median640 = statistics.median(t640)
        improvement = (median320 / median640 - 1.0) * 100.0
        shape_result["benchmark"] = {
            "groups": 31,
            "calls_per_group": 50,
            "t320_us": t320,
            "t640_us": t640,
            "t320_median_us": median320,
            "t640_median_us": median640,
            "improvement_percent": improvement,
        }
        if improvement < 5.0:
            raise AssertionError(f"M={m} improvement {improvement:.3f}% < 5%")
        results["shapes"][str(m)] = shape_result
        print(
            f"benchmark M={m}: {median320:.3f} -> {median640:.3f} us "
            f"({improvement:.3f}%)",
            flush=True,
        )
        del weight, x
        torch.cuda.empty_cache()

    weight = torch.randn((14336, K), device=device, dtype=torch.bfloat16)
    x = torch.randn((1, K), device=device, dtype=torch.bfloat16)
    neg = results["negative_cases"]
    expect_reject(
        "fp16",
        lambda: ops.LLMM1Strided(weight.half(), x.half(), 4, 640),
        neg,
    )
    bad_k_weight = torch.randn((14336, K - 8), device=device, dtype=torch.bfloat16)
    bad_k_x = torch.randn((1, K - 8), device=device, dtype=torch.bfloat16)
    expect_reject(
        "k_not_5120",
        lambda: ops.LLMM1Strided(bad_k_weight, bad_k_x, 4, 640),
        neg,
    )
    bad_m_weight = torch.randn((1024, K), device=device, dtype=torch.bfloat16)
    expect_reject(
        "unsupported_m",
        lambda: ops.LLMM1Strided(bad_m_weight, x, 4, 640),
        neg,
    )
    nondiv_weight = torch.randn((14335, K), device=device, dtype=torch.bfloat16)
    expect_reject(
        "m_not_divisible_by_4",
        lambda: ops.LLMM1Strided(nondiv_weight, x, 4, 640),
        neg,
    )
    x_noncontiguous = torch.randn((1, K * 2), device=device, dtype=torch.bfloat16)[:, ::2]
    expect_reject(
        "noncontiguous_activation",
        lambda: ops.LLMM1Strided(weight, x_noncontiguous, 4, 640),
        neg,
    )
    weight_noncontiguous = torch.randn(
        (K, 14336), device=device, dtype=torch.bfloat16
    ).t()
    expect_reject(
        "noncontiguous_weight",
        lambda: ops.LLMM1Strided(weight_noncontiguous, x, 4, 640),
        neg,
    )
    x_two_rows = torch.randn((2, K), device=device, dtype=torch.bfloat16)
    expect_reject(
        "n_not_1",
        lambda: ops.LLMM1Strided(weight, x_two_rows, 4, 640),
        neg,
    )
    x_misaligned = torch.randn((1, K + 1), device=device, dtype=torch.bfloat16)[:, 1:]
    assert x_misaligned.is_contiguous()
    expect_reject(
        "misaligned_activation",
        lambda: ops.LLMM1Strided(weight, x_misaligned, 4, 640),
        neg,
    )

    zero_weight = torch.zeros((14336, K), device=device, dtype=torch.bfloat16)
    zero_x = torch.zeros((1, K), device=device, dtype=torch.bfloat16)
    zero320, zero640 = run_pair(zero_weight, zero_x)
    results["special_values"]["zeros_bitwise"] = bitwise_equal(zero320, zero640)
    if not results["special_values"]["zeros_bitwise"]:
        raise AssertionError("zero case differs")

    special_weight = torch.zeros((14336, K), device=device, dtype=torch.bfloat16)
    special_x = torch.ones((1, K), device=device, dtype=torch.bfloat16)
    special_weight[0, 0] = math.nan
    special_weight[1, 0] = math.inf
    special_weight[2, 0] = -math.inf
    special320, special640 = run_pair(special_weight, special_x)
    special_match = torch.equal(torch.isnan(special320), torch.isnan(special640)) and torch.equal(
        torch.isinf(special320), torch.isinf(special640)
    )
    results["special_values"]["nan_inf_classification_match"] = special_match
    if not special_match:
        raise AssertionError("NaN/Inf classification differs")

    results["all_passed"] = True
    result_path = os.path.join(RESULT_DIR, "validation_results.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"all_passed result={result_path}", flush=True)


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"elapsed_seconds={time.time() - started:.3f}", flush=True)
