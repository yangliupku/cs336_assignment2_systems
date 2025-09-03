from cs336_basics.model import BasicsTransformerLM
import torch
from cs336_basics.nn_utils import cross_entropy, safe_nvtx_range
import timeit
import numpy as np
from cs336_basics.optimizer import AdamW
from torch.profiler import ProfilerActivity
import pandas as pd
import contextlib


def get_model_specs():
    common_spec = {
        "vocab_size": 10000,
        "context_length": 256,
        "rope_theta": 10000,
    }
    return {
        "small": {
            **common_spec,
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "medium": {
            **common_spec,
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        "large": {
            **common_spec,
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        "xl": {
            **common_spec,
            "d_model": 1600,
            "d_ff": 6400,
            "num_layers": 48,
            "num_heads": 25,
        },
        "2.7B": {
            **common_spec,
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
    }


def get_model_parameter_count(model: torch.nn.Module):
    num_params = 0
    for k, v in model.named_parameters():
        if "embedding" not in k:
            print(k, v.shape, v.numel())
            num_params += v.numel()
        else:
            print(k)
    print("total_params (M)", num_params / 1e6)
    return num_params


def run_basic_lm_model(
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 1024,
    d_ff: int = 4096,
    num_layers: int = 24,
    num_heads: int = 16,
    rope_theta: float = 10000,
    batch_size: int = 4,
    enable_backward: bool = True,
    device: str = None,
) -> callable:
    with safe_nvtx_range("define model"):
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        ).to(device)
    with safe_nvtx_range("define optimizer"):
        opt = AdamW(params=model.parameters())

    with safe_nvtx_range("define data"):
        inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        lables = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def run():
        with safe_nvtx_range("forward"):
            logits = model(inputs)
        if enable_backward:
            opt.zero_grad()
            with safe_nvtx_range("loss"):
                loss = cross_entropy(logits, lables)
            with safe_nvtx_range("backward"):
                loss.backward()
            with safe_nvtx_range("opt step"):
                opt.step()

    return run


def benchmark(
    run: callable,
    warmup_steps: int = 3,
    num_trials: int = 10,
    device: str = "cuda",
    mixed_precision_dtype: torch.dtype = None,
    enable_cuda_memory_profile: bool = False,
    cuda_memory_profile_output_fname: str = "memory_snapshot.pickle",
):
    if device == "cuda":
        assert torch.cuda.is_available()

    mixed_precision_context = (
        torch.autocast(device_type=device, dtype=mixed_precision_dtype)
        if mixed_precision_dtype
        else contextlib.nullcontext()
    )
    should_enable_cuda_memory_profile = enable_cuda_memory_profile and (device == "cuda")
    try:
        with safe_nvtx_range("warm up"):
            for i in range(warmup_steps):
                with mixed_precision_context:
                    run()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_times = []
        if should_enable_cuda_memory_profile:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        for i in range(num_trials):
            start = timeit.default_timer()
            with safe_nvtx_range(f"trail_{i}"):
                with mixed_precision_context:
                    run()

            if device == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            elapsed_times.append((end - start) * 1000)
        if should_enable_cuda_memory_profile:
            torch.cuda.memory._dump_snapshot(filename=cuda_memory_profile_output_fname)
            torch.cuda.memory._record_memory_history(enabled=None)
        elapsed_times = np.array(elapsed_times)
        return elapsed_times.mean(), elapsed_times.std()
    except torch.cuda.OutOfMemoryError:
        return -1, -1


def benchmark_basic_lm_model():
    specs = get_model_specs()
    context_lengths = [128, 256, 512]
    device = "cpu"
    results = []
    for k, spec in specs.items():
        if k != "2.7B":
            continue
        for context_length in context_lengths:
            for mixed_precision_dtype in [None, torch.bfloat16]:
                res = {
                    "model size": k,
                    "context length": context_length,
                    "mix precision": mixed_precision_dtype or "NA",
                }
                spec["context_length"] = context_length
                run = run_basic_lm_model(**spec, device=device, enable_backward=False)
                t = benchmark(
                    run,
                    device=device,
                    mixed_precision_dtype=mixed_precision_dtype,
                    enable_cuda_memory_profile=True,
                    cuda_memory_profile_output_fname=f"memory_snapshot_2p7B_{context_length}_{'with' if mixed_precision_dtype else 'wo'}mp.pickle",
                )
                res["forward time"] = t[0]
                run = run_basic_lm_model(**spec, device=device, enable_backward=True)
                t = benchmark(
                    run,
                    device=device,
                    mixed_precision_dtype=mixed_precision_dtype,
                    enable_cuda_memory_profile=True,
                    cuda_memory_profile_output_fname=f"memory_snapshot_2p7B_{context_length}_{'with' if mixed_precision_dtype else 'wo'}mp.pickle",
                )
                res["forward + backward time"] = t[0]
                results.append(res)
                print(res)
    return pd.DataFrame(results)


def cuda_profile(run: callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Run the code with the profiler
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # Output stack trace for visualization
        with_stack=with_stack,
        # Needed to export stack trace for visualization
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Print out table
    table = prof.key_averages().table(
        sort_by="cuda_time_total", max_name_column_width=80, row_limit=25
    )
    return table


def profile_basic_lm_model():
    device = "cpu"
    spec = get_model_specs()["medium"]
    run = run_basic_lm_model(**spec, device=device, enable_backward=False)
    # run cuda profiler
    table = cuda_profile(run)
    print(table)


if __name__ == "__main__":
    df = benchmark_basic_lm_model()
    print(df.to_markdown())
