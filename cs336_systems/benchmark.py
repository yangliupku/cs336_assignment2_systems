import pstats
from cs336_basics.model import BasicsTransformerLM
import torch
from cs336_basics.nn_utils import cross_entropy
import timeit
import numpy as np
from cs336_basics.optimizer import AdamW
from torch.profiler import ProfilerActivity
import cProfile


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
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)
    opt = AdamW(params=model.parameters())

    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    lables = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def run():
        logits = model(inputs)
        if enable_backward:
            opt.zero_grad()
            loss = cross_entropy(logits, lables)
            loss.backward()
            opt.step()

    return run


def benchmark(
    run: callable,
    warmup_steps: int = 3,
    num_trials: int = 10,
    device: str = "cuda",
):
    if device == "cuda":
        assert torch.cuda.is_available()
    for i in range(warmup_steps):
        run()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_times = []
    for i in range(num_trials):
        start = timeit.default_timer()
        run()
        if device == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        elapsed_times.append((end - start) * 1000)
    elapsed_times = np.array(elapsed_times)
    return elapsed_times.mean(), elapsed_times.std()


def benchmark_basic_lm_model():
    specs = get_model_specs()
    device = "mps"
    for k, spec in specs.items():
        print(f"-------{k}-------")
        run = run_basic_lm_model(**spec, device=device, enable_backward=False)
        t = benchmark(run, device=device)
        print("forward", t[0])
        run = run_basic_lm_model(**spec, device=device, enable_backward=True)
        t = benchmark(run, device=device)
        print("forward-backward", t[0])


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
    profile_basic_lm_model()
