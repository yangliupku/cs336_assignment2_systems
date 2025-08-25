from cs336_basics.model import BasicsTransformerLM
import torch
from cs336_basics.nn_utils import cross_entropy
import timeit
import numpy as np


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

    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    lables = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def run():
        logits = model(inputs)
        if enable_backward:
            loss = cross_entropy(logits, lables)
            loss.backward()

    return run


def benchmark(
    run: callable,
    description: str = "benchmarking",
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
    return elapsed_times.mean()


if __name__ == "__main__":
    run = run_basic_lm_model(device="mps")
    t = benchmark(run, device="mps")
    print(t)
