import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def evaluate_language_model(
    model,
    data_loader,
    device,
    max_batches=None,
    amp=False,
):
    if len(data_loader) == 0:
        raise ValueError("data_loader must contain at least one batch")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero")

    amp_enabled = amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    was_training = model.training
    model.eval()
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    batches = 0

    try:
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(data_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                context = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with context:
                    logits = model(x)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        y.reshape(-1),
                        reduction="sum",
                    )
                total_negative_log_likelihood += loss_sum.item()
                total_tokens += y.numel()
                batches += 1
    finally:
        model.train(was_training)

    loss = total_negative_log_likelihood / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "tokens": total_tokens,
        "batches": batches,
    }
