import torch


def d4_views(x):
    for k in range(4):
        rotated = torch.rot90(x, k, dims=(-2, -1))
        yield rotated
        yield torch.flip(rotated, dims=(-1,))


def tta_logits(model, images, metadata=None, mode="d4"):
    if mode != "d4":
        raise ValueError(f"Unsupported TTA mode: {mode}")

    logits_sum = None
    n = 0

    for view in d4_views(images):
        logits = model(view, metadata=metadata)
        if isinstance(logits, tuple):
            logits = logits[0]

        logits_sum = logits if logits_sum is None else logits_sum + logits
        n += 1

    return logits_sum / n