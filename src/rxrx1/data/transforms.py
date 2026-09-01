import torch
import torch.nn.functional as F


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    image = image.unsqueeze(0)
    image = F.interpolate(
        image,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return image.squeeze(0)