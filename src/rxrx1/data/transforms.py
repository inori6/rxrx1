import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
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



class GaussianNoise:
    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        noise = torch.randn_like(image) * self.std
        return image + noise


class GlobalIntensityScale:
    def __init__(
        self,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        p: float = 0.5,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        scale = random.uniform(self.scale_min, self.scale_max)
        return image * scale


class ChannelIntensityScale:
    def __init__(
        self,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        p: float = 0.5,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        scales = torch.empty(
            image.shape[0],
            1,
            1,
            device=image.device,
            dtype=image.dtype,
        ).uniform_(self.scale_min, self.scale_max)

        return image * scales


class RandomGamma:
    def __init__(
        self,
        gamma_min: float = 0.9,
        gamma_max: float = 1.1,
        p: float = 0.5,
    ):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        gamma = random.uniform(self.gamma_min, self.gamma_max)

        image_min = image.amin()
        image_max = image.amax()

        if image_max <= image_min:
            return image

        normalized = (image - image_min) / (image_max - image_min)
        normalized = normalized.pow(gamma)

        return normalized * (image_max - image_min) + image_min


class ChannelDropout:
    def __init__(
        self,
        p: float = 0.2,
        max_channels: int = 1,
    ):
        self.p = p
        self.max_channels = max_channels

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return image

        image = image.clone()

        num_channels = random.randint(
            1,
            min(self.max_channels, image.shape[0]),
        )

        channels = random.sample(
            range(image.shape[0]),
            num_channels,
        )

        image[channels] = 0

        return image


def build_transform(transform_config: list[dict]):
    pipeline = []

    for config in transform_config:
        name = config["name"]

        if name == "resize":
            size = config["size"]
            pipeline.append(
                transforms.Resize((size, size))
            )

        elif name == "horizontal_flip":
            pipeline.append(
                transforms.RandomHorizontalFlip(
                    p=config.get("p", 0.5)
                )
            )

        elif name == "vertical_flip":
            pipeline.append(
                transforms.RandomVerticalFlip(
                    p=config.get("p", 0.5)
                )
            )

        elif name == "rotation":
            pipeline.append(
                transforms.RandomRotation(
                    degrees=config.get("degrees", 15)
                )
            )

        elif name == "crop":
            pipeline.append(
                transforms.RandomResizedCrop(
                    size=config["size"],
                    scale=tuple(
                        config.get("scale", [0.8, 1.0])
                    ),
                    ratio=tuple(
                        config.get("ratio", [0.9, 1.1])
                    ),
                )
            )

        elif name == "affine":
            pipeline.append(
                transforms.RandomAffine(
                    degrees=config.get("degrees", 10),
                    translate=tuple(
                        config.get("translate", [0.05, 0.05])
                    ),
                    scale=tuple(
                        config.get("scale", [0.95, 1.05])
                    ),
                    shear=config.get("shear", 5),
                )
            )

        elif name == "global_intensity_scale":
            pipeline.append(
                GlobalIntensityScale(
                    scale_min=config.get("scale_min", 0.9),
                    scale_max=config.get("scale_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "channel_intensity_scale":
            pipeline.append(
                ChannelIntensityScale(
                    scale_min=config.get("scale_min", 0.9),
                    scale_max=config.get("scale_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "gamma":
            pipeline.append(
                RandomGamma(
                    gamma_min=config.get("gamma_min", 0.9),
                    gamma_max=config.get("gamma_max", 1.1),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "gaussian_noise":
            pipeline.append(
                GaussianNoise(
                    std=config.get("std", 0.01),
                    p=config.get("p", 0.5),
                )
            )

        elif name == "gaussian_blur":
            pipeline.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=config.get(
                                "kernel_size",
                                3,
                            ),
                            sigma=tuple(
                                config.get(
                                    "sigma",
                                    [0.1, 1.0],
                                )
                            ),
                        )
                    ],
                    p=config.get("p", 0.5),
                )
            )

        elif name == "channel_dropout":
            pipeline.append(
                ChannelDropout(
                    p=config.get("p", 0.2),
                    max_channels=config.get(
                        "max_channels",
                        1,
                    ),
                )
            )

        elif name == "random_erasing":
            pipeline.append(
                transforms.RandomErasing(
                    p=config.get("p", 0.25),
                    scale=tuple(
                        config.get("scale", [0.02, 0.15])
                    ),
                    ratio=tuple(
                        config.get("ratio", [0.5, 2.0])
                    ),
                    value=config.get("value", 0),
                )
            )

        elif name == "normalize":
            pipeline.append(
                transforms.Normalize(
                    mean=config["mean"],
                    std=config["std"],
                )
            )

        else:
            raise ValueError(
                f"Unknown transform: {name}"
            )

    return transforms.Compose(pipeline)