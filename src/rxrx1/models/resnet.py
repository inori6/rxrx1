import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_BUILDERS = {"resnet18": resnet18}
_WEIGHTS = {"resnet18": ResNet18_Weights.DEFAULT}


def _replace_input_conv(model, pretrained=True):
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.repeat(1, 2, 1, 1) / 2)
    model.conv1 = new_conv


def build_resnet(name="resnet18", num_classes=1108, pretrained=True):
    if name not in _BUILDERS:
        raise ValueError(f"Unsupported ResNet: {name}")

    weights = _WEIGHTS[name] if pretrained else None
    model = _BUILDERS[name](weights=weights)
    _replace_input_conv(model, pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = build_resnet("resnet18")
    print(model)