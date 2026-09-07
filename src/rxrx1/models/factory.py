from rxrx1.models.efficientnet import build_efficientnet
from rxrx1.models.efficientnet_concat_neck import build_efficientnet_concat_neck
from rxrx1.models.first_place_densenet import build_first_place_densenet
from rxrx1.models.first_place_efficientnet import build_first_place_efficientnet
from rxrx1.models.efficientnet_metric_neck import (
    build_efficientnet_metric_neck,
)

def build_model(model_config, num_classes, metric=None):
    name = model_config["name"]

    if name in {"efficientnet_b2", "efficientnet_b4"}:
        return build_efficientnet(
            name=name,
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
            dropout=model_config.get("dropout"),
            metadata=model_config.get("metadata"),
            metric=metric,
        )

    if name == "first_place_densenet161":
        return build_first_place_densenet(
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
            embedding_size=model_config.get("embedding_size", 1024),
            bn_momentum=model_config.get("bn_momentum", 0.05),
            num_cell_types=model_config.get("num_cell_types", 4),
        )

    if name == "first_place_efficientnet_b2":
        return build_first_place_efficientnet(
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
            embedding_size=model_config.get("embedding_size", 1024),
            bn_momentum=model_config.get("bn_momentum", 0.05),
            num_cell_types=model_config.get("num_cell_types", 4),
        )

    if name == "efficientnet_b2_concat_neck":
        return build_efficientnet_concat_neck(
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
            embedding_size=model_config.get("embedding_size", 1024),
            bn_momentum=model_config.get("bn_momentum", 0.05),
            metadata=model_config.get("metadata"),
        )

    if name == "efficientnet_b2_metric_neck":
        return build_efficientnet_metric_neck(
            num_classes=num_classes,
            pretrained=model_config.get("pretrained", True),
            dropout=model_config.get("dropout", 0.22),
            embedding_size=model_config.get("embedding_size", 1024),
            bn_momentum=model_config.get("bn_momentum", 0.05),
            metric=metric,
        )

    raise ValueError(f"Unsupported model: {name}")
