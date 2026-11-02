"""Checkpoint-compatible CIFAR CNN registry."""
from importlib import import_module

_MODULES = {**{name: "ResNet" for name in ("resnet8", "resnet20", "resnet32", "resnet56", "resnet110", "resnet8x4", "resnet32x4")},
            **{name: "WRN" for name in ("wrn_40_2", "wrn_40_1", "wrn_16_2", "wrn_16_1")},
            **{name: "VGG" for name in ("vgg8", "vgg11", "vgg13", "vgg16")},
            "ResNet50": "ResNet50", "ShuffleV1": "ShuffleNetV1", "ShuffleV2": "ShuffleNetV2", "mobilenetv1": "MobileNetv1"}
MODEL_NAMES = tuple(_MODULES)


def build_model(name, num_classes=100):
    if name not in _MODULES:
        raise ValueError(f"Unknown CIFAR model {name}; choices: {MODEL_NAMES}")
    return getattr(import_module("." + _MODULES[name], __name__), name)(num_classes=num_classes)[0]
