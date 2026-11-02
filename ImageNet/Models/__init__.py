"""Original project ImageNet models; parameter names and topology preserved."""
from . import ResNet, MobileNetv1

MODEL_NAMES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "mobilenetv1")


def build_model(name, num_classes=1000):
    if num_classes != 1000 or name not in MODEL_NAMES:
        raise ValueError("Use an ImageNet-1k model from " + str(MODEL_NAMES))
    module = MobileNetv1 if name == "mobilenetv1" else ResNet
    return getattr(module, name)()[0]
